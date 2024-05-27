import os
import time
import json
import argparse
import lightning as L
import torch
import transformers
import random
from PIL import Image
from torch.utils.data import Dataset

import sys
from pathlib import Path
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from vision_llama.model import LLaMA, VisionLLaMA, LLaMAConfig
from vision_llama.lora import mark_only_lora_as_trainable, lora
from vision_llama.tokenizer import Tokenizer
from torch.utils.data import DataLoader
from lightning.fabric.strategies import DeepSpeedStrategy
from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

from torchvision.transforms import (
    Compose,
    Normalize,
    RandomResizedCrop,
    ToTensor,
)

random.seed(10)

IGNORE_INDEX = -1
PAD_INDEX = 0
BOS_INDEX = 1
EOS_INDEX = 2

TEMPLATE_DICT = {
    'intro':(
    "\n### You are a helpful language and vision assistant."
    "\n### You are able to understand reference and target images user provides, and discriminates the visual difference between them."
    "\n### You should follow the instruction carefully and explain your answers clearly."
    ),
    'reference': (
        "<Reference image>"
    ),
    'target': (
        "<Target image>"
    ),
}


class CIRData(Dataset):
    def __init__(self, data_list, image_processor, tokenizer):
        self.data_list = data_list

        normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
        self.image_transform = Compose(
        [
            RandomResizedCrop((image_processor.size["height"], image_processor.size["width"]), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            ToTensor(),
            normalize,
        ]
    )

        self.tokenizer = tokenizer
        self.TD = TEMPLATE_DICT

    def __len__(self):
        return len(self.data_list)
    
    def check_decode(self, sample):
        sample = sample[sample!=-1]
        sample = sample[sample!=0]
        return self.tokenizer.decode(sample.cpu())
    
    def tokenize(self, string: str, bos=False, eos=False) -> torch.Tensor:
        return self.tokenizer.encode(string, bos=bos, eos=eos)

    def prepare_image(self, image):
        image = Image.open(image)
        if image.getbands()[0] != 'R':
            image = image.convert('RGB')
        return self.image_transform(image)

    def prepare_text(self, encoded_full_prompt, encoded_full_prompt_and_response, mask_inputs: bool = True):

        labels = encoded_full_prompt_and_response.clone()
        if mask_inputs:
            labels[:len(encoded_full_prompt)] = IGNORE_INDEX
        return {"input_ids": encoded_full_prompt_and_response, "labels": labels}

    def sample_one(self, image_info):
        image_ref = self.prepare_image(image_info['reference'])
        image_tar = self.prepare_image(image_info['target'])

        caption = image_info['caption']

        # Tokenizing parts of the template
        intro_tokens = self.tokenize("\n### You are a helpful language and vision assistant."
                                    "\n### You are able to understand reference and target images user provides."
                                    "\n### You should follow the instruction carefully and explain your answers clearly."
                                    "\n### Request:Analyze the given reference and target images and provide a description that transforms the reference to match the target.", bos=True, eos=False)
        reference_intro_tokens = self.tokenize(f"\n### Reference:", bos=False, eos=False)
        target_intro_tokens = self.tokenize(f"\n### Target:", bos=False, eos=False)
        response_tokens = self.tokenize("\n### Response:", bos=False, eos=False)

        # Calculate positions for the image placeholders based on token lengths
        image_placeholder_position_ref = len(intro_tokens) + len(reference_intro_tokens)
        image_placeholder_position_tar = len(intro_tokens) + len(reference_intro_tokens) + 32 + len(target_intro_tokens) # 32 is the number of PAD tokens you'll insert for the reference image

        # Insert PAD tokens at the appropriate positions for images
        template_tokens = torch.cat([
            intro_tokens,
            reference_intro_tokens,
            torch.tensor([PAD_INDEX]*32),
            target_intro_tokens,
            torch.tensor([PAD_INDEX]*32),
            response_tokens,
        ])

        template_tokens_with_response = torch.cat([
            intro_tokens,
            reference_intro_tokens,
            torch.tensor([PAD_INDEX]*32),
            target_intro_tokens,
            torch.tensor([PAD_INDEX]*32),
            response_tokens,
            self.tokenize(caption, bos=False, eos=True)
        ])

        # Prepare labels by masking the prompt
        language_dict = self.prepare_text(template_tokens, template_tokens_with_response)

        # Return the sample dictionary
        return dict(image_ref=image_ref, 
                    image_tar=image_tar, 
                    input_ids=language_dict['input_ids'], 
                    labels=language_dict['labels'], 
                    image_positions=[image_placeholder_position_ref, image_placeholder_position_tar])
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_info = self.data_list[idx]
        return self.sample_one(image_info)

class DataCollatorImage(object):
    def __call__(self, instances):
        image_ref, image_tar, input_ids, labels, image_positions = tuple([instance[key] for instance in instances] for key in ("image_ref", "image_tar", "input_ids", "labels", "image_positions"))
        
        image_ref = torch.stack(image_ref).bfloat16()
        image_tar = torch.stack(image_tar).bfloat16()
        
        # Pad input sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=PAD_INDEX)
        
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        return dict(image_ref=image_ref, image_tar=image_tar, input_ids=input_ids, labels=labels, image_positions=image_positions)


num_query_tokens = 32

lora_alpha = 16
lora_dropout = 0.05

# Hyperparameters
eval_interval = 100
save_interval = 10000
log_interval = 1

learning_rate = 2e-4

devices = 8
batch_size = 128 // devices
micro_batch_size = 8

gradient_accumulation_steps = batch_size // micro_batch_size
weight_decay = 0.02
block_size = 2048
warmup_steps = 100
lr_steps = 100000 // micro_batch_size // devices
num_workers = 10

ds_config = {
    "train_micro_batch_size_per_gpu": micro_batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "zero_optimization": {"stage": 2},
}

def main(args):
    fabric = L.Fabric(
        accelerator="cuda", 
        devices=devices, 
        strategy=DeepSpeedStrategy(config=ds_config), 
        precision="bf16"
    )
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    
    InstBLIPpretrained = f'Salesforce/{args.qf_model}'

    if fabric.global_rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)

    config = LLaMAConfig().from_name(args.llama_size)
    hidden_size = config.n_embd
    config.block_size = block_size

    checkpoint = torch.load(os.path.join(args.llama_path,args.llama_type,args.llama_size,'state_dict.pth'))
    tokenizer = Tokenizer(os.path.join(args.llama_path,args.llama_type,'tokenizer.model'))

    train_transform = transformers.AutoImageProcessor.from_pretrained(InstBLIPpretrained)

    with open(args.dataset) as f:
        train_list = json.loads(f.read())
    
    train_dataset = CIRData(data_list=train_list, image_processor=train_transform, tokenizer=tokenizer)
    
    train_loader = DataLoader(train_dataset, num_workers=num_workers, batch_size=micro_batch_size, shuffle=True, collate_fn=DataCollatorImage())
    train_loader = fabric.setup_dataloaders(train_loader)

    with fabric.device, lora(r=args.lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
        language_model = LLaMA(config).bfloat16()
        language_model.load_state_dict(checkpoint, strict=False)

        mark_only_lora_as_trainable(language_model)
        IBLIP = transformers.InstructBlipForConditionalGeneration.from_pretrained(InstBLIPpretrained, torch_dtype=torch.bfloat16)

        for param in IBLIP.parameters():
            param.requires_grad = False

        vision_model = IBLIP.vision_model.eval()
        qformer = IBLIP.qformer
        query_tokens = IBLIP.query_tokens
        language_projection = torch.nn.Linear(768, hidden_size, bias=False)

        model = VisionLLaMA(
                        vision_model=vision_model,
                        qformer=qformer,
                        query_tokens=query_tokens,
                        language_projection=language_projection,
                        language_model=language_model,
                        tokenizer=tokenizer)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model, optimizer = fabric.setup(model.bfloat16(), optimizer)

    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"Number of trainable parameters: {num_params}")

    train(fabric, model, optimizer, train_loader, args)

def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    args
) -> None:
        
    step_count = 0
    iter_num = 0

    total_iter = len(train_loader) * args.num_epochs
    
    for epoch in range(args.num_epochs):
        for batch in train_loader:
            t0 = time.time()
            
            image_ref, image_tar, input_ids, labels, image_positions = batch.get('image_ref'), batch.get('image_tar'), batch.get('input_ids'), batch.get('labels'), batch.get('image_positions')
            logits = model(image_ref, image_tar, input_ids, image_positions)
            
            loss = loss_fn(logits, labels.long())

            if step_count <= warmup_steps:
                # linear warmup
                lr = learning_rate * step_count / warmup_steps
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            if iter_num ==  total_iter // 2:
                # lr scaling
                lr = learning_rate * 0.1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            with fabric.no_backward_sync(model, enabled=((iter_num + 1) % gradient_accumulation_steps != 0)):
                fabric.backward(loss / gradient_accumulation_steps)

            if (iter_num + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                step_count += 1

            dt = time.time() - t0
            if iter_num % log_interval == 0:
                fabric.print(f"epoch: {epoch} ({(iter_num/total_iter)*100:.4f}%) loss {loss.item():.4f}, time: {dt*1000:.2f}ms, lr: {param_group['lr']:.6f}")
            iter_num += 1

        if epoch+1==args.num_epochs:
            save_path = os.path.join(args.out_dir, f"cirr_lora_{args.llama_type}_{args.llama_size}_{args.qf_model}_{args.lora_r}_{epoch+1}")
            fabric.save(save_path, {"model": model})
            fabric.barrier()
            if fabric.global_rank == 0:
                # Create a consolidated checkpoint with the same name next to the deepspeed checkpoint
                convert_zero_checkpoint_to_fp32_state_dict(save_path, save_path+'.pth')


def loss_fn(logits, targets):
    logits = logits[:, -targets.size(1) :, :]
    # shift the targets such that output n predicts token n+1
    logits = logits[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    return loss

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--qf_model", type=str, default="instructblip-vicuna-13b")
    parser.add_argument("--out_dir", type=str, default="LLMs")
    parser.add_argument("--dataset", type=str, default="datasets/cirr_train.json")
    parser.add_argument("--llama_size", type=str, default="13B")
    parser.add_argument("--llama_type", type=str, default="lit-llama2")
    parser.add_argument("--llama_path", type=str, default="", help="path to llama checkpoint")
    args = parser.parse_args()

    main(args)
