import json
import argparse
import sys
import os

from pathlib import Path
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lightning as L
import torch
import transformers

from vision_llama.generate import generate_from_image, truncate_output_to_eos
from vision_llama.model import LLaMA, VisionLLaMA, LLaMAConfig
from vision_llama.lora import mark_only_lora_as_trainable, lora
from vision_llama.tokenizer import Tokenizer
from torch.utils.data import DataLoader, Subset
from generate.datasets import CIRDataTest, DataCollatorImageTest

# Hyperparameters
lora_alpha = 16
lora_dropout = 0.0
max_new_tokens = 32
temperature = 0.2
top_k = 50
micro_batch_size = 64
devices = 1
block_size = 2048
num_workers = 10

ds_config = {
    "train_micro_batch_size_per_gpu": micro_batch_size,
    "zero_optimization": {"stage": 2},
}   

@torch.no_grad()
def testCIR(fabric: L.Fabric, tokenizer, model: torch.nn.Module, val_loader, data_json, save_name) -> torch.Tensor:
    fabric.print("Testing ...", flush=True)

    with open(save_name, 'w') as f:
        for batch in val_loader:
            image_ref, image_tar, input_ids, image_idx, image_positions = batch.get('image_ref'), batch.get('image_tar'), batch.get('input_ids'), batch.get('image_idx'), batch.get('image_positions')

            if 'caption' in data_json[0].keys():

                labels = []
                references = []
                targets = []
                for idx in image_idx:
                    labels += [data_json[idx]['caption']]
                    references += [data_json[idx]['reference']]
                    targets += [data_json[idx]['target']]
            
                outputs = compute_score(fabric, tokenizer, model, image_ref.bfloat16(), image_tar.bfloat16(), input_ids, image_positions, references, targets)

                for i in range(len(outputs)):
                    towrite = "ref:%s|tar:%s|output:%s|label:%s\n"%(references[i],targets[i],outputs[i],labels[i])
                    f.write(towrite)
                
            else:
                references = []
                targets = []
                for idx in image_idx:
                    references += [data_json[idx]['reference']]
                    targets += [data_json[idx]['target']]
                
                outputs = compute_score(fabric, tokenizer, model, image_ref.bfloat16(), image_tar.bfloat16(), input_ids, image_positions, references, targets)

                for i in range(len(outputs)):
                    towrite = "ref:%s|tar:%s|output:%s\n"%(references[i],targets[i],outputs[i])
                    f.write(towrite)

    fabric.print("Done!", flush=True)


def compute_score(fabric, tokenizer, model, image_ref, image_tar, idx, image_positions, references, targets):
    outputs = generate_from_image(
        model,
        image_ref=image_ref,
        image_tar=image_tar,
        image_positions=image_positions,
        idx=idx,
        max_seq_length=block_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    outputs = outputs.cpu()

    output_all = []
    
    for i, output in enumerate(outputs):
        output = truncate_output_to_eos(output, tokenizer.eos_id)
        output = output[output!=0]
        output = tokenizer.decode(output)
        output = output.split('### Response:')[-1].split(':')[-1]
        output_all.append(output.replace('\n', ''))
        fabric.print(references[i], targets[i], "Output:", output, flush=True)

    return output_all

def main(args):
  
    InstBLIPpretrained = "Salesforce/"+args.qf_model
    
    fabric = L.Fabric(
        accelerator="cuda", 
        devices=devices,
        precision="bf16"
    )
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    config = LLaMAConfig().from_name(args.llama_size)
    hidden_size = config.n_embd
    config.block_size = block_size

    checkpoint = torch.load(os.path.join(args.llama_path,args.llama_type,args.llama_size,'state_dict.pth'))
    tokenizer = Tokenizer(os.path.join(args.llama_path,args.llama_type,'tokenizer.model'))

    test_transform = transformers.AutoImageProcessor.from_pretrained(InstBLIPpretrained)

    with open(args.dataset) as f:
        val_list = json.loads(f.read())
    save_path = f'./{os.path.splitext(os.path.basename(args.dataset))[0]}_{os.path.splitext(os.path.basename(args.ckpt))[0]}.txt'
    val_dataset = CIRDataTest(data_list=val_list, image_processor=test_transform, tokenizer=tokenizer)
    ckpt_dir = f'{args.ckpt}'
    
    val_loader = DataLoader(val_dataset, 
                            num_workers=num_workers, 
                            batch_size=micro_batch_size, 
                            shuffle=False, 
                            drop_last=False, 
                            collate_fn=DataCollatorImageTest())
    val_loader = fabric.setup_dataloaders(val_loader)

    with fabric.device, lora(r=args.lora_r, alpha=lora_alpha, dropout=lora_dropout, enabled=True):
        language_model = LLaMA(config).bfloat16()
        language_model.load_state_dict(checkpoint, strict=False)
        mark_only_lora_as_trainable(language_model)

        IBLIP = transformers.InstructBlipForConditionalGeneration.from_pretrained(InstBLIPpretrained, torch_dtype=torch.bfloat16)

        vision_model = IBLIP.vision_model.eval()
        qformer = IBLIP.qformer
        query_tokens = IBLIP.query_tokens
        language_projection = torch.nn.Linear(768, hidden_size, bias=False)

        del IBLIP
        torch.cuda.empty_cache()

        for param in vision_model.parameters():
            param.requires_grad = False

        model = VisionLLaMA(
                        vision_model=vision_model,
                        qformer=qformer,
                        query_tokens=query_tokens,
                        language_projection=language_projection,
                        language_model=language_model,
                        tokenizer=tokenizer)

    model.load_state_dict(torch.load(ckpt_dir), strict=True)
    model.eval().bfloat16()

    testCIR(fabric, tokenizer, model, val_loader, val_list, save_path)

if __name__ == "__main__":

    torch.set_float32_matmul_precision("high")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="datasets/cirr_train.json")
    parser.add_argument("--llama_size", type=str, default="13B")
    parser.add_argument("--ckpt", type=str, default="", required=True)
    parser.add_argument("--llama_type", type=str, default="lit-llama2")
    parser.add_argument("--llama_path", type=str, default="", help="path to llama checkpoint")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--qf_model", default="instructblip-vicuna-13b", type=str)
    args = parser.parse_args()
    main(args)
