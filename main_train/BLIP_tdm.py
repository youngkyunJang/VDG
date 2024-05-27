import torch
import os
import argparse
import json
import shutil
import math
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
import lightning as L
import transformers
from lightning.fabric.strategies import DeepSpeedStrategy
import torch.nn.functional as F
from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

import einops
import random

import sys
from pathlib import Path
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

class Loader(Dataset):
    def __init__(self, processor, image_list, extra_list=None):
        super().__init__()
        self.image_list = image_list
        self.extra_list = extra_list
        self.processor = processor

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Access image_list
        data = self.image_list[idx]
        image_ref, image_tar, input_ids, text_attention_mask = self.process_data(data)

        # Access extra_list
        if self.extra_list is not None:
            # Create a pseudo-random index based on idx
            extra_idx = random.randrange(len(self.extra_list))
            extra_data = self.extra_list[extra_idx]
            image_ref_e, image_tar_e, input_ids_e, text_attention_mask_e = self.process_data(extra_data)

            return (image_ref, image_tar, input_ids, text_attention_mask), (image_ref_e, image_tar_e, input_ids_e, text_attention_mask_e)
        return image_ref, image_tar, input_ids, text_attention_mask
    
    def process_data(self, data):
        image_ref = data["reference"]
        image_tar = data["target"]

        if isinstance(data["caption"], list):
            text = random.choice(data["caption"])
        else:
            text = data["caption"]

        image_ref = self.processor(images=Image.open(image_ref).convert('RGB'))["pixel_values"][0]
        image_tar = self.processor(images=Image.open(image_tar).convert('RGB'))["pixel_values"][0]

        text_processed = self.processor(text=text, padding="max_length", max_length=32, truncation=True, return_tensors="pt")
        input_ids = text_processed["input_ids"][0]
        text_attention_mask = text_processed["attention_mask"][0]

        return image_ref, image_tar, input_ids, text_attention_mask

class BLIP_CIR(torch.nn.Module):
    def __init__(self, vision_model, text_model, vision_proj, text_proj, tdm_head) -> None:
        super().__init__()
        self.vision_model = vision_model
        self.text_model = text_model
        self.vision_proj = vision_proj
        self.text_proj = text_proj
        self.tdm_head = tdm_head

        self.temp = torch.nn.Parameter(0.07 * torch.ones([]))
        # self.temp = 0.07

    def forward(self, image=None, text=None, text_attention_mask=None, image_embs=None):
        # If only images are provided, we assume we need to perform image forward pass
        if image is not None and text is None:
            image_embs = self.vision_model(image)[0]
            image_feat = F.normalize(self.vision_proj(image_embs[:, 0, :]))
            return image_embs, image_feat

        # If text and image embeddings are provided, we assume a composed forward pass
        elif text is not None and image_embs is not None:
            composed_embs = self.text_model(
                input_ids=text,
                attention_mask=text_attention_mask,
                encoder_hidden_states=image_embs,
                encoder_attention_mask=torch.ones(image_embs.size()[:-1], dtype=torch.long),
            )[0]
            composed_feat = F.normalize(self.text_proj(composed_embs[:, 0, :]))
            return composed_embs, composed_feat

        else:
            raise ValueError("Invalid combination of inputs for forward pass.")

def main(args):

    ds_config = {
    "train_micro_batch_size_per_gpu": args.batch_size,
    "zero_optimization": {"stage": 2},
    }

    fabric = L.Fabric(
        accelerator="cuda", 
        devices=args.world_size,
        strategy=DeepSpeedStrategy(config=ds_config), 
        precision="bf16"
    )
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    CIR_list = []
    save_name = ""
    for file_name in args.dataset:
        with open(file_name, 'r') as f:
            # Here, we assume each file contains a JSON array
            data = json.load(f)
            # Extending the combined list with data from the current file
            CIR_list.extend(data)
        save_name += os.path.splitext(os.path.basename(file_name))[0]
    args.dataset = save_name

    fabric.print(f"Total training samples: {len(CIR_list)}")

    BLIPpretrained = f'Salesforce/{args.model}'

    with fabric.device:
        BLIP = transformers.BlipForImageTextRetrieval.from_pretrained(BLIPpretrained)
        processor = transformers.BlipProcessor.from_pretrained(BLIPpretrained)
        vision_model = BLIP.vision_model
        text_model = BLIP.text_encoder
        vision_proj = BLIP.vision_proj
        text_proj = BLIP.text_proj
        itm_head = BLIP.itm_head

        for param in vision_model.parameters():
            param.requires_grad = False
        for param in vision_proj.parameters():
            param.requires_grad = False
        for param in itm_head.parameters():
            param.requires_grad = False

        model = BLIP_CIR(
                        vision_model=vision_model,
                        text_model=text_model,
                        vision_proj=vision_proj,
                        text_proj=text_proj,
                        tdm_head=itm_head).bfloat16()
    
    if args.extra_dataset:
        extra_list = []
        for file_name in args.extra_dataset:
            with open(file_name, 'r') as f:
                data = json.load(f)
                extra_list.extend(data)
            save_name += os.path.splitext(os.path.basename(file_name))[0]
        args.dataset = save_name

        fabric.print(f"Total extra samples: {len(extra_list)}")
        train_set = Loader(processor, CIR_list, extra_list)
    else:
        train_set = Loader(processor, CIR_list)
        
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        shuffle=True,
    )

    train_loader = fabric.setup_dataloaders(train_loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    model, optimizer = fabric.setup(model, optimizer)

    train(fabric, model, optimizer, train_loader)

def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
) -> None:
    L.seed_everything(1234, workers=True)
    model.train(True)

    iter = 0
    total_iter = len(train_loader) * args.epochs

    from loss import HardNegativeNCE
    criterion = HardNegativeNCE()

    bs = args.batch_size * 2 if args.extra_dataset!=None else args.batch_size
        
    for epoch in range(args.epochs):
        optimizer.zero_grad()

        for data in train_loader:
            # Cosine LR
            lr = (args.init_lr - args.min_lr) * 0.5 * (1.0 + math.cos(math.pi * iter / total_iter)) + args.min_lr
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            if len(data)==2:
                # Unpack data from image_list and extra_list
                (image_ref, image_tar, text, text_attention_mask), (image_ref_e, image_tar_e, text_e, text_attention_mask_e) = data
            else:
                # Unpack data only from image_list
                image_ref, image_tar, text, text_attention_mask = data

            with torch.no_grad():
                model.temp.clamp_(0.001, 0.5)

            ref_emb, _ = model(image=image_ref.bfloat16())
            tar_emb, x_tar = model(image=image_tar.bfloat16())
            _, x_composed = model(image_embs=ref_emb, text=text, text_attention_mask=text_attention_mask)

            x_composed_all = fabric.all_gather(x_composed, sync_grads=True)
            x_tar_all = fabric.all_gather(x_tar, sync_grads=True)
            x_composed_all = einops.rearrange(x_composed_all, "d b e -> (d b) e")
            x_tar_all = einops.rearrange(x_tar_all, "d b e -> (d b) e")

            loss_itc = criterion(x_composed_all, x_tar_all, model.temp)

            if args.extra_dataset!=None:
                image_ref = torch.cat([image_ref, image_ref_e], dim=0)
                image_tar = torch.cat([image_tar, image_tar_e], dim=0)
                text = torch.cat([text, text_e], dim=0)
                text_attention_mask = torch.cat([text_attention_mask, text_attention_mask_e], dim=0)

                ref_emb, _ = model(image=image_ref.bfloat16())
                tar_emb, x_tar = model(image=image_tar.bfloat16())
                _, x_composed = model(image_embs=ref_emb, text=text, text_attention_mask=text_attention_mask)

                x_composed_all = fabric.all_gather(x_composed, sync_grads=True)
                x_tar_all = fabric.all_gather(x_tar, sync_grads=True)
                x_composed_all = einops.rearrange(x_composed_all, "d b e -> (d b) e")
                x_tar_all = einops.rearrange(x_tar_all, "d b e -> (d b) e")

                loss_itc += criterion(x_composed_all, x_tar_all, model.temp)

            ## TDM                
            with torch.no_grad():          
                _, x_tdm = model(image_embs=tar_emb, text=text, text_attention_mask=text_attention_mask)
                x_tdm_all = fabric.all_gather(x_tdm, sync_grads=True)
                x_tdm_all = einops.rearrange(x_tdm_all, "d b e -> (d b) e")

                sim_i2t = x_tdm @ x_tar_all.t()
                sim_t2i = x_tar @ x_tdm_all.t()

                weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1) + 1e-4
                weights_i2t.fill_diagonal_(0)
                weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1) + 1e-4
                weights_t2i.fill_diagonal_(0)

            # select a negative image for each text
            tar_emb_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                tar_emb_neg.append(tar_emb[neg_idx])
            tar_emb_neg = torch.stack(tar_emb_neg, dim=0)

            # select a negative text for each image
            text_ids_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_ids_neg.append(text[neg_idx])
                text_atts_neg.append(text_attention_mask[neg_idx])

            text_ids_neg = torch.stack(text_ids_neg, dim=0)
            text_atts_neg = torch.stack(text_atts_neg, dim=0)

            image_embeds_all = torch.cat([tar_emb, tar_emb_neg, tar_emb], dim=0)
            text_ids_all = torch.cat([text, text, text_ids_neg], dim=0)
            text_atts_all = torch.cat([text_attention_mask, text_attention_mask, text_atts_neg], dim=0)

            # Exclude cls token for tdm
            text_atts_all[:, 0] = 0

            tdm_embs, _ = model(image_embs=image_embeds_all, text=text_ids_all, text_attention_mask=text_atts_all)
            tdm_logits = model.tdm_head(tdm_embs.bfloat16())

            tdm_logits = tdm_logits * text_atts_all.unsqueeze(-1)
            tdm_logits = tdm_logits.sum(dim=1) / text_atts_all.sum(dim=1, keepdim=True).float()
            tdm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)], dim=0).to(fabric.device)

            loss_tdm = F.cross_entropy(tdm_logits, tdm_labels)

            fabric.barrier()
            loss = loss_itc + loss_tdm
            fabric.print(f"epoch {epoch} iter {iter} ({(iter/total_iter)*100:.4f}%) lr {lr:.6f} loss_itc {loss_itc.item():.4f} loss_tdm {loss_tdm.item():.4f}")

            fabric.backward(loss)
            optimizer.step()
            
            iter += 1
        if epoch+1 == args.epochs:
            save_path = os.path.join(args.output_dir, f"BLIP_tdm_{args.model}_{args.dataset}_{epoch+1}")
            fabric.save(save_path, {"model": model})
            fabric.barrier()
            if fabric.global_rank == 0:
                # Create a consolidated checkpoint with the same name next to the deepspeed checkpoint
                convert_zero_checkpoint_to_fp32_state_dict(save_path, save_path+'.pth')
                if os.path.isdir(save_path):
                    shutil.rmtree(save_path)  # use shutil.rmtree to delete a directory recursively

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,help='Batch size per GPU')
    parser.add_argument("--model", type=str, default="blip-itm-large-coco")
    parser.add_argument('--epochs', default=6, type=int)
    parser.add_argument('--dataset', nargs='+', help='List of file names', type=str)
    parser.add_argument('--extra_dataset', nargs='+', help='List of file names', type=str, default=None)
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--init_lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=0.0)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--output_dir', default='CIR_out', help='path where to save, empty for no saving')
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--world_size', default=8, type=int, help='number of distributed processes')
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)