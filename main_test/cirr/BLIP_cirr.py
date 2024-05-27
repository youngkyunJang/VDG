import json
import argparse
import os

import sys
from pathlib import Path
wd = Path(__file__).parent.parent.parent.resolve()
sys.path.append(str(wd))

import lightning as L
import torch
import transformers
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
from main_train.BLIP_tdm import BLIP_CIR

# Hyperparameters
micro_batch_size = 64
devices = 1
block_size = 2048
num_workers = 10

class Loader(Dataset):
    def __init__(self, image_list, image_path, processor):
        super().__init__()
        self.image_list = image_list
        self.processor = processor
        self.image_path = image_path

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.image_list[idx]
        image_ref = data["reference"]
        image_ref = Image.open(f"{self.image_path}{image_ref}.png").convert('RGB')

        text = data["caption"]
        image_set_names = [f"{self.image_path}{n}.png" for n in data["img_set"]["members"]]

        image_set = []
        for name in image_set_names:
            image_set.append(self.processor(images=Image.open(name).convert('RGB'), return_tensors="pt")['pixel_values'][0])

        text = self.processor(text=text, padding="max_length", max_length=32, truncation=True, return_tensors="pt")
        input_ids = text["input_ids"][0]
        text_attention_mask = text["attention_mask"][0]
        image_ref = self.processor(images=image_ref, return_tensors="pt")['pixel_values'][0]

        return image_ref, input_ids, text_attention_mask, image_set

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))

    image_ref_list, input_ids_list, text_attention_mask_list, image_set_list = zip(*batch)
    image_ref_batch = torch.utils.data.dataloader.default_collate(image_ref_list)
    input_ids_batch = torch.utils.data.dataloader.default_collate(input_ids_list)
    text_attention_mask_batch = torch.utils.data.dataloader.default_collate(text_attention_mask_list)
    image_set_batch = torch.stack([torch.stack(img_set, dim=0) for img_set in image_set_list], dim=0)

    return image_ref_batch, input_ids_batch, text_attention_mask_batch, image_set_batch

def main(args):
    fabric = L.Fabric(
        accelerator="cuda",
        devices=devices,
        precision="32-true"
    )
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    BLIPpretrained = f'Salesforce/{args.model}'

    with fabric.device:
        BLIP = transformers.BlipForImageTextRetrieval.from_pretrained(BLIPpretrained)
        processor = transformers.BlipProcessor.from_pretrained(BLIPpretrained)
        vision_model = BLIP.vision_model
        text_model = BLIP.text_encoder
        vision_proj = BLIP.vision_proj
        text_proj = BLIP.text_proj
        itm_head = BLIP.itm_head

        del BLIP
        torch.cuda.empty_cache()

        model = BLIP_CIR(
                        vision_model=vision_model,
                        text_model=text_model,
                        vision_proj=vision_proj,
                        text_proj=text_proj,
                        tdm_head=itm_head)

    test_list = args.test
    with open(test_list) as f:
        test_list = json.loads(f.read())

    test_dataset = Loader(test_list, args.image_path, processor)
    test_loader = DataLoader(test_dataset, num_workers=num_workers, batch_size=micro_batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)
    test_loader = fabric.setup_dataloaders(test_loader)

    model.load_state_dict(torch.load(f'{args.ckpt}'), strict=True)
    model.eval()

    model.to(fabric.device)
    test(fabric, model, test_loader, test_list, f"{os.path.splitext(os.path.basename(args.ckpt))[0]}")


@torch.no_grad()
def test(fabric: L.Fabric, model: torch.nn.Module, test_loader, test_list, model_name) -> torch.Tensor:
    fabric.print("Testing ...")

    img = torch.tensor([], dtype=torch.float32).to(fabric.device)
    cmb = torch.tensor([], dtype=torch.float32).to(fabric.device)
    img_set = torch.tensor([], dtype=torch.float32).to(fabric.device)

    unique_images = {}

    for samples in test_loader:
        image_ref, text, text_attention_mask, image_set = samples

        ref_emb, x_i = model(image=image_ref)
        _, x_composed = model(image_embs=ref_emb, text=text, text_attention_mask=text_attention_mask)

        set_tmp = []

        for i in range(image_set.size(1)):
            _, x_i_set = model(image=image_set[:,i])
            set_tmp.append(x_i_set)

        set_tmp = torch.stack(set_tmp, dim=1)

        img = torch.cat((img,x_i), dim=0)
        cmb = torch.cat((cmb,x_composed), dim=0)
        img_set = torch.cat((img_set, set_tmp), dim=0)

    image_names = [item["reference"] for item in test_list]
    image_ids = [item["reference_id"] for item in test_list]
    pairids = [item["pairid"] for item in test_list]

    unique_image_ids = []
    indices = []  # This will store the indices of the unique IDs in the original list

    for i, id in enumerate(image_ids):
        if id not in unique_image_ids:
            unique_image_ids.append(id)
            indices.append(i)  # Save the index of the first occurrence of this ID

    # Step 2: Filter out the non-unique elements using the indices we found
    unique_img = img[indices]  # This assumes 'img' is a tensor; it will use advanced indexing
    unique_image_names = [image_names[i] for i in indices]

    image_set_names = [item["img_set"]["members"] for item in test_list]
    image_set_ids = [item["img_set"]["member_ids"] for item in test_list]

    result = retrieve_top_50(cmb, unique_img, unique_image_names, pairids, unique_image_ids, image_ids)
    result["version"] = "rc2"
    result["metric"] = "recall"

    dict2save = json.dumps(result)

    with open('out/cirr_%s_top.json'%model_name, 'w') as file:
        file.write(dict2save)

    result = retrieve_subset_3(cmb, img_set, image_set_names, pairids, image_ids, image_set_ids)
    result["version"] = "rc2"
    result["metric"] = "recall_subset"

    dict2save = json.dumps(result)

    with open('out/cirr_%s_subset.json'%model_name, 'w') as file:
        file.write(dict2save)

    fabric.print("Done!")

def retrieve_top_50(query_embeddings, gallery_embeddings, image_names, pairids, gallery_image_ids, query_image_ids):
    retrieval_results = {}
    for query_idx, query in enumerate(query_embeddings):
        similarity = torch.mv(gallery_embeddings, query)  # compute similarity for this specific gallery set
        for i in range(len(gallery_image_ids)):
            if query_image_ids[query_idx] == gallery_image_ids[i]:
                similarity[i] = -10
        _, top_img_indices = torch.topk(similarity, 50)  # get the top 50 or fewer
        retrieval_results[pairids[query_idx]] = [image_names[i] for i in top_img_indices]  # adjusted to handle per-gallery image names
    return retrieval_results

def retrieve_subset_3(query_embeddings, gallery_embeddings, image_names, pairids, image_ids, image_set_ids):
    retrieval_results = {}
    # For each query, compute similarity only with the corresponding gallery set and retrieve top 3
    for query_idx, query in enumerate(query_embeddings):
        similarity = torch.mv(gallery_embeddings[query_idx], query)  # compute similarity for this specific gallery set
        for i in range(len(image_set_ids[query_idx])):
            if image_ids[query_idx] == image_set_ids[query_idx][i]:
                similarity[i] = -10
        _, top_img_indices = torch.topk(similarity, 3)  # get the top 3 or fewer
        retrieval_results[pairids[query_idx]] = [image_names[query_idx][i] for i in top_img_indices]  # adjusted to handle per-gallery image names

    return retrieval_results

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default='datasets/cirr_test.json')
    parser.add_argument("--model", type=str, default='blip-itm-large-coco')
    parser.add_argument("--llama", type=str, default='lit-llama2')
    parser.add_argument("--ckpt", type=str, default='BLIP_blip-itm-large-coco_cirr')
    parser.add_argument("--image_path", type=str, default='', help'put the path to the image folder here')
    args = parser.parse_args()

    main(args)
