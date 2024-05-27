from torch.utils.data import Dataset
from PIL import Image
import torch

IGNORE_INDEX = -1
PAD_INDEX = 0
BOS_INDEX = 1
EOS_INDEX = 2

class CIRDataTest(Dataset):
    def __init__(self, data_list, image_processor, tokenizer):
        self.data_list = data_list

        self.image_processor = image_processor
        self.tokenizer = tokenizer

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
        return self.image_processor(image, return_tensors="pt")['pixel_values'][0]

    def prepare_text(self, encoded_full_prompt, encoded_full_prompt_and_response, mask_inputs: bool = True):

        labels = encoded_full_prompt_and_response.clone()
        if mask_inputs:
            labels[:len(encoded_full_prompt)] = IGNORE_INDEX
        return {"input_ids": encoded_full_prompt_and_response, "labels": labels}


    def sample_one(self, image_info, idx):
        image_ref = self.prepare_image(image_info['reference'])
        image_tar = self.prepare_image(image_info['target'])

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
        language_dict = {"input_ids": template_tokens}

        return dict(image_ref=image_ref, image_tar=image_tar, input_ids=language_dict['input_ids'], image_idx=idx, image_positions=[image_placeholder_position_ref, image_placeholder_position_tar])
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_info = self.data_list[idx]
        return self.sample_one(image_info, idx)
    
class DataCollatorImageTest(object):
    def __call__(self, instances):
        image_ref, image_tar, input_ids, image_positions, image_idx = tuple([instance[key] for instance in instances] for key in ("image_ref", "image_tar", "input_ids", "image_positions", "image_idx"))
        
        image_ref = torch.stack(image_ref).bfloat16()
        image_tar = torch.stack(image_tar).bfloat16()
        
        # Pad input sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=PAD_INDEX)
    
        return dict(image_ref=image_ref, image_tar=image_tar, input_ids=input_ids, image_positions=image_positions, image_idx=image_idx)
    