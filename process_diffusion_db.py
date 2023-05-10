import json
from datasets import load_dataset
import torch
import open_clip
import numpy as np
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser("create a dataset with hps")
parser.add_argument("--source_dataset", default='large_first_1m', type=str)
parser.add_argument("--clip_checkpoint", type=str)
parser.add_argument("--meta_file", type=str)

args = parser.parse_args()

# Load the dataset
dataset = load_dataset('poloclub/diffusiondb', args.source_dataset, ignore_verifications=True)

# Load the model
model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', device='cuda')
params = torch.load(args.clip_checkpoint)['state_dict']
model.load_state_dict(params)
tokenizer = open_clip.get_tokenizer('ViT-L-14')

def is_blurry(image):
    image = np.array(image).astype(float)
    return max((image[:,:-1] - image[:,1:]).max(), (image[:,:-1] - image[:,1:]).max()) < 10.0

class dataset_wrapper:
    def __init__(self, dataset, preprocess, tokenizer) -> None:
        self.dataset = dataset
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = self.preprocess(item['image'])
        tokens = self.tokenizer(item['prompt'])[0]
        ret = dict(
            id=idx,
            image=image,
            tokens=tokens,
            blurry=is_blurry(item['image']),
            prompt=item['prompt'],
        )
        return ret

dataloader = DataLoader(dataset_wrapper(dataset['train'], preprocess, tokenizer), batch_size=20,
                          num_workers=4, shuffle=False)

def to_device(d, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in d.items()}

with open(args.meta_file, 'w') as f:
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = to_device(batch, "cuda")
            
            image_features = model.encode_image(batch['image'])
            text_features = model.encode_text(batch['tokens'])
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (image_features @ text_features.T).diag()
            
            for s, p, i, b in zip(similarity, batch['prompt'], batch['id'], batch['blurry']):
                f.write(json.dumps(dict(
                    id=i.item(),
                    prompt=p,
                    score=s.item() * 100,
                    blurry=b.item(),
                )) + '\n')