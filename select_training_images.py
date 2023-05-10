import json
from datasets import load_dataset
import torch
import open_clip
import os
import numpy as np
import argparse
from tqdm import tqdm
from collections import Counter

parser = argparse.ArgumentParser("create a dataset with human preference")
parser.add_argument("--source_dataset", default='large_first_1m', type=str)
parser.add_argument("--positive_folder", type=str)
parser.add_argument("--negative_folder", type=str)
parser.add_argument("--meta_file", type=str)
parser.add_argument("--output_meta", type=str)
parser.add_argument("--quality", default=95, type=int)

args = parser.parse_args()

# Load the dataset with the `random_1k` subset
dataset = load_dataset('poloclub/diffusiondb', args.source_dataset, ignore_verifications=True)

with open(args.meta_file, 'r') as f:
    meta = [json.loads(line) for line in f.readlines()]

pt2idx = dict()
for d in meta:
    if d['blurry']:
        continue
    if d['prompt'] not in pt2idx:
        pt2idx[d['prompt']] = []
    pt2idx[d['prompt']].append(d['id'])

pt2idx = {k: v for k, v in  pt2idx.items() if len(v) >= 4}

if not os.path.exists(args.positive_folder):
    os.makedirs(args.positive_folder)
if not os.path.exists(args.negative_folder):
    os.makedirs(args.negative_folder)

def softmax(scores):
    mean = sum(scores) / len(scores)
    new_score = [s - mean for s in scores]
    exp = [np.exp(s) for s in new_score]
    denom = sum(exp)
    return [s / denom for s in exp]

positive_counter = 0
negative_counter = 0
with open(args.output_meta, 'w') as f:
    # log images with the maximum confidence over ...
    for batch_id, (prompt, ids) in tqdm(enumerate(pt2idx.items())):
        scores = [meta[i]['score'] for i in ids]
        probs = softmax(scores)
        neg_probs = softmax([-s for s in scores])
        if max(probs) > 2 / len(probs):
            good_id = probs.index(max(probs))
            file_name_good = f"{positive_counter}.jpg"
            dataset['train'][ids[good_id]]['image'].save(os.path.join(args.positive_folder, file_name_good), quality=args.quality)
            positive_counter += 1
            f.write(json.dumps(dict(
                id=positive_counter + negative_counter,
                file_name=os.path.join(os.path.basename(args.positive_folder), file_name_good),
                prompt=prompt,
                type='positive',
                confidence=max(probs),
                batch_size=len(probs),
            )) + '\n')
        if max(neg_probs) > 2 / len(probs):
            bad_id = neg_probs.index(max(neg_probs))
            file_name_bad = f"{negative_counter}.jpg"
            dataset['train'][ids[bad_id]]['image'].save(os.path.join(args.negative_folder, file_name_bad), quality=args.quality)
            negative_counter += 1
            f.write(json.dumps(dict(
                id=positive_counter + negative_counter,
                file_name=os.path.join(os.path.basename(args.negative_folder), file_name_bad),
                prompt=prompt,
                type='negative',
                confidence=max(neg_probs),
                batch_size=len(probs),
            )) + '\n')