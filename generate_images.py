import os
import torch
from diffusers.models.cross_attention import LoRACrossAttnProcessor
from diffusers import StableDiffusionPipeline
import json

import argparse

parser = argparse.ArgumentParser("generate images from a unet lora checkpoint")
parser.add_argument("--unet_weight", default="", type=str)
parser.add_argument("--prompts", default="validation_prompts_500.json", type=str)
parser.add_argument("--folder", default="dump", type=str)
parser.add_argument("--negative_prompt", default="Weird image. ", type=str)
parser.add_argument("--world_size", default=-1, type=int)
parser.add_argument("--rank", default=-1, type=int)

args = parser.parse_args()

use_command_line_rank = args.world_size > 0 and args.rank >= 0

unet_weight = args.unet_weight
negative_prompt = args.negative_prompt

local_rank = 0
for v in ('LOCAL_RANK', 'MPI_LOCALRANKID', 'SLURM_LOCALID', 'OMPI_COMM_WORLD_LOCAL_RANK'):
    if v in os.environ:
        local_rank = int(os.environ[v])
        if use_command_line_rank:
            # make sure that only use one GPU when directly specify world size and rank from command line
            assert local_rank == 0
        break

if use_command_line_rank:
    rank = args.rank
else:
    rank = 0
    for v in ('RANK', 'PMI_RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK'):
        if v in os.environ:
            rank = int(os.environ[v])
            break

if use_command_line_rank:
    world_size = args.world_size
else:
    world_size = 1
    for v in ('WORLD_SIZE', 'PMI_SIZE', 'SLURM_NTASKS', 'OMPI_COMM_WORLD_SIZE'):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

with open(args.prompts) as f:
    pairs = json.load(f)

pipeline = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
).to(local_rank)

if unet_weight:
    model_weight = torch.load(unet_weight, map_location='cpu')
    unet = pipeline.unet
    lora_attn_procs = {}
    lora_rank = list(set([v.size(0) for k, v in model_weight.items() if k.endswith("down.weight")]))
    assert len(lora_rank) == 1
    lora_rank = lora_rank[0]
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRACrossAttnProcessor(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank
        ).to(local_rank)
    unet.set_attn_processor(lora_attn_procs)
    unet.load_state_dict(model_weight, strict=False)

if not os.path.exists(args.folder):
    os.makedirs(args.folder)

generator = torch.Generator(device='cuda').manual_seed(rank + 1)

for i, pair in enumerate(pairs):
    if i % world_size != rank:
        continue
    if os.path.exists(f"{args.folder}/{i}.jpg"):
        continue
    with torch.no_grad():
        raw_images = pipeline([pair], num_inference_steps=50, generator=generator, negative_prompt=[negative_prompt]).images
    for j, image in enumerate(raw_images):
        image.save(f"{args.folder}/{i}.jpg", quality=90)