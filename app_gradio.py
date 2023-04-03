import torch
from diffusers.models.cross_attention import LoRACrossAttnProcessor
from diffusers import StableDiffusionPipeline
import hashlib

import gradio as gr

# load a pretrained sd model, and then load the LoRA weights
def create_model(weight_path):
    model = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
    ).to("cuda")
    if not weight_path:
        return model
    model_weight = torch.load(weight_path, map_location='cpu')
    unet = model.unet
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
        ).to("cuda")
    unet.set_attn_processor(lora_attn_procs)
    unet.load_state_dict(model_weight, strict=False)
    return model

original = create_model("")
adapted = create_model("adapted_model.bin")

def inference(prompt):
    # create a hash of the prompt
    hash_object = hashlib.sha256(prompt.encode())
    hex_dig = hash_object.hexdigest()
    integer_value = int(hex_dig, 16)
    # create a pytorch generator based on the prompt
    generator = torch.Generator(device='cuda').manual_seed(integer_value % 2147483647)
    baseline_image = original(prompt, num_inference_steps=50, generator=generator, negative_prompt="Weird image. ").images[0]
    generator = torch.Generator(device='cuda').manual_seed(integer_value % 2147483647)
    adapted_image = adapted(prompt, num_inference_steps=50, generator=generator, negative_prompt="Weird image. ").images[0]
    return baseline_image, adapted_image

example_list = [
    "a painting of a waterfall in the middle of a forest, concept art, inspired by andreas rocha, fantasy art, in a beautiful crystal caverine, portal into anotheer dimension, underground lake, sparkling cove, vortex river, plants inside cave, moonlight shafts, intricate sparkling atmosphere, enchanting",
    "photoetch of prize - winning marble albino dog sculpture, intricate details, a turing pattern, aesthetic complexity, midjourney, black background, photorealistic, shpongle, glo - fi, psychedelic, first person view, realistic lighting, intricate, elite, contre - jour lighting",
    "photoetch of marble cacti sculpture with intricate details, black background, midjourney, mathematical formulas, sacred geometry, photorealistic, shpongle, glo - fi, psychedelic, first person view, realistic lighting, retro, intricate, elite, contre - jour lighting",
    "a female angel with blonde pigtails, pale skin and blue eyes, flying, pixar's style, detailed texture, hd, wings, cute, blushed, adorableness, smile, teeth braces",
    "soviet constructivism style, extremly detailed epic painting of a glowing greek sun god apollo zeus in triumphant pose surrounded by thunder, glorious, dark background, masterpiece, trending on artstation, stark red and black and beige and gold, constructivism, by mike mignola and joseph leyendecker and edward hopper",
    "an aquarium with a galaxy and fish inside floating in space with planet earth in the background, concept art, highly detailed photorealistic, dynamic hdr.",
    "a masterpiece landscape, david coulter, mike barr, greg rutkowski, anton fadeev, caspar david friedrich, ferdinand knab, hdr, trending on artstation, cel - shaded, oil painting, professional photography, colorful, complex, epic, realistic colors, hyperdetailed, intricate",
    "girl with fox ears, tired eyes, long wavy orange hair, light brown trenchcoat, forest background, focus on face, pretty, moody lighting, painterly, illustration by shigenori soejima ",
    "masterpiece portrait of an aesthetic mage woman, ice spell, 3 0 years old woman, ( katheryn winnick like ), black dynamic hair, wearing silver diadem with blue gems inlays, silver necklace, painting by joachim bergauer and magali villeneuve, atmospheric effects, chaotic blue sparks dynamics in the background, intricate, artstation, fantasy ",
    "male character study of male tori spelling, clear faces, screenwriter, introvert, outsider, geek, disturbed, emotional, character sheet, fine details, concept design, contrast, kim jung gi, pixar and da vinci, trending on artstation, 8 k, full body and head, turnaround, front view, back view, ultra wide angle ",
    "by yoji shinkawa, concept art of a beautiful woman with purple long hair wearing a large witch hat, ( highly detailed ), flying on broomstick, dynamic lighting, cinematic lighting, neon rim lighting, dreamy night sky background",
    "allen williams, asymmetrical fantantasy cute caracter of mystical ser, cartoon, hight resolution, subrealism, accene, miracle, homiest, rustic estyle, ancestral, 8 k, realism, vértigo estyle pint",
    "emma watson in lara croft costume, war scene, hyper realistic, dramatic cold light, very detailed face, 8 k resolution, photo realistic",
    "a picture of a forest elf clothed in flowers and leaves standing on a stone in an enchanted forest, high fantasy, elegant, epic, detailed, intricate, digital painting, concept art, realistic detailed face, smooth, focus, rim light,",
    "fairy princess, highly detailed, d & d, fantasy, highly detailed, digital painting, trending on artstation, concept art, sharp focus, illustration, art by artgerm and greg rutkowski and magali villeneuve",
    "cardboard knight in game of thrones, artstation trent, portrait, classic paint, heartstone style, wlop style",
    "pixel art digital lion art. wallpaper 3 d pixel art 8 k suoer detailed 3 2 bit. amazing pixel art details. flowers. pixel art. many flowers in the foreground",
    "portrait art of hatsune miku 8 k ultra realistic, lens flare, atmosphere, glow, detailed, intricate, full of colour, cinematic lighting, trending on artstation, 4 k, matte, hyperrealistic, focused, extreme details, unreal engine 5, cinematic, masterpiece",
    "ellyne, beautiful, queen of the unicorns, brown hair, crown, cinematic lighting, 8 k",
    "cute anime supergirl, short blonde hair, concept art, detailed, dark light, digital painting, elegant,",
    "lion king extreme ultra highly detailed full extreme detailed neon tech, hyperdetailed, distopic, by john blanche and greg rutkowski, trending on artstation, depth shading, volumetric light,, digital painting illustration, lighting tean and orange colors, super detailed colors, cinematic lighting colors",
    "a wolf that is merged with ornate silver jewelry and armor, made of celtic knots and other ornatmental patterns, in a snowy forest with a pink orange yellow background, snow flurries, standing on a rock, soft focus, dreamy, realistic 3 d oil painting, unreal engine",
    "fantasy bandit camp, realistic, highly detailed, intricate detailed, trending on artstation by ross tran",
    "(( beautiful girl )), elegant, ultra realistic digital art, pencil drawing, grimdark vintage woodcut sepia, ultra - detailed, hyper detailed, crazy details, intricate details, unreal engine, 8 k, full resolution, super detailed, sharp focus, architectural, volume, by paolo eleuteri serpieri",
    "keanu reeves wearing superman suit flying in the air like a god",
]

with gr.Blocks() as demo:
    with gr.Row():
        prompt = gr.Textbox(lines=1, label="Prompt")
        button = gr.Button("Generate")
    with gr.Row():
        with gr.Column():
            gr.Markdown('''
                ## Original Stable Diffusion
            ''')
            baseline_output = gr.Image(label="Original Stable Diffusion", type="pil")
        with gr.Column():
            gr.Markdown('''
                ## Adaptated model
            ''')
            adapted_output = gr.Image(label="Adapted", type="pil")
    gr.Markdown('''
                ## Example inputs
            ''')
    examples = gr.Examples(examples=example_list, label="Examples", inputs=[prompt], cache_examples=False, fn=inference, outputs=[baseline_output, adapted_output])
    button.click(inference, inputs=prompt, outputs=[baseline_output, adapted_output])

demo.queue(concurrency_count=1)
demo.launch()
