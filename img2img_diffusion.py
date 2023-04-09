import requests
import torch
from PIL import Image
from io import BytesIO
import os
import matplotlib.pyplot as plt
from diffusers import StableDiffusionImg2ImgPipeline

# MAC OS Exection
os.system('set COMMANDLINE_ARGS=--skip-torch-cuda-test --precision full --no-half')

device = "mps"
model_id_or_path = "runwayml/stable-diffusion-v1-5"
model_id_or_path = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path)
pipe = pipe.to(device)
pipe.enable_attention_slicing()

# GTA Dataset Synthetic Image
url = "https://download.visinf.tu-darmstadt.de/data/from_games/data/17086_image.png"

response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image = init_image.resize((int(init_image.width/2), int(init_image.height/2)))

prompt3 = "camera, photorealistic, photograph, accurate"

strength_list = [0.4]
guidance_scale = [5]
inf_steps = [200]
prompt_list = [prompt3]

for s in strength_list:
    for g in guidance_scale:
        for i in inf_steps:
            for pr, val in enumerate(prompt_list):

                images = pipe(prompt=val, image=init_image, strength=s, num_inference_steps=i, guidance_scale=g, num_images_per_prompt=1).images

                fig, axarr = plt.subplots(1,2)
                fig.tight_layout()

                axarr[0].imshow(init_image)
                axarr[0].axis('off')
                axarr[0].set_title('Original')

                axarr[1].imshow(images[0])
                axarr[1].axis('off')
                axarr[1].set_title('Transformed')

                plt.savefig('output_strength_%f_guidance_%d_steps_%d_prompt_%d.png' % (s,g,i, pr), dpi=300)

                plt.close('all')