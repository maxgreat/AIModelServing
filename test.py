import torch

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
import PIL.Image as Image

import time
import argparse

parser = argparse.ArgumentParser(description='Testing environment and models')
parser.add_argument('models', metavar='N', type=str, nargs='+',
                    help='List of model to test from [image2vid, superresolution, objectdetection]')
parser.add_argument('--compile', action='store_true', 
                    help='Compile the model or not')
parser.add_argument('--cuda', action='store_true',
                    help='Compile the model or not')
parser.add_argument('--input', default="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png")
parser.add_argument('--output')

args = parser.parse_args()

if 'image2vid' in args.models:
    model_name = "stabilityai/stable-video-diffusion-img2vid"
    print(f"Testing Image to Video with model : {model_name}")
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        model_name, torch_dtype=torch.float16, variant="fp16"
    )
    if args.cuda:
        pipe = pipe.to("cuda")
    if args.compile:
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

    # Load the conditioning image
    if args.input.startswith('http'):
        print("Downloading image")
        image = load_image(args.input)
    else:
        image = Image.open(args.input)

    if image.format == 'GIF':
        print('Converting image')
        image.seek(0)
        image = image.copy().convert("RGB")
    print("Original mage size :", image.size)
    image = image.resize((128, int(128*image.size[1]/image.size[0]))) #for debug purpose - to be removed
    generator = torch.manual_seed(42)
    print("Start generating with image size :", image.size)
    t = time.time()
    with torch.no_grad():
        frames = pipe(image, num_frames=14, num_inference_steps=20, motion_bucket_id=127, decode_chunk_size=6, generator=generator).frames[0]
    print(f"Generated video in {time.time() - t} sec")
    print('Saving video...')
    if args.output is None:
        args.output = 'generated.gif'
    #export_to_video(frames, args.output, fps=7)
    frames[0].convert("RGBA")
    frames[0].putalpha(255)
    frames[0].save(args.output, save_all=True, append_images=frames[1:])
    print('Done')
    