from flask import Flask, request, jsonify, url_for
from PIL import Image
from io import BytesIO
import torch
import torch
import uuid
import time
import os
from threading import Thread

from diffusers import StableVideoDiffusionPipeline, StableDiffusionUpscalePipeline, StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline




app = Flask(__name__)


### MODELS ###
text2img = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16).to('cuda')
#img2img = StableDiffusionImg2ImgPipeline(**text2img.components).to('cuda')
#inpaint = StableDiffusionInpaintPipeline(**text2img.components).to('cuda')

videopipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid", torch_dtype=torch.float16, variant="fp16"
).to("cuda")
videopipe.unet = torch.compile(videopipe.unet, mode="reduce-overhead", fullgraph=True)
generator = torch.manual_seed(42)

superresolutionpipe = StableDiffusionUpscalePipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", variant='fp16', torch_dtype=torch.float16).to("cuda")



### IMAGE TO VIDEO

def offline_video_diffusion(file, filename_out):
    image = Image.open(BytesIO(file.read()))
    if image.format == 'GIF':
        print('Converting image')
        image.seek(0)
        image = image.copy()
    image = image.convert("RGB")
    #print("Original image size :", image.size)
    max_size = 64
    image = image.resize((max_size, int(max_size*image.size[1]/image.size[0]))) #for debug purpose - to be removed
    generator = torch.manual_seed(42)
    print("Start generating with image size :", image.size)
    t = time.time()
    with torch.no_grad():
        frames = videopipe(image, num_frames=14, num_inference_steps=10, motion_bucket_id=127, decode_chunk_size=6, generator=generator).frames[0]
    print(f"Generated video in {time.time() - t} sec")
    
    frames[0].convert("RGBA")
    frames[0].putalpha(255)
    
    print("Start generating")
    
    #video_filename = f"generated_{uuid.uuid4().hex}.gif"
    frames[0].save(filename_out, save_all=True, append_images=frames[1:])


@app.route('/video_diffusion', methods=['POST'])
def video_diffusion():
    # Check if an image is provided
    if 'image' not in request.files:
        return "No image provided", 400
        
    file = request.files['image']   
    if file:
        public_folder = 'static/'  # Example folder name
        video_filename = f"generated_{uuid.uuid4().hex}.gif"
        video_filepath = os.path.join(public_folder, video_filename)
        
        # Generate a public URL for the video
        video_url = url_for('static', filename=video_filepath, _external=True)

        thread = Thread(target=offline_video_diffusion, args=(file, video_filepath))
        thread.start()
        
        # Return the URL
        return jsonify({'video_url': video_url})

    return "Invalid request", 400



### UPSCALING ###

def offline_supperesolution(file, file_out, prompt=None):
    image = Image.open(BytesIO(file.read()))
    if image.format == 'GIF':
        print('Converting image')
        image.seek(0)
        image = image.copy()
    image = image.convert("RGB")
    upscaled_image = superresolutionpipe(prompt=prompt, image=image).images[0]
    upscaled_image.save(file_out)

@app.route('/superresolution', methods=['POST'])
def superresolution():
    if 'image' not in request.files:
        return "No image provided", 400
        
    file = request.files['image']   
    prompt = request.form.get('prompt')

    if file:
        public_folder = 'static/'
        image_filename = f"generated_{uuid.uuid4().hex}.png"
        image_filepath = os.path.join(public_folder, image_filename)
        
        video_url = url_for('static', filename=f'{image_filepath}', _external=True)
        thread = Thread(target=offline_supperesolution, args=(file, image_filepath, prompt))
        thread.start()
        
        # Return the URL
        return jsonify({'image_url': video_url})

    return "Invalid request", 400


### TEXT TO IMAGE ###

def offline_imagegeneration(prompt, image_filepath, negative_prompt=None, width=None, height=None, num_inference=None):
    if num_inference is None:
        num_inference = 20
    image = text2img(prompt=prompt, height=height, width=width, 
                     negative_prompt=negative_prompt, 
                     num_inference_steps=num_inference).images[0]
    image.save(image_filepath)

@app.route('/imagegeneration', methods=['POST'])
def imagegeneration():
    prompt = request.form.get('prompt')
    negative_prompt = request.form.get('neg_prompt')
    width = request.form.get('width')
    height = request.form.get('height')
    num_inference = request.form.get('inference_step')

    if prompt:
        public_folder = 'static/'
        image_filename = f"generated_{uuid.uuid4().hex}.png"
        image_filepath = os.path.join(public_folder, image_filename)
        
        image_url = url_for('static', filename=f'{image_filepath}', _external=True)
        thread = Thread(target=offline_imagegeneration, args=(prompt, image_filepath, negative_prompt, width, height, num_inference))
        thread.start()
        
        # Return the URL
        return jsonify({'image_url': image_url})

    return "Invalid request", 400




if __name__ == '__main__':
    app.run(host='localhost', port=8080, debug=True)