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

from celery import Celery

app = Flask(__name__)
celery = Celery(app.name, broker='amqp://localhost')
celery.conf.task_annotations = {'tasks.perform_prediction': {'rate_limit': '1/s'}}


### MODELS ###
text2img = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
if torch.cuda.is_available():
    text2img = text2img.to('cuda')

img2img = StableDiffusionImg2ImgPipeline(**text2img.components)
if torch.cuda.is_available():
    img2img = img2img.to('cuda')

inpaint = StableDiffusionInpaintPipeline(**text2img.components)
if torch.cuda.is_available():
    inpaint = inpaint.to('cuda')

videopipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid", torch_dtype=torch.float16, variant="fp16"
)
if torch.cuda.is_available():
    videopipe = videopipe.to('cuda')
videopipe.unet = torch.compile(videopipe.unet, mode="reduce-overhead", fullgraph=True)
generator = torch.manual_seed(42)

superresolutionpipe = StableDiffusionUpscalePipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", variant='fp16', torch_dtype=torch.float16)
if torch.cuda.is_available():
    superresolutionpipe = superresolutionpipe.to('cuda')



### TASK CHECK ####

@app.route('/task/<task_id>', methods=['GET'])
def get_task_status(task_id):
    task = celery.AsyncResult(task_id)

    if task.state == 'PENDING':
        # Job hasn't started yet
        response = {
            'state': task.state,
            'status': 'Pending...'
        }
    elif task.state == 'SUCCESS':
        # Task completed successfully, return the result
        return jsonify(task.result)
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'result': task.result,
            'status': str(task.info)  # Optional: additional info about the task
        }
    else:
        # Something went wrong in the background job
        response = {
            'state': task.state,
            'status': str(task.info),  # Error message or info
        }
    return jsonify(response)


### UPSCALING ###

@celery.task
def offline_supperesolution(data):
    print('Launch offline superresolution with Celery')
    image_data = data['image_data']
    prompt = data['prompt']
    image = Image.open(BytesIO(image_data))
    image_filename = f"generated_{uuid.uuid4().hex}.png"
    if image.format == 'GIF':
        print('Converting image')
        image.seek(0)
        image = image.copy()
    image = image.convert("RGB")
    upscaled_image = superresolutionpipe(prompt=prompt, 
                                         image=image).images[0]
    #upscaled_image.save(file_out)
    return upscaled_image

@app.route('/superresolution', methods=['POST'])
def superresolution():
    if 'image' not in request.files:
        return "No image provided", 400

    image = request.files['image']
    if image:
        prompt = request.form.get('prompt', '')  # Default to empty string if not provided
        # Read the image and prepare data for the task
        image_data = image.read()
        task_data = {'image_data': image_data, 'prompt': prompt}

        # Launch the task
        task = offline_supperesolution.delay(task_data)
        return jsonify({"task_id": task.id}), 202
        
    return "Invalid request : cannot read image", 400




### IMAGE TO VIDEO

@celery.task
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
    video_url = url_for('static', filename=filename_out, _external=True)
    return video_url


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

        task = offline_video_diffusion.delay(file, video_filepath)
        return jsonify({"task_id": task.id}), 202

    return "Invalid request : cannot read image", 400



### TEXT TO IMAGE ###

@celery.task
def offline_imagegeneration(prompt, image_filepath, negative_prompt=None, width:int = None, height:int = None, num_inference:int = 25):
    print(f'generating image with prompt : {prompt}, neg : {negative_prompt}, width : {width}, height : {height}, num_inference: {num_inference}')
    if width is not None:
        width = int(width)
    if height is not None:
        height = int(height)
    image = text2img(prompt=prompt, height=height, width=width, 
                     negative_prompt=negative_prompt, 
                     num_inference_steps=num_inference).images[0]
    image.save(image_filepath)
    image_url = url_for('static', filename=f'{image_filepath}', _external=True)
    return image_url

@app.route('/imagegeneration', methods=['POST'])
def imagegeneration():
    prompt = request.form.get('prompt')
    negative_prompt = request.form.get('neg_prompt')
    width = request.form.get('width')
    height = request.form.get('height')
    num_inference = request.form.get('inference_step')
    if num_inference is None:
        num_inference=25

    if prompt:
        public_folder = 'static/'
        image_filename = f"generated_{uuid.uuid4().hex}.png"
        image_filepath = os.path.join(public_folder, image_filename)
        
        task = offline_video_diffusion.delay(prompt, image_filepath, negative_prompt, width, height, num_inference)
        return jsonify({"task_id": task.id}), 202

    return "Invalid request", 400




if __name__ == '__main__':
    app.run(host='localhost', port=8080, debug=True)