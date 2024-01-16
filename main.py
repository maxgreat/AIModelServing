from flask import Flask, request, send_file
from PIL import Image
from io import BytesIO
import torch
import torch
import uuid
import time

from diffusers import StableVideoDiffusionPipeline, StableDiffusionUpscalePipeline
from diffusers.utils import export_to_video

app = Flask(__name__)

videopipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid", torch_dtype=torch.float16, variant="fp16"
)
videopipe.to("cuda")
#videopipe.unet = torch.compile(videopipe.unet, mode="reduce-overhead", fullgraph=True)
generator = torch.manual_seed(42)

superresolutionpipe = StableDiffusionUpscalePipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", variant='fp16', torch_dtype=torch.float16).to("cuda")


@app.route('/video_diffusion', methods=['POST'])
def video_diffusion():
    # Check if an image is provided
    if 'image' not in request.files:
        return "No image provided", 400
        
    file = request.files['image']
    if file:
        # Convert the image file to an Image object
        image = Image.open(BytesIO(file.read()))
        if image.format == 'GIF':
            print('Converting image')
            image.seek(0)
            image = image.copy()
        image = image.convert("RGB")
        print("Original image size :", image.size)
        max_size = 256
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
        
        video_filename = f"generated_{uuid.uuid4().hex}.gif"
        frames[0].save(video_filename, save_all=True, append_images=frames[1:])

        # Send the video file back
        return send_file(video_filename, mimetype='image/gif')

    return "Invalid request", 400


@app.route('/superresolution', methods=['POST'])
def superresolution():
    if 'image' not in request.files:
        return "No image provided", 400
        
    file = request.files['image']



if __name__ == '__main__':
    app.run(host='localhost', port=8080, debug=True)