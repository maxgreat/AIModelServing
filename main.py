from flask import Flask, request, send_file
from PIL import Image
from io import BytesIO
import torch
import torch
import uuid

from diffusers import StableVideoDiffusionPipeline, StableDiffusionUpscalePipeline
from diffusers.utils import export_to_video

app = Flask(__name__)

videopipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid", torch_dtype=torch.float16, variant="fp16"
)
videopipe.to("cuda")
videopipe.unet = torch.compile(videopipe.unet, mode="reduce-overhead", fullgraph=True)
generator = torch.manual_seed(42)

superresolutionpipe = StableDiffusionUpscalePipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", revision="fp16", torch_dtype=torch.float16).to("cuda")


@app.route('/video_diffusion', methods=['POST'])
def video_diffusion():
    # Check if an image is provided
    if 'image' not in request.files:
        return "No image provided", 400
        
    file = request.files['image']
    if file:
        # Convert the image file to an Image object
        image = Image.open(BytesIO(file.read())).convert('RGB')
        w, h = image.size
        if h % 64 != 0 or w % 64 != 0 or h + w > 2048:
            width, height = map(lambda x: min(x - x % 64, 1024), (w, h))
            image = image.resize((width, height))
            print(
                f"WARNING: Your image is of size {h}x{w} which is not divisible by 64 or to big. We are resizing to {height}x{width}!"
            )

        print("Start generating")
        with torch.no_grad():
            frames = videopipe(image, decode_chunk_size=6, generator=generator).frames
        
        video_filename = f"generated_{uuid.uuid4().hex}.mp4"
        export_to_video(frames, video_filename, fps=7)


        # Send the video file back
        return send_file(video_filename, mimetype='video/mp4')

    return "Invalid request", 400


@app.route('/superresolution', methods=['POST'])
def superresolution():
    if 'image' not in request.files:
        return "No image provided", 400
        
    file = request.files['image']



if __name__ == '__main__':
    app.run(debug=True)