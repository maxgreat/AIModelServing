import torch

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

print("Creating model")
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid", torch_dtype=torch.float16, variant="fp16"
)
pipe.to("cuda")
#pipe.enable_model_cpu_offload()
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

# Load the conditioning image
print("Downloading image")
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png")
image = image.resize((1024, 573))

w, h = image.size
if h % 64 != 0 or w % 64 != 0:
    width, height = map(lambda x: x - x % 64, (w, h))
    image = image.resize((width, height))
    print(
        f"WARNING: Your image is of size {h}x{w} which is not divisible by 64. We are resizing to {height}x{width}!"
    )

generator = torch.manual_seed(42)
print("Start generating")
with torch.no_grad():
    frames = pipe(image, num_frames=6, num_inference_steps=5, motion_bucket_id=127, decode_chunk_size=6, generator=generator).frames[0]
export_to_video(frames, "generated.mp4", fps=7)