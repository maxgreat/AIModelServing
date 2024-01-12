import requests

url="http://localhost:8000/video_diffusion"
image_path = "test_image.png"
with open(image_path, 'rb') as image:
    files = {'image': (image_path, image, 'multipart/form-data')}
    print("Reaching :", url)
    response = requests.post(url, files=files)
    print(response)