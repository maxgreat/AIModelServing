import requests

url="http://localhost:8080/superresolution"
image_path = 'test_image.png'
with open(image_path, 'rb') as image:
    files = {'image': (image_path, image, 'multipart/form-data')}
    print("Reaching :", url)
    response = requests.post(url, files=files)
    print(response)

url = f"http://localhost:8000/task/{response['task_id']}/"
response_task = requests.post(url)
print(response_task)