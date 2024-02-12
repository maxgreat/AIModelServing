import requests

url="http://localhost:8080/superresolution"
image_path = 'test_image.png'
with open(image_path, 'rb') as image:
    files = {'image': (image_path, image, 'multipart/form-data')}
    print("Reaching :", url)
    response = requests.post(url, files=files)
    print(response)

j = response.json()
url = f"http://localhost:8000/task/{j['task_id']}/"
response_task = requests.post(url)
print(response_task)



url="http://localhost:8080/imagegeneration"
data = {
    'prompt': 'picture of a cat',
    'negative_prompt': 'ugly, dog, incorrect, limb',
    'width': '256',
    'height': '256',
    'num_inference': '10'
}
response = requests.post(url, data=data)
print(response)

j = response.json()
url = f"http://localhost:8000/task/{j['task_id']}/"
response_task = requests.post(url)
print(response_task)