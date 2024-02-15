import os
from PIL import Image
import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import argparse
import tqdm
import numpy as np

args = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=100))
args.add_argument('-r', '--reference', help='select an source image', dest='reference_path')
args.add_argument('-e', '--exclude', help='directory with images to exclude', dest='exclude_path')
args.add_argument('-t', '--target', help='select an target image, video or directory', dest='target_path')
args.add_argument('-o', '--output', help='select output file or directory', dest='output_path')
args.add_argument('--similar-face-distance', help='face distance used for recognition', dest='similar_face_distance', type=float, default=0.85)
args.add_argument('-b','--batch_size', help='batch size', dest='batch_size', default=1, type=int)
args = args.parse_args()

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0)
swapper = insightface.model_zoo.get_model('models/inswapper_128.onnx', providers=['CUDAExecutionProvider'])


def swap_face(frame, face, reference_face):
    for face in faces:
        frame = swapper.get(frame, face, reference_face, paste_back=True)
    return frame


if __name__ == '__main__':
    img_reference = cv2.imread(args.reference_path)
    reference_face = app.get(img_reference)[-1]   

    if os.path.isdir(args.target_path):
        fun = lambda x : os.path.isfile(os.path.join(args.target_path,x))
        files_list = filter(fun, os.listdir(args.target_path))

        # Create a list of files in directory along with the size
        size_of_file = [
            (f,os.stat(os.path.join(args.target_path, f)).st_size)
            for f in files_list
        ]
        filesToExclude = os.listdir(args.exclude_path)
        fun = lambda x : x[1]
        if not os.path.exists(args.output_path):
            os.mkdir(args.output_path)
        for f,_ in sorted(size_of_file,key = fun): 
            if f in filesToExclude:
                continue
            
            output_image_path = args.output_path + os.path.basename(f)
            if os.path.exists(output_image_path):
                continue
            print("Handling :", f)  
            
            if(f.lower().endswith(('png', 'jpg', 'jpeg'))): #image
                print("as an image")
                img = cv2.imread(f)
                faces = app.get(img)
                res = img.copy()
                frame = swap_face(res, faces, reference_face)[0]
                cv2.imwrite(output_image_path, frame)
            elif(f.lower().endswith(('gif'))): #gif
                print('as a gif')
                img = cv2.imread(f)
                duration = []
                list_image = []
                for i in tqdm.tqdm(range(img.n_frames)):
                    img.seek(i)
                    duration.append(img.info['duration'])
                    new_file_name = f'temp/{os.path.basename(f).split(".")[0]}_{i}.png'
                    frame = swap_face(img, faces, reference_face)[0]
                    cv2.imwrite(new_file_name, frame)
                    list_image.append(new_file_name)
                image_output = Image.open(list_image[0])
                image_output.save(output_image_path, save_all=True, append_images=[Image.open(i) for i in list_image[1:]], duration=duration, loop=0)
                [os.remove(i) for i in list_image]
            elif(f.lower().endswith(('mp4'))): #video
                print('as a video')


                np.stack(arrays, axis=0).shape
                #ffmpeg -i main.m4v -i commentary.m4v -c copy -map 0:v -map 0:a -map 1:a final.m4v
