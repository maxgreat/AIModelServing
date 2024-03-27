import os
from PIL import Image
import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import argparse
import tqdm
import numpy as np
import subprocess
import queue

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


def swap_face(frame, faces, reference_face):
    for face in faces:
        frame = swapper.get(frame, face, reference_face, paste_back=True)
    return frame


if __name__ == '__main__':
    img_reference = cv2.imread(args.reference_path)
    reference_face = app.get(img_reference)[-1]   

    if os.path.isdir(args.target_path):
        fun_fullpath = lambda x: os.path.join(args.target_path,x)
        fun_exists = lambda x : os.path.isfile(x)
        files_list = filter(fun_exists, map(fun_fullpath, os.listdir(args.target_path)))

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
            output_image_path = args.output_path + os.path.basename(f)
            if os.path.basename(f) in filesToExclude or os.path.basename(f).replace(' ', '') in filesToExclude:
                print(f"Skipping {f}")
                continue
            if os.path.exists(output_image_path) or os.path.exists(output_image_path.replace(' ', '')):
                print("Already exists")
                continue
            print("Handling :", f)  
            
            if(f.lower().endswith(('png', 'jpg', 'jpeg'))): #image
                print("as an image")
                img = cv2.imread(f)
                faces = app.get(img)
                res = img.copy()
                frame = swap_face(res, faces, reference_face)
                print("Frame shape :", frame.shape)
                cv2.imwrite(output_image_path, frame)
            elif(f.lower().endswith(('gif')) or f.lower().endswith(('webp'))): #gif
                print('as a gif or a webp')
                img = Image.open(f)
                duration = []
                list_image = []
                for i in tqdm.tqdm(range(img.n_frames)):
                    img.seek(i)
                    try:
                        duration.append(img.info['duration'])
                    except Exception:
                        duration.append(0)
                    new_file_name = f'temp/{os.path.basename(f).split(".")[0]}_{i}.png'
                    img.save(new_file_name)
                    img_png = cv2.imread(new_file_name)
                    faces = app.get(img_png)
                    frame = swap_face(img_png, faces, reference_face)
                    cv2.imwrite(new_file_name, frame)
                    list_image.append(new_file_name)
                image_output = Image.open(list_image[0])
                image_output.save(output_image_path, save_all=True, append_images=[Image.open(i) for i in list_image[1:]], duration=duration, loop=0)
                [os.remove(i) for i in list_image]
            elif(f.lower().endswith(('mp4', '.avi'))): #video
                print(f'{f} is a video')
                cap = cv2.VideoCapture(f, apiPreference=cv2.CAP_FFMPEG )
                framerate = int(cap.get(cv2.CAP_PROP_FPS))
                nbframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_width = int(cap.get(3))
                frame_height = int(cap.get(4))
                out = cv2.VideoWriter(output_image_path,cv2.VideoWriter_fourcc(*"mp4v"), framerate, (frame_width,frame_height))
                #q = queue.Queue()
                frames_batch = []
                for i in tqdm.tqdm(range(nbframe)):
                    ret, frame = cap.read()
                    
                    # if frame is read correctly ret is True, and add frame to batch
                    if ret:
                        frames_batch.append(frame)
                        
                        # When batch size is met or it's the last frame, process the batch
                        if len(frames_batch) == args.batch_size or i == nbframe - 1:
                            # Here, swap_face function needs to be able to handle batches of frames.
                            # faces detection for all frames in batch
                            faces_batch = [app.get(frame) for frame in frames_batch]  
                            # Processing batch of frames for face swapping
                            processed_frames = [swap_face(frame, faces, reference_face) for frame, faces in zip(frames_batch, faces_batch)]
                            
                            # Write each processed frame to output
                            for processed_frame in processed_frames:
                                out.write(processed_frame)
                            
                            # Clear the frames batch after processing
                            frames_batch = []
                    else:
                        print(f'End of video after {i} frames')
                        break
                cap.release()
                out.release()
                #np.stack(arrays, axis=0).shape
                #subprocess.run(f"ffmpeg -i {output_image_path} -i {f} -c copy -map 0:v -map 0:a? -map 1:a {output_image_path}".split(' '))
                
