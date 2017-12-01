import numpy as np
import cv2
import uuid
import os
import shutil
from scipy import misc
import tensorflow as tf
from PIL import Image, ImageDraw
from io import BytesIO
from flask import send_file
import align.detect_face

minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor
gpu_memory_fraction = 1.0
image_size = 160
margin = 32


class FaceDetect:
    def __init__(self):
        self.pnet = None
        self.rnet = None
        self.onet = None
        self.create_mtcnn()

    def create_mtcnn(self):
        print('Creating networks and loading parameters')
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                self.pnet, self.rnet, self.onet = align.detect_face.create_mtcnn(sess, None)

    def detect_faces_core(self, img_arr, img_name=''):

        bounding_boxes, _ = align.detect_face.detect_face(img_arr, minsize, self.pnet, self.rnet, self.onet, threshold,
                                                          factor)
        if len(bounding_boxes) == 0:
            raise IndexError('Can not detect face from image {}'.format(img_name))

        return bounding_boxes

    def detect_faces_in_video(self, file_stream):
        tmp_video_dir = '/tmp/face/{}'.format(uuid.uuid4())
        if not os.path.isdir(tmp_video_dir):
            os.makedirs(tmp_video_dir)

        input_path = tmp_video_dir + '/input{}'.format(file_stream.filename)
        output_path = tmp_video_dir + '/output{}'.format(file_stream.filename)
        file_stream.save(input_path)
        self.detect_faces_in_video_process(input_path, output_path)

        with open(output_path, mode='rb') as f:
            byte_io = BytesIO(f.read())

        # byte_io.seek(0)

        # shutil.rmtree(tmp_video_dir)

        video_obj = send_file(byte_io, mimetype='audio/mp4')
        return video_obj

    def detect_faces_in_video_process(self, input_path, output_path):
        # Open the input movie file
        input_movie = cv2.VideoCapture(input_path)

        frame_width = input_movie.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = input_movie.get(cv2.CAP_PROP_FPS)

        # Create an output movie file (make sure resolution/frame rate matches input video!)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_movie = cv2.VideoWriter(output_path, fourcc, fps, (int(frame_width), int(frame_height)))

        # Initialize some variables
        frame_number = 0

        while True:
            # Grab a single frame of video
            ret, frame = input_movie.read()
            frame_number += 1

            # Quit when the input video file ends
            if not ret:
                break

            try:
                face_locations = self.detect_faces_core(frame)
            except Exception:
                pass
            else:
                # Label the results
                for left, top, right, bottom, _ in face_locations:
                    # Draw a box around the face
                    cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), 2)

            # Write the resulting image to the output video file
            print("Writing frame {}".format(frame_number))
            output_movie.write(frame)

        # All done!
        input_movie.release()
        output_movie.release()

    def detect_faces_in_image(self, file_stream):
        img_arr = misc.imread(file_stream, mode='RGB')

        bounding_boxes = self.detect_faces_core(img_arr, file_stream.filename)

        img = Image.fromarray(img_arr)
        draw = ImageDraw.Draw(img)

        img_size = np.asarray(img_arr.shape)[0:2]
        for det in bounding_boxes:
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0], 0)
            bb[1] = np.maximum(det[1], 0)
            bb[2] = np.minimum(det[2], img_size[1])
            bb[3] = np.minimum(det[3], img_size[0])

            draw.rectangle(((bb[0], bb[1]), (bb[2], bb[3])), outline="red")

        byte_io = BytesIO()

        img.save(byte_io, 'PNG')
        byte_io.seek(0)

        return send_file(byte_io, mimetype='image/png')

    def crop_faces_in_image(self, file_streams):
        nrof_samples = len(file_streams)
        img_list = [None] * nrof_samples
        for i in range(nrof_samples):
            file_stream = file_streams[i]
            img_arr = misc.imread(file_stream, mode='RGB')

            bounding_boxes = self.detect_faces_core(img_arr, file_stream.filename)

            img_size = np.asarray(img_arr.shape)[0:2]

            det = np.squeeze(bounding_boxes[0, 0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
            cropped = img_arr[bb[1]:bb[3], bb[0]:bb[2], :]
            aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            prewhitened = prewhiten(aligned)
            img_list[i] = prewhitened
        images = np.stack(img_list)
        return images


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y
