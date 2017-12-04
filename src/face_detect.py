import numpy as np
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
            crop_img_arrs = self.crop_faces_in_image_item(img_arr, file_stream.filename)

            img_list[i] = crop_img_arrs[0]
        images = np.stack(img_list)
        return images

    def crop_faces_in_image_item(self, img_arr, filename=None, bounding_boxes=None):
        if bounding_boxes is None:
            bounding_boxes = self.detect_faces_core(img_arr, filename)

        img_size = np.asarray(img_arr.shape)[0:2]

        crop_img_arrs = []
        for det in bounding_boxes:
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
            cropped = img_arr[bb[1]:bb[3], bb[0]:bb[2], :]
            aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            prewhitened = prewhiten(aligned)
            crop_img_arrs.append(prewhitened)
        return crop_img_arrs


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y
