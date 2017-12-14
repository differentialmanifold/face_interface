import os
import numpy as np
import tensorflow as tf
from scipy import misc
import align.detect_face
from face_exception import FaceException

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
            raise FaceException(FaceException.NO_FACE_ERR_CODE, 'Can not detect face from image {}'.format(img_name))

        return bounding_boxes


face_detect_obj = FaceDetect()


def detect_faces_core(img_arr, img_name=''):
    bounding_boxes = face_detect_obj.detect_faces_core(img_arr, img_name=img_name)

    bounding_boxes = [(int(left), int(top), int(right), int(bottom)) for (left, top, right, bottom, _) in
                      bounding_boxes]

    return bounding_boxes


def find_largest_face(bounding_boxes):
    bounding_box = max(bounding_boxes, key=lambda rect: (rect[2] - rect[0]) * (rect[3] - rect[1]))
    return bounding_box


def crop_face_in_image_all(img_arr, filename=None, bounding_boxes=None):
    if bounding_boxes is None:
        bounding_boxes = detect_faces_core(img_arr, filename)

    crop_img_arrs = []
    for bb in bounding_boxes:
        cropped = face_align(image_size, img_arr, bb)
        crop_img_arrs.append(cropped)

    return crop_img_arrs


def crop_face_in_image_largest(img_arr, filename=None):
    bounding_boxs = detect_faces_core(img_arr, filename)
    max_bb = find_largest_face(bounding_boxs)

    cropped = face_align(image_size, img_arr, max_bb)
    return cropped


def face_align(image_size, img_arr, bounding_box=None):
    if bounding_box is None:
        bounding_boxes = detect_faces_core(img_arr)
        bounding_box = find_largest_face(bounding_boxes)

    img_shape = np.asarray(img_arr.shape)[0:2]
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(bounding_box[0] - margin / 2, 0)
    bb[1] = np.maximum(bounding_box[1] - margin / 2, 0)
    bb[2] = np.minimum(bounding_box[2] + margin / 2, img_shape[1])
    bb[3] = np.minimum(bounding_box[3] + margin / 2, img_shape[0])
    cropped = img_arr[bb[1]:bb[3], bb[0]:bb[2], :]
    aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
    prewhitened = prewhiten(aligned)

    return prewhitened


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y
