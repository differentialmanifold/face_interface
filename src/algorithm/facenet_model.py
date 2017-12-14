import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from scipy import misc
import align.detect_face

minsize = 20  # minimum size of face
mtcnn_threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor
gpu_memory_fraction = 1.0
image_size = 160
margin = 32

model_path = os.path.join(os.path.dirname(__file__), '../../data/20170512-110547/20170512-110547.pb')


class FacenetFaceRecognize:
    def __init__(self):
        self.pnet = None
        self.rnet = None
        self.onet = None
        self.create_mtcnn()
        self.inception_resnet_v1 = inception_resnet()

    def create_mtcnn(self):
        print('Creating networks and loading parameters')
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                self.pnet, self.rnet, self.onet = align.detect_face.create_mtcnn(sess, None)

    def detect_face(self, img_arr):
        bounding_boxes, _ = align.detect_face.detect_face(img_arr, minsize, self.pnet, self.rnet, self.onet,
                                                          mtcnn_threshold, factor)

        bounding_boxes = [(left, top, right, bottom) for (left, top, right, bottom, _) in
                          bounding_boxes]
        return bounding_boxes

    def trans_image_emb(self, img, bounding_box):
        cropped = face_align(image_size, img, bounding_box)

        emb = self.inception_resnet_v1([cropped])[0]

        return emb


def inception_resnet():
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            # Load the model
            load_model(model_path)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            inception_resnet_fun = lambda images: sess.run(embeddings, feed_dict={images_placeholder: images,
                                                                                  phase_train_placeholder: False})
            return inception_resnet_fun


def load_model(model):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if os.path.isfile(model_exp):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt_file = None
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


def face_align(image_size, img_arr, bounding_box):
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
