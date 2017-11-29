import os
import tensorflow as tf
import re
from tensorflow.python.platform import gfile
import numpy as np

model_path = os.path.join(os.path.dirname(__file__), '../data/20170512-110547')
thresholds = 1.2


class FaceVerify:
    def __init__(self):
        self.inception_resnet_v1 = inception_resnet()

    def compare_face_in_image(self, detect_obj, image_files):
        images = detect_obj.crop_faces_in_image(image_files)

        emb = self.inception_resnet_v1(images)

        nrof_images = len(image_files)

        if nrof_images != 2:
            raise Exception('num of images should be 2')

        dist = np.sqrt(np.sum(np.square(np.subtract(emb[0, :], emb[1, :])))).astype(np.float64)

        return {'issame': bool(dist < thresholds), 'distance': dist, 'thresholds': thresholds}


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
