import os
import tensorflow as tf
import re
import cv2
from tensorflow.python.platform import gfile
import numpy as np
import face_detect

model_path = os.path.join(os.path.dirname(__file__), '../data/20170512-110547/20170512-110547.pb')
known_images_dir = os.path.join(os.path.dirname(__file__), '../data/known_images')
thresholds = 1.2


class FaceVerify:
    def __init__(self):
        self.inception_resnet_v1 = inception_resnet()
        self.known_embs = self.load_known_images()

    def load_known_images(self):
        if not os.path.exists(known_images_dir):
            os.makedirs(known_images_dir)

        name_emb_tuples = []
        for basename in os.listdir(known_images_dir):
            name = basename.rsplit('.', maxsplit=1)[0]
            image_path = os.path.join(known_images_dir, basename)
            img_arr = face_detect.crop_face_in_image_largest(cv2.imread(image_path), basename)

            emb = self.inception_resnet_v1([img_arr])[0]
            name_emb_tuples.append((name, emb))
        return name_emb_tuples

    def find_name_in_database(self, face_encoding):
        names, embs = zip(*self.known_embs)
        distances = face_distance_arr(np.array(embs), face_encoding)

        min_index = np.argmin(distances)

        print('name: {}, distance: {}'.format(names[min_index], distances[min_index]))

        face_name = None
        if distances[min_index] < thresholds:
            face_name = names[min_index]
        return face_name


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


face_verify = FaceVerify()


def trans_faces_to_embs(face_arrs):
    embs = face_verify.inception_resnet_v1(face_arrs)
    return embs


def get_largest_face_embs(img_arrs):
    crop_frames = [face_detect.crop_face_in_image_largest(image) for image in img_arrs]
    embs = trans_faces_to_embs(crop_frames)
    return embs


def face_distance(encoding1, encoding2):
    return np.linalg.norm(encoding1 - encoding2).astype(np.float64)


def face_distance_arr(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty(0)
    return np.linalg.norm(face_encodings - face_to_compare, axis=1).astype(np.float64)


def compare_face_in_image(img_arrs):
    embs = get_largest_face_embs(img_arrs)

    nrof_images = len(img_arrs)

    if nrof_images != 2:
        raise Exception('num of images should be 2')

    dist = face_distance(embs[0], embs[1])

    issame = bool(dist < thresholds)

    similarity = max(0.0, 100 - (50 / thresholds ** 2) * float(dist) ** 2)

    return {'issame': issame, 'similarity': similarity}


def compare_two_faces_in_image(img_arr):
    crop_img_arrs = face_detect.crop_face_in_image_all(img_arr)

    if len(crop_img_arrs) != 2:
        raise Exception('Should contain two faces in one image')

    embs = trans_faces_to_embs(crop_img_arrs)

    dist = face_distance(embs[0], embs[1])
    issame = bool(dist < thresholds)

    similarity = max(0.0, 100 - (50 / thresholds ** 2) * float(dist) ** 2)

    return {'issame': issame, 'similarity': similarity}


def trans_bounding_position(bounding_boxes, img_arr):
    def _trim_css_to_bounds(css, image_shape):
        return max(css[0], 0), max(css[1], 0), min(css[2], image_shape[1]), min(css[3], image_shape[0])

    return [_trim_css_to_bounds(face, img_arr.shape) for face in bounding_boxes]


def draw_for_image(img_arr, bounding_boxes, with_name=False):
    # add name text
    face_names = []
    if with_name and len(face_verify.known_embs) > 0:
        face_arrs = face_detect.crop_face_in_image_all(img_arr, bounding_boxes=bounding_boxes)
        face_embs = trans_faces_to_embs(face_arrs)

        for face_encoding in face_embs:
            face_name = face_verify.find_name_in_database(face_encoding)

            face_names.append(face_name)

    face_locations = trans_bounding_position(bounding_boxes, img_arr)

    for i, (left, top, right, bottom) in enumerate(face_locations):
        # Draw a box around the face
        cv2.rectangle(img_arr, (left, top), (right, bottom), (0, 255, 0))

        if len(face_names) > 0:
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img_arr, face_names[i], (left + 6, bottom - 6), font, 0.5, (0, 255, 0))

    return img_arr
