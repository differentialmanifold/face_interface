import os
import cv2
import numpy as np
import face_detect

from face_exception import FaceException

known_images_dir = os.path.join(os.path.dirname(__file__), '../data/known_images')
thresholds = face_detect.thresholds


class FaceVerify:
    def __init__(self):
        self.known_embs = self.load_known_images()

    def load_known_images(self):
        if not os.path.exists(known_images_dir):
            os.makedirs(known_images_dir)

        name_emb_tuples = []
        for basename in os.listdir(known_images_dir):
            name = basename.rsplit('.', maxsplit=1)[0]
            image_path = os.path.join(known_images_dir, basename)
            emb = face_detect.get_face_emb_largest(cv2.imread(image_path), basename)

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


face_verify = FaceVerify()


def get_largest_face_embs(img_arrs):
    embs = [face_detect.get_face_emb_largest(image) for image in img_arrs]
    return embs


def face_distance(encoding1, encoding2):
    return np.linalg.norm(encoding1 - encoding2).astype(np.float64)


def face_distance_arr(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty(0)
    return np.linalg.norm(face_encodings - face_to_compare, axis=1).astype(np.float64)


def compare_face_in_image(img_arrs, show_distance=False):
    embs = get_largest_face_embs(img_arrs)

    nrof_images = len(img_arrs)

    if nrof_images != 2:
        raise FaceException(FaceException.PARAMETER_NUM_ERR_CODE, 'num of images should be 2')

    dist = face_distance(embs[0], embs[1])

    issame = bool(dist < thresholds)

    similarity = max(0.0, 100 - (50 / thresholds ** 2) * float(dist) ** 2)

    result = {'issame': issame, 'similarity': similarity}

    if show_distance:
        result['dist'] = dist

    return result


def compare_two_faces_in_image(img_arr):
    embs = face_detect.get_face_emb_all(img_arr)

    if len(embs) != 2:
        raise FaceException(FaceException.FACE_NUM_ERR_CODE, 'Should contain two faces in one image')

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
        face_embs = face_detect.get_face_emb_all(img_arr, bounding_boxes=bounding_boxes)

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
