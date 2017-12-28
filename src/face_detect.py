from face_exception import FaceException

from algorithm.facenet_model import FacenetFaceRecognize

face_recognize = FacenetFaceRecognize()
thresholds = 1.2


# from algorithm.dlib_model import DlibFaceRecognize
#
# face_recognize = DlibFaceRecognize()
# thresholds = 0.6


def detect_faces_core(img_arr, img_name=''):
    bounding_boxes = face_recognize.detect_face(img_arr)

    if len(bounding_boxes) == 0:
        raise FaceException(FaceException.NO_FACE_ERR_CODE, 'Can not detect face from image {}'.format(img_name))

    bounding_boxes = [(int(left), int(top), int(right), int(bottom)) for (left, top, right, bottom) in
                      bounding_boxes]

    return bounding_boxes


def find_largest_face(bounding_boxes):
    bounding_box = max(bounding_boxes, key=lambda rect: (rect[2] - rect[0]) * (rect[3] - rect[1]))
    return bounding_box


def get_face_emb_all(img_arr, filename=None, bounding_boxes=None):
    if bounding_boxes is None:
        bounding_boxes = detect_faces_core(img_arr, filename)

    embs = []
    for bb in bounding_boxes:
        emb = face_recognize.trans_image_emb(img_arr, bb)
        embs.append(emb)

    return embs


def get_face_emb_largest(img_arr, filename=None):
    bounding_boxs = detect_faces_core(img_arr, filename)
    max_bb = find_largest_face(bounding_boxs)

    emb = face_recognize.trans_image_emb(img_arr, max_bb)
    return emb
