import dlib
import os
import numpy as np

cnn_path = os.path.join(os.path.dirname(__file__), '../../data/dlib_model/mmod_human_face_detector.dat')

face_rec_model_path = os.path.join(os.path.dirname(__file__),
                                   '../../data/dlib_model/dlib_face_recognition_resnet_model_v1.dat')

predictor_path5 = os.path.join(os.path.dirname(__file__), '../../data/dlib_model/shape_predictor_5_face_landmarks.dat')


class DlibFaceRecognize:
    def __init__(self):
        self.cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_path)
        self.facerec = dlib.face_recognition_model_v1(face_rec_model_path)
        self.predictor5 = dlib.shape_predictor(predictor_path5)

    def detect_face(self, img):
        dets = self.cnn_face_detector(img, 1)
        bounding_boxes = [(det.rect.left(), det.rect.top(), det.rect.right(), det.rect.bottom()) for det in dets]

        return bounding_boxes

    def trans_image_emb(self, img, bounding_box):
        det = dlib.rectangle(bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3])

        shape = self.predictor5(img, det)

        face_descriptor = self.facerec.compute_face_descriptor(img, shape)
        return np.array(face_descriptor)
