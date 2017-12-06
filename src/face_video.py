import cv2
import uuid
import os
import shutil
import numpy as np
from io import BytesIO
from flask import send_file

known_images_dir = '/tmp/face/known_images'


class FaceVideo:
    def __init__(self, detect_obj=None, compare_obj=None):
        self.detect_obj = detect_obj
        self.compare_obj = compare_obj
        self.known_embs = self.load_known_images()

    def load_known_images(self):
        if not os.path.exists(known_images_dir):
            os.makedirs(known_images_dir)

        name_emb_tuples = []
        for basename in os.listdir(known_images_dir):
            name = basename.rsplit('.', maxsplit=1)[0]
            image_path = os.path.join(known_images_dir, basename)
            img_arr = self.detect_obj.crop_faces_in_image_item(cv2.imread(image_path), basename)[0]

            emb = self.compare_obj.inception_resnet_v1([img_arr])[0]
            name_emb_tuples.append((name, emb))
        return name_emb_tuples

    def detect_faces_in_video(self, file_stream):
        tmp_video_dir = '/tmp/face/{}'.format(uuid.uuid4())
        if not os.path.isdir(tmp_video_dir):
            os.makedirs(tmp_video_dir)

        input_path = tmp_video_dir + '/input{}'.format(file_stream.filename)
        output_path = tmp_video_dir + '/output{}.avi'.format(file_stream.filename)
        file_stream.save(input_path)
        self.detect_faces_in_video_process(input_path, output_path)

        with open(output_path, mode='rb') as f:
            byte_io = BytesIO(f.read())

        byte_io.seek(0)

        shutil.rmtree(tmp_video_dir)

        video_obj = send_file(byte_io, mimetype='video/avi')
        return video_obj

    def find_name_in_database(self, face_encoding):
        names, embs = zip(*self.known_embs)
        distances = face_distance(np.array(embs), face_encoding)

        min_index = np.argmin(distances)

        print('name: {}, distance: {}'.format(names[min_index], distances[min_index]))

        face_name = None
        if distances[min_index] < self.compare_obj.thresholds:
            face_name = names[min_index]
        return face_name

    def detect_faces_in_video_process(self, input_path, output_path, with_name=True):
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
                face_locations = self.detect_obj.detect_faces_core(frame)
            except Exception:
                pass
            else:
                # add name text
                face_names = []
                if with_name and len(self.known_embs) > 0:
                    crop_img_arrs = self.detect_obj.crop_faces_in_image_item(frame, bounding_boxes=face_locations)
                    face_encodings = self.compare_obj.inception_resnet_v1(crop_img_arrs)

                    for face_encoding in face_encodings:
                        face_name = self.find_name_in_database(face_encoding)

                        face_names.append(face_name)

                # Label the results
                for (left, top, right, bottom, _), name in zip(face_locations, face_names):
                    left = int(left)
                    top = int(top)
                    right = int(right)
                    bottom = int(bottom)
                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (0, 0, 255))

            # Write the resulting image to the output video file
            print("Writing frame {}".format(frame_number))
            output_movie.write(frame)

        # All done!
        input_movie.release()
        output_movie.release()
        cv2.destroyAllWindows()

    def trans_video_to_vector(self, video_path):
        frame_vectors = []

        input_movie = cv2.VideoCapture(video_path)

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
                face_locations = self.detect_obj.detect_faces_core(frame)
            except Exception:
                pass
            else:
                crop_img_arrs = self.detect_obj.crop_faces_in_image_item(frame, bounding_boxes=face_locations)
                face_encodings = self.compare_obj.inception_resnet_v1(crop_img_arrs)

                frame_vectors.append(face_encodings[0])

            print("transfer frame {}".format(frame_number))

        # All done!
        input_movie.release()

        return frame_vectors

    def detect_video_in_database_api(self, video_stream):
        tmp_video_dir = '/tmp/face/{}'.format(uuid.uuid4())
        if not os.path.isdir(tmp_video_dir):
            os.makedirs(tmp_video_dir)

        video_path = os.path.join(tmp_video_dir, video_stream.filename)

        video_stream.save(video_path)

        name = self.detect_video_in_database(video_path)

        shutil.rmtree(tmp_video_dir)

        if name is None:
            obj = {'verify': False}
        else:
            obj = {'verify': True, 'name': name}

        return obj

    def detect_video_in_database(self, video_path):
        video_embs = self.trans_video_to_vector(video_path)

        name_map = dict()

        for video_emb in video_embs:
            name = self.find_name_in_database(video_emb)
            if name is not None:
                name_map[name] = name_map.get(name, 0) + 1

        max_name = None
        if len(name_map) > 0:
            max_name = max(name_map, key=name_map.get)
        return max_name

    def compare_video_image_api(self, video_stream, image_stream):
        tmp_video_dir = '/tmp/face/{}'.format(uuid.uuid4())
        if not os.path.isdir(tmp_video_dir):
            os.makedirs(tmp_video_dir)

        video_path = os.path.join(tmp_video_dir, video_stream.filename)
        image_path = os.path.join(tmp_video_dir, image_stream.filename)

        video_stream.save(video_path)
        image_stream.save(image_path)

        result = self.compare_video_image(video_path, image_path)

        shutil.rmtree(tmp_video_dir)
        return result

    def compare_video_image(self, video_path, image_path):
        video_embs = self.trans_video_to_vector(video_path)
        image_emb = self.compare_obj.transfer_image_to_vector(image_path)

        distances = face_distance(np.array(video_embs), image_emb)

        print(distances)

        min_distance = min(distances).astype(np.float64)

        return {'issame': bool(min_distance < self.compare_obj.thresholds), 'distance': min_distance,
                'thresholds': self.compare_obj.thresholds}


def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty(0)
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)
