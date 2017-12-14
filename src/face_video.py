import cv2
import uuid
import os
import shutil
import numpy as np
from io import BytesIO
from flask import send_file
import face_detect
import face_compare


def detect_faces_in_video(file_stream):
    tmp_video_dir = '/tmp/face/{}'.format(uuid.uuid4())
    if not os.path.isdir(tmp_video_dir):
        os.makedirs(tmp_video_dir)

    input_path = tmp_video_dir + '/input{}'.format(file_stream.filename)
    output_path = tmp_video_dir + '/output{}.avi'.format(file_stream.filename)
    file_stream.save(input_path)
    detect_faces_in_video_process(input_path, output_path)

    with open(output_path, mode='rb') as f:
        byte_io = BytesIO(f.read())

    byte_io.seek(0)

    shutil.rmtree(tmp_video_dir)

    video_obj = send_file(byte_io, mimetype='video/avi')
    return video_obj


def detect_faces_in_video_process(input_path, output_path):
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
            face_locations = face_detect.detect_faces_core(frame)
        except Exception:
            pass
        else:
            frame = face_compare.draw_for_image(frame, face_locations, with_name=True)

        # Write the resulting image to the output video file
        print("Writing frame {}".format(frame_number))
        output_movie.write(frame)

    # All done!
    input_movie.release()
    output_movie.release()
    cv2.destroyAllWindows()


def trans_video_to_vector(video_path):
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
            face_encoding = face_compare.get_largest_face_embs([frame])[0]
            frame_vectors.append(face_encoding)
        except Exception:
            print("frame {} has no face".format(frame_number))

        print("transfer frame {}".format(frame_number))

    # All done!
    input_movie.release()

    return frame_vectors


def detect_video_in_database_api(video_stream):
    tmp_video_dir = '/tmp/face/{}'.format(uuid.uuid4())
    if not os.path.isdir(tmp_video_dir):
        os.makedirs(tmp_video_dir)

    video_path = os.path.join(tmp_video_dir, video_stream.filename)

    video_stream.save(video_path)

    name = detect_video_in_database(video_path)

    shutil.rmtree(tmp_video_dir)

    if name is None:
        obj = {'verify': False}
    else:
        obj = {'verify': True, 'name': name}

    return obj


def detect_video_in_database(video_path):
    video_embs = trans_video_to_vector(video_path)

    name_map = dict()

    for video_emb in video_embs:
        name = face_compare.face_verify.find_name_in_database(video_emb)
        if name is not None:
            name_map[name] = name_map.get(name, 0) + 1

    max_name = None
    if len(name_map) > 0:
        max_name = max(name_map, key=name_map.get)
    return max_name


def compare_video_image_api(video_stream, image_stream):
    tmp_video_dir = '/tmp/face/{}'.format(uuid.uuid4())
    if not os.path.isdir(tmp_video_dir):
        os.makedirs(tmp_video_dir)

    video_path = os.path.join(tmp_video_dir, video_stream.filename)
    image_path = os.path.join(tmp_video_dir, image_stream.filename)

    video_stream.save(video_path)
    image_stream.save(image_path)

    result = compare_video_image(video_path, image_path)

    shutil.rmtree(tmp_video_dir)
    return result


def compare_video_image(video_path, image_path):
    video_embs = trans_video_to_vector(video_path)
    image_emb = face_compare.get_largest_face_embs([cv2.imread(image_path)])[0]

    distances = face_compare.face_distance_arr(np.array(video_embs), image_emb)

    print(distances)

    dist = min(distances)

    thresholds = face_compare.thresholds

    issame = bool(dist < thresholds)

    similarity = max(0.0, 100 - (50 / thresholds ** 2) * float(dist) ** 2)

    return {'issame': issame, 'similarity': similarity}
