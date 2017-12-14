import face_detect
import face_compare
from scipy import misc
from PIL import Image
from io import BytesIO
from flask import send_file


def trans_file_arr(file_streams):
    nrof_samples = len(file_streams)
    img_list = []
    for i in range(nrof_samples):
        file_stream = file_streams[i]
        img_arr = misc.imread(file_stream, mode='RGB')
        img_list.append((img_arr, file_stream.filename))

    return img_list


def detect_faces_in_image(file_stream):
    img_arr, img_name = trans_file_arr([file_stream])[0]

    bounding_boxes = face_detect.detect_faces_core(img_arr, img_name)

    img_arr = face_compare.draw_for_image(img_arr, bounding_boxes)

    img = Image.fromarray(img_arr)

    byte_io = BytesIO()

    img.save(byte_io, 'PNG')
    byte_io.seek(0)

    return send_file(byte_io, mimetype='image/png')


def compare_face_in_image(image_files):
    img_objs = trans_file_arr(image_files)
    img_arrs = [img_obj[0] for img_obj in img_objs]

    return face_compare.compare_face_in_image(img_arrs)


def compare_two_faces_in_image(file_stream):
    img_arr, img_name = trans_file_arr([file_stream])[0]

    return face_compare.compare_two_faces_in_image(img_arr)
