# This is a _very simple_ example of a web service that recognizes faces in uploaded images.
# Upload an image file and it will check if the image contains a picture of face.

import sys
import os.path
from flask import Flask, json, request, Response

sys.path.append(os.path.dirname(__file__))

import face_implement
import face_video
from face_exception import FaceException

# You can change this to any folder on your system
ALLOWED_EXTENSIONS = {'png', 'jpg', 'tif', 'mp4', 'avi'}
VIDEO_EXTENSIONS = {'mp4', 'avi'}

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_video(filename):
    return filename.rsplit('.', 1)[1].lower() in VIDEO_EXTENSIONS


def transfer_response(obj, status_code=200):
    predict_results = json.dumps(obj, ensure_ascii=False)
    return Response(
        response=predict_results,
        mimetype="application/json; charset=UTF-8",
        status=status_code
    )


def verify(req, names):
    if req.method != 'POST':
        raise FaceException(FaceException.WRONG_REQUEST_CODE, 'request should be POST')

    for name in names:
        if name not in req.files:
            raise FaceException(FaceException.FILE_NAME_ERR_CODE, 'form-data key should be {}'.format(name))

        file = req.files[name]
        if file.filename == '':
            raise FaceException(FaceException.FILE_NAME_ERR_CODE, 'filename is empty')

        if not allowed_file(file.filename):
            raise FaceException(FaceException.WRONG_EXTENSION_CODE,
                                'file extension should be {}'.format(ALLOWED_EXTENSIONS))


@app.route('/face/detect', methods=['GET', 'POST'])
def face_detect():
    try:
        key_names = ['file']
        verify(request, key_names)

        file_stream = request.files[key_names[0]]

        if is_video(file_stream.filename):
            send_obj = face_video.detect_faces_in_video(file_stream)
        else:
            send_obj = face_implement.detect_faces_in_image(file_stream)
    except FaceException as exp:
        return transfer_response({'data': None, 'code': exp.errcode, 'message': exp.message})

    return send_obj


@app.route('/face/compare', methods=['GET', 'POST'])
def face_comapre():
    try:
        key_names = ['file_1', 'file_2']
        verify(request, key_names)
        image_files = [request.files[key] for key in key_names]
        json_obj = face_implement.compare_face_in_image(image_files)

    except FaceException as exp:
        return transfer_response({'data': None, 'code': exp.errcode, 'message': exp.message})

    return transfer_response({'data': json_obj, 'code': FaceException.NORMAL_CODE, 'message': 'ok'})


@app.route('/face/compare/twofaces', methods=['GET', 'POST'])
def face_compare_two_faces():
    try:
        key_names = ['file']
        verify(request, key_names)
        image_file = request.files[key_names[0]]
        json_obj = face_implement.compare_two_faces_in_image(image_file)
    except FaceException as exp:
        return transfer_response({'data': None, 'code': exp.errcode, 'message': exp.message})

    return transfer_response({'data': json_obj, 'code': FaceException.NORMAL_CODE, 'message': 'ok'})


@app.route('/face/compare/video_image', methods=['GET', 'POST'])
def face_compare_video_image():
    try:
        key_names = ['file_1', 'file_2']
        verify(request, key_names)
        files = [request.files[key] for key in key_names]
        json_obj = face_video.compare_video_image_api(files[0], files[1])
    except FaceException as exp:
        return transfer_response({'data': None, 'code': exp.errcode, 'message': exp.message})

    return transfer_response({'data': json_obj, 'code': FaceException.NORMAL_CODE, 'message': 'ok'})


@app.route('/face/compare/database/video', methods=['GET', 'POST'])
def face_compare_vedio_in_database():
    try:
        key_names = ['file']
        verify(request, key_names)
        file = request.files[key_names[0]]
        json_obj = face_video.detect_video_in_database_api(file)
    except FaceException as exp:
        return transfer_response({'data': None, 'code': exp.errcode, 'message': exp.message})

    return transfer_response({'data': json_obj, 'code': FaceException.NORMAL_CODE, 'message': 'ok'})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, threaded=True)
