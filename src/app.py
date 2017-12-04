# This is a _very simple_ example of a web service that recognizes faces in uploaded images.
# Upload an image file and it will check if the image contains a picture of face.
# For example:
#
# $ curl -F "image=@/path/to/image.jpg" -o "/path/to/image1.jpg" http://127.0.0.1:5001/face/detect
#
# Return detected file to output position
#
# Upload two image file and check if these images contain same face.
# For example:
#
# $ curl -F "image_1=@/path/to/image_1.jpg" -F "image_2=@/path/to/image_2.jpg" http://127.0.0.1:5001/face/verify
#
# Returns:
#
# {
#   "distance": 0.5756304860115051,
#   "issame": true,
#   "thresholds": 1.2
# }
#
# the smaller distance, the more similarity the two faces
# 1.2 is the thresholds, smaller distance represents same face, and vice versa.
import sys
import os.path

sys.path.append(os.path.dirname(__file__))

from flask import Flask, json, request, Response, jsonify
from face_detect import FaceDetect
from face_compare import FaceVerify
from face_video import FaceVideo

# You can change this to any folder on your system
ALLOWED_EXTENSIONS = {'png', 'jpg', 'mp4', 'avi'}
VIDEO_EXTENSIONS = {'mp4', 'avi'}

app = Flask(__name__)

detect_obj = FaceDetect()
verify_obj = FaceVerify()
detect_video_obj = FaceVideo(detect_obj, verify_obj)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_video(filename):
    return filename.rsplit('.', 1)[1].lower() in VIDEO_EXTENSIONS


def transfer_response(message, status_code):
    predict_results = json.dumps({'message': message}, ensure_ascii=False)
    return Response(
        response=predict_results,
        mimetype="application/json; charset=UTF-8",
        status=status_code
    )


def verify(req, names):
    if req.method != 'POST':
        raise Exception('request should be POST')

    for name in names:
        if name not in req.files:
            raise Exception('form-data key should be {}'.format(name))

        file = req.files[name]
        if file.filename == '':
            raise Exception('filename is empty')

        if not allowed_file(file.filename):
            raise Exception('file extension should be {}'.format(ALLOWED_EXTENSIONS))


@app.route('/')
def abc():
    return 'this is a test'


@app.route('/face/detect', methods=['GET', 'POST'])
def face_detect():
    try:
        key_names = ['file']
        verify(request, key_names)

        file_stream = request.files[key_names[0]]

        if is_video(file_stream.filename):
            send_obj = detect_video_obj.detect_faces_in_video(file_stream)
        else:
            send_obj = detect_obj.detect_faces_in_image(file_stream)
    except Exception as exp:
        return transfer_response(str(exp), 400)

    return send_obj


@app.route('/face/verify', methods=['GET', 'POST'])
def face_comapre():
    try:
        key_names = ['file_1', 'file_2']
        verify(request, key_names)
        image_files = [request.files[key] for key in key_names]
        json_obj = verify_obj.compare_face_in_image(detect_obj, image_files)

    except Exception as exp:
        return transfer_response(str(exp), 400)

    return jsonify(json_obj)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
