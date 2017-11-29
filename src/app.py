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

# You can change this to any folder on your system
ALLOWED_EXTENSIONS = {'png', 'jpg'}

app = Flask(__name__)

detect_obj = FaceDetect()
verify_obj = FaceVerify()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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
            raise Exception('file should be jpg or png')


@app.route('/')
def abc():
    return 'this is a test'


@app.route('/face/detect', methods=['GET', 'POST'])
def face_detect():
    try:
        key_names = ['image']
        verify(request, key_names)

        send_obj = detect_obj.detect_faces_in_image(request.files[key_names[0]])
    except Exception as exp:
        return transfer_response(str(exp), 400)

    return send_obj


@app.route('/face/verify', methods=['GET', 'POST'])
def face_comapre():
    try:
        key_names = ['image_1', 'image_2']
        verify(request, key_names)
        image_files = [request.files[key] for key in key_names]
        json_obj = verify_obj.compare_face_in_image(detect_obj, image_files)

    except Exception as exp:
        return transfer_response(str(exp), 400)

    return jsonify(json_obj)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
