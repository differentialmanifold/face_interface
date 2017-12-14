This is a _very simple_ example of a web service that recognizes faces in uploaded images.
Upload an image file and it will check if the image contains a picture of face.
For example:

$ curl -F "file=@/path/to/file.jpg" -o "/path/to/outputfile.jpg" http://127.0.0.1:5001/face/detect

Return detected file to output position

Upload two image file and check if these images contain same face.
For example:

$ curl -F "file_1=@/path/to/file_1.jpg" -F "file_2=@/path/to/file_2.jpg" http://127.0.0.1:5001/face/compare

Returns:

{
    "code": 0,
    "data": {
        "issame": false,
        "similarity": 32.00019383010613
    },
    "message": "ok"
}

the smaller distance, the more similarity the two faces
1.2 is the thresholds, smaller distance represents same face, and vice versa.

Compare two faces in one image:
$ curl -F "file=@/path/to/file.jpg" http://127.0.0.1:5001/face/compare/twofaces

Compare video and face:
$ curl -F "file_1=@/path/to/video.mp4" -F "file_2=@/path/to/image.jpg" http://127.0.0.1:5001/face/compare/video_image

Detect face in video exist in database
$ curl -F "file=@/path/to/file.mp4" http://127.0.0.1:5001/face/compare/database/video 

To run the app
cd /face_interface/src
export FLASK_APP=app.py
flask run --host=0.0.0.0 --port=5001 --with-threads

based on git@github.com:davidsandberg/facenet.git
model data can be download there