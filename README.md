This is a _very simple_ example of a web service that recognizes faces in uploaded images.
Upload an image file and it will check if the image contains a picture of face.
For example:

$ curl -F "image=@/path/to/image.jpg" -o "/path/to/image1.jpg" http://127.0.0.1:5001/face/detect

Return detected file to output position

Upload two image file and check if these images contain same face.
For example:

$ curl -F "image_1=@/path/to/image_1.jpg" -F "image_2=@/path/to/image_2.jpg" http://127.0.0.1:5001/face/verify

Returns:

{
  "distance": 0.5756304860115051,
  "issame": true,
  "thresholds": 1.2
}

the smaller distance, the more similarity the two faces
1.2 is the thresholds, smaller distance represents same face, and vice versa.

To run the app
cd /face_interface/src
export FLASK_APP=app.py
flask run --host=0.0.0.0 --port=5000 --with-threads

based on git@github.com:davidsandberg/facenet.git
model data can be download there