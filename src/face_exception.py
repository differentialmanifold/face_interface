class FaceException(Exception):
    NORMAL_CODE = 0

    WRONG_EXTENSION_CODE = 100

    WRONG_REQUEST_CODE = 200

    FILE_NAME_ERR_CODE = 300

    PARAMETER_NUM_ERR_CODE = 400

    FACE_NUM_ERR_CODE = 500

    NO_FACE_ERR_CODE = 600

    OTHER_ERR_CODE = 1000

    def __init__(self, errcode, message):
        self.errcode = errcode
        self.message = message
