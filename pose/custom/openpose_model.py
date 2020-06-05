import os
import sys
import cv2
import numpy as np
from sys import platform

dir_path = os.path.dirname(os.path.dirname(__file__))

try:
    sys.path.append(dir_path + '\\openpose')
    print(dir_path)
    os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '\\openpose;' + dir_path + '\\3rdparty;'
    import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. \
            Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e


class OpenposeModel(object):
    def __init__(self, hand=False, face=False):
        self.op_params = {
            "model_folder": dir_path + "\models",
            "body": 1,
            "render_pose": 1,
            "render_threshold": 0.1,
            "hand": hand,
            "hand_render": 1,
            "hand_render_threshold": 0.2,
            "face": face,
            "face_render": 1,
            "face_render_threshold": 0.4,
            "disable_blending": False  # black blackgroud if True
        }

        self.datum = op.Datum()
        self.op_wrapper = op.WrapperPython()
        self.op_wrapper.configure(self.op_params)
        self.op_wrapper.start()

    def update_wrapper(self, key, value):
        self.op_params[key] = value
        self.op_wrapper.configure(self.op_params)
        self.op_wrapper.start()

    def get_keypoints(self):
        body_keypoints = self.datum.poseKeypoints
        hand_keypoints = self.datum.handKeypoints
        face_keypoints = self.datum.faceKeypoints
        return body_keypoints, hand_keypoints, face_keypoints

    def get_original_image(self):
        return self.datum.cvInputData

    def get_rendered_image(self):
        return self.datum.cvOutputData

    def __call__(self, img):
        self.datum.cvInputData = img
        self.op_wrapper.emplaceAndPop([self.datum])
        result = self.datum.cvOutputData
        pose_ids = self.datum.id
        # pose_ids = None

        pose = self.datum.poseKeypoints
        hand = self.datum.handKeypoints
        face = self.datum.faceKeypoints
        # return result, (pose, hand, face)
        return result, pose_ids, (pose, hand, face)

    @staticmethod
    def body_bbox(body):
        if np.sum(body[:, 2]) > 25*0.5:
            body = np.array(body[:, :2])
            x = body.nonzero()
            v = np.array(body[x]).reshape(-1, 2)
            return cv2.boundingRect(v)
        else:
            return None
