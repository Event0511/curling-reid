#!/usr/bin/env python 
# -*- coding:utf-8 -*-
__author__ = 'Eventory'

import cv2
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torch.autograd import Variable
from torchvision import datasets, transforms
from tracking.yolov3.models import Darknet
from tracking.yolov3.utils import load_classes, non_max_suppression
from tracking.tracker.sort import SORT
from tqdm import tqdm
# from interface.module.controller import Controller


class PersonTracker(object):
    def __init__(self, img_size=416):
        super().__init__()
        self.model = Darknet('../tracking/cfg/yolov3.cfg', img_size=img_size)
        self.model.load_weights('../tracking/data/yolov3.weights')
        self.model.cuda()
        self.model.eval()
        self.classes = load_classes('../tracking/cfg/coco.names')
        self.tracker = SORT()

    def track(self, frame, img_size=416, area_threshold=0.005):
        fh, fw, _ = frame.shape
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pilimg = Image.fromarray(frame)
        # detection
        # _start_time = datetime.now()
        detections = self.detect_image(pilimg, img_size=img_size)
        # _cost_time = datetime.now() - _start_time

        # image and bbox transition
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        image = np.array(pilimg)
        pad_x = max(image.shape[0] - image.shape[1], 0) * (img_size / max(image.shape))
        pad_y = max(image.shape[1] - image.shape[0], 0) * (img_size / max(image.shape))
        unpad_h = img_size - pad_y
        unpad_w = img_size - pad_x

        rects = []
        obj_ids = []
        confs = []

        if detections is not None:
            # logger.debug('detect frame {} in {}, get detections {}'.format(
            #     frame_idx+1, str(_cost_time), detections.shape))
            # print('detect frame {} in {}, get detections {}'.format(
            #     frame_idx + 1, str(_cost_time), detections.shape))
            # , dets[num_trackers-1, 4]
            tracked_detections = self.tracker.update(detections.cpu())
            unique_labels = detections[:, -1].cpu().unique()
            num_unique_labels = len(unique_labels)
            # calculate detection result info, and filter out according to label and threshold
            for x1, y1, x2, y2, obj_id, cls_pred, conf_prd in tracked_detections:
                label = self.classes[int(cls_pred)]
                if label != 'person':
                    continue
                box_h = int(((y2 - y1) / unpad_h) * frame.shape[0])
                box_w = int(((x2 - x1) / unpad_w) * frame.shape[1])
                if (box_w * box_h) / (fw * fh) < area_threshold:
                    continue
                y1 = int(((y1 - pad_y // 2) / unpad_h) * frame.shape[0])
                x1 = int(((x1 - pad_x // 2) / unpad_w) * frame.shape[1])
                rects.append((x1, y1, box_w, box_h))
                obj_ids.append(obj_id)
                confs.append(conf_prd)
            # pos, imgs = PersonTracker.get_people(frame, rects)
        return frame, obj_ids, rects, confs

    def detect_image(self, img, img_size=416, conf_threshold=0.6, nms_threshold=0.5):
        # resize and pad image
        ratio = min(img_size / img.size[0], img_size / img.size[1])
        imw = round(img.size[0] * ratio)
        imh = round(img.size[1] * ratio)
        img_transforms = transforms.Compose([
            transforms.Resize((imh, imw)),
            transforms.Pad((
                max(int((imh - imw) / 2), 0),
                max(int((imw - imh) / 2), 0)), fill=(128, 128, 128)),
            transforms.ToTensor(),
        ])

        # convert image to Tensor
        Tensor = torch.cuda.FloatTensor
        tensor = img_transforms(img).float()
        tensor = tensor.unsqueeze_(0)
        input_image = Variable(tensor.type(Tensor))

        # run inference on the model and get detections
        with torch.no_grad():
            detections = self.model(input_image)
            detections = non_max_suppression(detections, 80, conf_threshold, nms_threshold)
        return detections[0]

    @staticmethod
    def get_people(frame, rects):
        people_position = []
        people_img = []
        fh, fw, _ = frame.shape
        if len(rects) == 0:
            return people_position, people_img
        for rect in rects:
            x, y, w, h = rect
            people_position.append((x+w/2, y+h/2))
            y1 = max(0, min(y, fh - 1))
            y2 = max(0, min(y+h, fh - 1))
            x1 = max(0, min(x, fw - 1))
            x2 = max(0, min(x+w, fw - 1))
            people_img.append(frame[y1:y2, x1:x2])
        return people_position, people_img

    @staticmethod
    def people_select(positions, people, areas):
        select_people = []
        for i in range(len(positions)):
            for area in areas:
                if area.has(positions[i]):
                    select_people.append(people[i])

    @staticmethod
    def body_bbox(body):
        if np.sum(body[:, 2]) > 25*0.3:
            body = np.array(body[:, :2])
            x = body.nonzero()
            v = np.array(body[x]).reshape(-1, 2)
            return cv2.boundingRect(v)
        else:
            return None


class DetectArea(object):
    def __init__(self, x, y, w, h):
        super().__init__()
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def has(self, position):
        px, py = position
        if px in range(self.x, self.x + self.w) and py in range(self.y, self.y + self.h):
            return True
        else:
            return False

    def __call__(self):
        return self.x, self.y, self.w, self.h


if __name__ == '__main__':
    cap = cv2.VideoCapture('../media/curling-HD.avi')
    cv2.namedWindow('detect result')
    video_nframe = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    tracker = PersonTracker()
    # controller = Controller()
    # loop over the video
    for frame_idx in tqdm(range(int(video_nframe))):
    # while True:
        ok, frame = cap.read()
        if not ok:
            break
        result, obj_ids, rects, confs = tracker.track(frame)
        # result = controller.draw_rects(result)
        cv2.imshow('detect result', result)
        cv2.waitKey(1)

