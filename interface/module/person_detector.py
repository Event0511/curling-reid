#!/usr/bin/env python 
# -*- coding:utf-8 -*-
__author__ = 'MSI'

import copy
import cv2
import numpy as np


class PersonDetector(object):
    def __init__(self):
        super().__init__()
        self.img = None
        self.keypoints = None
        self.rects = None

    def update(self, result, keypoints):
        self.img = result
        self.keypoints = keypoints
        self.rects = self.get_rects()

    def get_people(self):
        if self.keypoints[0].size == 1:
            return
        people_position = []
        people_img = []
        for rect in self.rects:
            x, y, w, h = rect
            people_position.append((x+w/2, y+h/2))
            people_img.append(self.img[y:y+h, x:x+w])
        return people_position, people_img

    def get_rects(self):
        if self.keypoints[0].size == 1:
            return
        rects = []
        bodys = self.keypoints[0]
        if bodys[0].size >= 1:
            for body in bodys:  # 每只手
                rect = self.body_bbox(body)
                if rect:
                    rects.append(rect)
        return rects

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
