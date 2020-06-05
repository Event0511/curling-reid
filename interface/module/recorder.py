#!/usr/bin/env python 
# -*- coding:utf-8 -*-
__author__ = 'MSI'

import numpy as np
import torch
import cv2
from PIL import Image
from reid.utils.data import transforms as T
from torchvision.transforms import functional as F


class TimingRecorder(object):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    transformer = T.Compose([
        T.RectScale(256, 128),
        T.ToTensor(),
        normalizer,
    ])

    def __init__(self, threshold=180):
        # parameters
        self.img_threshold = threshold
        # image info
        self.img = {}
        # people info
        self.people_id = []  # 人员id
        self.obj2pid = {}  # 人员对应的对象id
        self.people_img = {}  # 各人员过去一段时序上的图像信息

    def add_person(self, person_id, obj_id, person_img, p_threshold=30):
        pilimg = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
        person_img = TimingRecorder.transformer(pilimg)
        # 记录obj_id与pid的对应关系
        self.obj2pid[obj_id] = person_id
        if len(self.obj2pid) > p_threshold:
            self.obj2pid.pop(0)
        # 若首次添加，则初始化对应obj_id和img
        if person_id not in self.people_id:
            self.people_id.append(person_id)
            self.people_img[person_id] = [person_img]
        else:
            # when the img list is full, pop the first one
            if len(self.people_img[person_id]) >= self.img_threshold:
                self.people_img[person_id].pop(0)
            self.people_img[person_id].append(person_img)

    @DeprecationWarning
    def update_img(self, prev, curr):
        self.img['prev'] = self.img['prev'] if prev is None else prev
        self.img['curr'] = curr

    def __call__(self):
        pids = []
        imgs = None
        for pid in self.people_img.keys():
            for person in self.people_img[pid]:
                pids.append(pid)
                person = person.view(1, 3, 256, 128)
                if imgs is None:
                    imgs = person
                else:
                    imgs = torch.cat((imgs, person), dim=0)
        uids = list(range(len(pids)))
        return uids, pids, imgs

    @staticmethod
    def pack_query_data(people_imgs):
        """
        pack the query data read by opencv
        :param imgs: image data waiting to query
        :return:
        """
        uids = list(range(-len(people_imgs), 0))
        pids = list(range(-len(people_imgs), 0))
        imgs = None

        for person_img in people_imgs:
            pil_img = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
            img = TimingRecorder.transformer(pil_img)
            img = img.view(1, 3, 256, 128)
            if imgs is None:
                imgs = img
            else:
                imgs = torch.cat((imgs, img), dim=0)

        return uids, pids, imgs


if __name__ == '__main__':
    pass
