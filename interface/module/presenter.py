#!/usr/bin/env python 
# -*- coding:utf-8 -*-
__author__ = 'MSI'

import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

from interface.module.recorder import TimingRecorder
from interface.module.reid_evaluator import ReIDEvaluator
from interface.module.person_tracker import PersonTracker


class Presenter(object):
    def __init__(self, model_path, conf_threshold):
        super().__init__()
        self.recorder = TimingRecorder()
        self.tracker = PersonTracker()
        # self.detector = PersonDetector()
        self.evaluator = ReIDEvaluator(model_path)
        # parameters
        self.conf_threshold = conf_threshold
        # attributes
        self.area = None
        self.current_id = 0
        self.frame_pids = []
        self.frame_rects = None
        self.confs = None
        # draw setting
        self.cmap = plt.get_cmap('tab20b')
        self.bbox_palette = [self.cmap(i)[:3] for i in np.linspace(0, 1, 1000)]
        random.shuffle(self.bbox_palette)

    def generate_id(self):
        pid = str(self.current_id).zfill(3)
        self.current_id += 1
        return pid

    # 分割当前帧的“新”“旧”obj_id人员对象（跟踪系统判定的新人员）
    def divide_person_index(self, obj_ids):
        frame_ids = []
        new_indexs = []
        old_indexs = []
        for index, obj_id in enumerate(obj_ids):
            # 若上一帧有该对象，说明是在跟踪人员，无需重识别，将obj_id加入frame_ids
            if obj_id in self.frame_pids:
                frame_ids.append(obj_id)
                old_indexs.append(index)
            # 若上一帧无对象，但已有对应关系，说明是已知人员，无需重识别，将对应ID加入frame_ids
            elif obj_id in self.recorder.obj2pid.keys():
                frame_ids.append(self.recorder.obj2pid[obj_id])
                old_indexs.append(index)
            # 若上一帧无对象，且无对应关系，记录原obj_id，加入“新”人员序号列表并返回
            else:
                frame_ids.append(obj_id)
                new_indexs.append(index)
        return frame_ids, new_indexs, old_indexs

    def draw_rects(self, frame, label='person'):
        for pid, rect, conf in zip(self.frame_pids, self.frame_rects, self.confs):
            x1, y1, box_w, box_h = rect
            color = self.bbox_palette[int(pid) % len(self.bbox_palette)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame,
                          (x1, y1),
                          (x1 + box_w, y1 + box_h),
                          color, 2)
            cv2.rectangle(frame,
                          (x1, y1 - 35),
                          (x1 + len(label) * 19 + 60, y1),
                          color, -1)
            cv2.putText(frame,
                        '{}{}{}'.format(label, pid, '' if conf > self.conf_threshold else '?'),
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 3)
        # rects = self.detector.rects
        # for i in range(len(self.frame_pids)):
        #     x, y, w, h = rects[i]
        #     cv2.rectangle(result, (x, y), (x + w, y + h), (255, 255, 255), 2)
        #     cv2.putText(result, "ID:"+self.frame_pids[i], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame

    def update_people(self, frame):
        result, obj_ids, self.frame_rects, self.confs = self.tracker.track(frame)
        positions, people = self.tracker.get_people(frame, self.frame_rects)
        # 如果库为空，首次仅做记录，不进行识别
        if len(self.recorder.people_img) == 0:
            for obj_id, person in zip(obj_ids, people):
                pid = self.generate_id()
                self.recorder.add_person(pid, obj_id, person)
                self.frame_pids.append(pid)
            return people
        # 如果库非空，且存在待识别区域，则对当前画面所有符合识别条件的人员进行识别
        elif self.area is not None:
            pass
        # Todo: 后续人员的录入和重识别，每一帧判断是否有新objid人员对象，拿出来与库中的人员比较，如果距离过大则判定为新人员，
        # Todo: 录入新id（len(people_img)）和图像；如果识别为原有人员，则更新其图像库
        # 获取当前帧的“新”obj_id人员对象（跟踪系统判定的新人员）
        self.frame_pids, new_indexs, old_indexs = self.divide_person_index(obj_ids)
        if len(new_indexs) > 0:
            # 对所有“新”人员做重识别工作
            targets = []
            for index in new_indexs:
                targets.append(people[index])
            # 设置query集与gallery集
            q = self.recorder.pack_query_data(targets)
            g = ReIDEvaluator.data_cat(q, self.recorder())
            target_ids, dists, candidates = self.evaluator.query(q, g)
            print("target_ids:", target_ids)
            print("dist:", dists)
            # info_zip = zip([self.confs[i] for i in new_indexs], new_indexs, targets)
            info_zip = zip(dists, new_indexs, targets)
            info_zip = sorted(info_zip, reverse=True)
            matched = [self.frame_pids[i] for i in old_indexs]
            # for i, (conf, index, person) in enumerate(info_zip):
            # 按照重识别的匹配距离由小到大匹配
            for i, (dist, index, person) in enumerate(info_zip):
                # 置信度较低时，仅做识别，不做记录
                if target_ids[i] == '0':
                    pid = self.generate_id()
                    if self.confs[i] > self.conf_threshold:
                        self.recorder.add_person(pid, obj_ids[index], person)
                    self.frame_pids[index] = pid
                else:
                    # 若与本次已匹配的人员重复，则查找下一个候选人
                    for j in range(len(candidates[i])):
                        target_ids[i] = candidates[i][j]
                        if target_ids[i] not in matched:
                            print("tid ", target_ids[i], "matched ", matched)
                            break
                    if self.confs[i] > self.conf_threshold:
                        self.recorder.add_person(target_ids[i], obj_ids[index], person)
                    self.frame_pids[index] = target_ids[i]
                matched.append(self.frame_pids[index])

        return people


if __name__ == '__main__':
    dict = {'1':1, '2':2}
    dict2 = {'3':3, '4':4}
    l1 = [0, 2]
    l2 = [1,2,3,4,5,6,7,8]
    print([l2[i] for i in l1])
    # for i, (a, b) in enumerate(zip(dict, dict2)):
    #     print(i, a, b)
