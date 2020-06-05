#!/usr/bin/env python 
# -*- coding:utf-8 -*-
__author__ = 'MSI'

import cv2 as cv
import torch
import reid.models as models
import numpy as np
import os.path as osp
from torch import nn
from reid.feature_extraction import extract_cnn_feature
from reid.models.embedding import EltwiseSubEmbed
from reid.models.multi_branch import SiameseNet
from reid.utils import to_numpy
from collections import OrderedDict

from interface.module.recorder import TimingRecorder


class ReIDEvaluator(object):
    def __init__(self, path):
        super().__init__()
        self.model_path = path
        self.model = self.load_model()
        self.distance_threshold = 50

    def load_model(self, model_arch='resnet50'):
        # Create model
        # # 利用resnet50建立base_model
        base_model = models.create(model_arch, cut_at_pooling=True)
        # # 建立嵌入模型
        embed_model = EltwiseSubEmbed(use_batch_norm=True, use_classifier=True,
                                      num_features=2048, num_classes=2)
        model = SiameseNet(base_model, embed_model)
        model = nn.DataParallel(model).cuda()
        checkpoint = load_checkpoint(self.model_path)
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        model.load_state_dict(checkpoint)
        return torch.nn.DataParallel(base_model).cuda()

    def query(self, query, gallery, dist_threshold=40):
        """
        Calculate and sort the pairwise distances between query images and gallery images
        :param query: target images (tensor type)
        :param gallery: gallery images (tensor type)
        :param dist_threshold: images whose distance further than this will be recognized as a new one
        :return: list of result person ids
        """
        features, labels = extract_features(self.model, gallery)
        quids, qpids, _ = query
        guids, gpids, _ = gallery
        # 计算query与其gallery的distance
        distmat = pairwise_distance(features, zip(quids, qpids), zip(guids, gpids))
        # Sort according to the first stage distance
        distmat = to_numpy(distmat)
        rank_indices = np.argsort(distmat, axis=1)
        result_ids = []
        result_dist = []
        candidates = []
        for target, rank in enumerate(rank_indices):
            # for index in rank:
            #     print("Pairwise distance of target {:} and person {:} = {:}".format(target,\
            #  gpids[index], distmat[target][index]))
            frank = filer_rank(qpids, gpids, rank)
            result_dist.append(distmat[target][frank[0]])
            candidates.append([gpids[r] for r in frank])
            if distmat[target][frank[0]] < dist_threshold:
                result_ids.append(gpids[frank[0]])
            else:
                result_ids.append('0')
        return result_ids, result_dist, candidates

    @staticmethod
    def data_cat(dataA, dataB):
        Auids, Apids, Aimgs = dataA
        Buids, Bpids, Bimgs = dataB
        uids = Auids + Buids
        pids = Apids + Bpids
        imgs = torch.cat((Aimgs, Bimgs), dim=0)
        return uids, pids, imgs


def filer_rank(qpids, gpids, rank):
    n_rank = []
    for r in rank:
        if gpids[r] not in qpids:
            n_rank.append(r)
    return n_rank


def extract_features(model, data):
    model.eval()
    features = OrderedDict()
    labels = OrderedDict()
    uids, pids, imgs = data
    outputs = extract_cnn_feature(model, imgs)

    for uid, pid, output in zip(uids, pids, outputs):
        features[uid] = output
        labels[uid] = pid

    return features, labels


def pairwise_distance(features, query, gallery):
    x = torch.cat([features[f].unsqueeze(0) for f, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)

    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist


def load_checkpoint(fpath):
    if osp.isfile(fpath):
        checkpoint = torch.load(fpath)
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))


if __name__ == '__main__':
    project_path = 'E:\\workspace\\GraduationProject\\FD-GAN\\'
    model_path = project_path + 'reid\\models\\stageIII\\best_net_E.pth'
    recorder = TimingRecorder()
    evaluator = ReIDEvaluator(model_path)
    p1 = cv.imdecode(np.fromfile(project_path+'datasets\\market1501\\images\\00000001_00_0001.jpg', dtype=np.uint8), -1)
    p2 = cv.imdecode(np.fromfile(project_path+'datasets\\market1501\\images\\00000001_00_0002.jpg', dtype=np.uint8), -1)
    p3 = cv.imdecode(np.fromfile(project_path+'datasets\\market1501\\images\\00000002_00_0001.jpg', dtype=np.uint8), -1)
    p4 = cv.imdecode(np.fromfile(project_path+'datasets\\market1501\\images\\00000003_00_0001.jpg', dtype=np.uint8), -1)
    p5 = cv.imdecode(np.fromfile(project_path+'datasets\\market1501\\images\\00000001_00_0003.jpg', dtype=np.uint8), -1)
    for i in range(10):
        recorder.add_person('001', p2)
        recorder.add_person('002', p3)
        recorder.add_person('003', p4)
        recorder.add_person('001', p5)
    q = recorder.pack_query_data([p1])
    g = ReIDEvaluator.data_cat(q, recorder())

    evaluator.query(q, g)
