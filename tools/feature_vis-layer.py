# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import multiprocessing as mp
import os

import cv2
import detectron2.data.transforms as T
import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.modeling.poolers import ROIPooler
from sklearn.manifold import TSNE
from tqdm import tqdm

from defrcn.dataloader import build_detection_test_loader
from main import Trainer
from skimage import io
from torch import nn
import matplotlib.pyplot as plt


class FeatureExtract(object):
    """
    1: 网络不更新梯度,输入需要梯度更新
    2: 使用目标类别的得分做反向传播
    """

    def __init__(self, net, layer_name, cfg):
        self.net = net
        self.layer_name = layer_name
        # save feature
        self.feature = []
        # save gradient
        self.gradient = None
        self.net.eval()
        self.handlers = []
        self.ts = TSNE(n_components=2, init='pca', random_state=42, perplexity=8)


        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON

        self.pooler = net.roi_heads.pooler

        self._register_hook()

    def _get_features_hook(self, module, input, output):
        self.feature.append(output)

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                # self.handlers.append(module.register_backward_hook(self._get_grads_hook))
                return True
        print(f"Layer {self.layer_name} not found in Model!")

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, inputs):
        """

        :param inputs: {"image": [C,H,W], "height": height, "width": width}
        # :param index: 第几个边框
        :return:
        """

        self.net.zero_grad()
        _ = self.net.inference(inputs)
        return

    def draw_tsne(self, labels, boxes, method_name):
        select = tuple(int(label) for label in labels)
        colors = plt.get_cmap('tab20')(select)
        edge_color = ['w'] * int(len(labels) * 0.6) + ['black'] * int(len(labels) * 0.4)
        # 如果每次只sample一个样本的话，在process data时就不会让不同尺寸的样本对齐，之后再stack就会出问题
        box_feat = []
        for i in range(len(labels)):
            feat = self.feature[i]
            box = boxes[i]
            box_feat.append(self.pooler([feat], [box.to('cuda')]).squeeze())
        box_feat = torch.stack(box_feat).view(30, -1).to('cpu')
        tsne_feature = self.ts.fit_transform(box_feat)
        print("finish fitting")
        plt.scatter(tsne_feature[:, 0], tsne_feature[:, 1], c=colors, edgecolors=edge_color)
        plt.title(f't-SNE Visualization of {method_name}')
        plt.savefig(f'tsne_{method_name}.svg', format='svg', bbox_inches='tight')


def tsne(cfg, model, method_name):
    data_loader = build_detection_test_loader(cfg, "rdd_trainval_all1_3shot_seed6")
    print(f"data length {len(data_loader)}")

    # layer_name = "roi_heads.box_predictor.cls_score"
    # layer_name ="proposal_generator.rpn_head.objectness_logits"
    # layer_name = "proposal_generator.rpn_head.anchor_deltas"
    layer_name = "backbone.res4.22.conv3.norm"
    fe = FeatureExtract(model, layer_name, cfg)
    labels = []
    boxes = []
    with torch.no_grad():
        tqdm_loader = tqdm(data_loader)
        for (i_iter, input) in enumerate(tqdm_loader):
            labels.append(input[0]["instances"]._fields["gt_classes"])
            boxes.append(input[0]["instances"]._fields["gt_boxes"])
            fe(input)
            tqdm_loader.set_description(f"sample_number {i_iter + 1}")
    fe.draw_tsne(labels, boxes, method_name)


# def hist(cfg, model):
#     evaluator =
#     data_loader = build_detection_test_loader(cfg, "rdd_test")


# def extract_roi_features(self, img, boxes):
#     """
#     :param img:
#     :param boxes:
#     :return:
#     """
#     mean = torch.tensor([0.406, 0.456, 0.485]).reshape((3, 1, 1)).to(self.device)
#     std = torch.tensor([[0.225, 0.224, 0.229]]).reshape((3, 1, 1)).to(self.device)
#
#     img = img.transpose((2, 0, 1))
#     img = torch.from_numpy(img).to(self.device)
#     images = [(img / 255. - mean) / std]
#     images = ImageList.from_tensors(images, 0)
#     conv_feature = self.imagenet_model(images.tensor[:, [2, 1, 0]])[1]  # size: BxCxHxW
#     box_features = self.roi_pooler([conv_feature],boxes).squeeze(2).squeeze(2)
#     activation_vectors = box_features
#
#     # box_features = self.roi_pooler([conv_feature], boxes).squeeze(2).squeeze(2)
#     # activation_vectors = self.imagenet_model.fc(box_features)
#     return activation_vectors


if __name__ == "__main__":
    path = [
        "/home/wxq/od/DeFRCN/checkpoints/dynamic/1/3shot_seed6/",
        "/home/wxq/od/DeFRCN/checkpoints/upsample/1/3shot_seed6/",
        "/home/wxq/od/DeFRCN/checkpoints/rdd2/defrcn_gfsod_r101_novel1/tfa-like/3shot_seed6/"
    ]

    method_name = [
        "dynamic",
        "upsample",
        "FSRDD"
    ]

    select = 1

    root_path = path[select]
    cfg = get_cfg()
    cfg.merge_from_file(root_path + "config.yaml")
    cfg.MODEL.WEIGHTS = root_path + "model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 设置阈值，用于过滤低置信度的预测

    # 构建模型
    model = Trainer.build_model(cfg)
    # 加载权重
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    tsne(cfg, model, method_name[select])
