from detectron2.structures import Instances
from detectron2.utils.visualizer import Visualizer
from detectron2.data.detection_utils import read_image
from defrcn.engine.defaults import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetMapper, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from xml.etree import ElementTree as ET
import os
import cv2
import random


def prediction_res(names, rand_range = 0):
    root = '../datasets/RDD'

    for name in names:
        img = cv2.imread(os.path.join(root, 'JPEGImages', name+'.jpg'))

        tree = ET.parse(os.path.join(root, 'Annotations', name+'.xml'))
        for obj in tree.findall("object"):
            if obj.find("name").text in ["D43", "Repair", "D01", "D11"]:
                box_color = (255, 0, 255)
            else:
                box_color = (0, 255, 255)
            bbox = obj.find("bndbox")
            box = [
                int(float(bbox.find("xmin").text)),
                int(float(bbox.find("ymin").text)),
                int(float(bbox.find("xmax").text)),
                int(float(bbox.find("ymax").text)),
                obj.find("name").text]
            print(box)
            cv2.rectangle(img, (box[0] + random.randint(-rand_range, rand_range), box[1]+ random.randint(-rand_range, rand_range)), (box[2]+ random.randint(-rand_range, rand_range), box[3]+ random.randint(-rand_range, rand_range)), color=box_color, thickness=2)
            cv2.putText(img, obj.find("name").text,
                        (box[0]-5, box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=2)
        cv2.imwrite(name+f'{rand_range}.jpg', img)


def tnse(features):
    # PCA?
    ts = TSNE(n_components=2, init='pca', random_state=42)
    features_tsne = ts.fit_transform(features)
    plt.scatter(features_tsne[:, 0], features_tsne[:, 1])
    plt.title('t-SNE Visualization of Features')
    plt.savefig('tsne.svg', format='svg', bbox_inches='tight')


def generate_feat_from_pth(root_path, location='RPN'):
    # program
    # visualize:generate_feat_from_pth -> default_predictor:get_rpn_feature -> rcnn:get_rpn_feature

    # data
    # visualize:config of data -> defaults:iter dataloader -> rcnn:calc each datq
    # 1. model config
    cfg = get_cfg()
    cfg.merge_from_file(root_path + "config.yaml")
    cfg.MODEL.WEIGHTS = root_path + "xxx"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 设置阈值，用于过滤低置信度的预测

    # 2. create predictor
    predictor = DefaultPredictor(cfg)
    if location == "RPN":
        return predictor.get_rpn_feature()


def prediction_draw(names):

    root = '../datasets/RDD'

    for name in names:
        img = cv2.imread(os.path.join(root, 'JPEGImages', name+'.jpg'))
        rand_range = 10
        objects = [[1, 353, 635, 583, 'D00']]
        for box in objects:
            if box[4] in ["D43", "Repair", "D01", "D11"]:
                box_color = (255, 0, 255)
            else:
                box_color = (0, 255, 255)

            cv2.rectangle(img, (box[0] + random.randint(-rand_range, rand_range), box[1]+ random.randint(-rand_range, rand_range)), (box[2]+ random.randint(-rand_range, rand_range), box[3]+ random.randint(-rand_range, rand_range)), color=box_color, thickness=2)
            cv2.putText(img, box[4],
                        (box[0]-5, box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=2)
        cv2.imwrite(name+f'{rand_range}.jpg', img)


if __name__ == "__main__":

    names = ['United_States_003258']


    # dynamic_path = "checkpoints/dynamic/1/3shot_seed6/"
    # upsample_path = "checkpoints/upsample/1/3shot_seed6/"
    # spcb_path = "checkpoints/rdd1/1/3shot_seed6/"
    #
    # tnse(dynamic_path)
    for i in [0, 5]:
        prediction_res(names, i)
    prediction_draw(names)
