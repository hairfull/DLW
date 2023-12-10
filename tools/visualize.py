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

def prediction_res():

    names = ['United_States_004761']

    root = '../datasets/RDD'

    for name in names:
        img = cv2.imread(os.path.join(root, 'JPEGImages', name+'.jpg'))

        tree = ET.parse(os.path.join(root, 'Annotations', name+'.xml'))
        objects = []
        box_color = (255, 0, 255)
        for obj in tree.findall("object"):
            bbox = obj.find("bndbox")
            box = [
                int(float(bbox.find("xmin").text)),
                int(float(bbox.find("ymin").text)),
                int(float(bbox.find("xmax").text)),
                int(float(bbox.find("ymax").text)),
                obj.find("name").text]
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=box_color, thickness=2)
            cv2.putText(img, obj.find("name").text,
                        (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1)
        cv2.imwrite(name+'.jpg', img)


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


if __name__ == "__main__":

    dynamic_path = "checkpoints/dynamic/1/3shot_seed6/"
    upsample_path = "checkpoints/upsample/1/3shot_seed6/"
    spcb_path = "checkpoints/rdd1/1/3shot_seed6/"

    tnse(dynamic_path)