from detectron2.structures import Instances
from detectron2.utils.visualizer import Visualizer
from detectron2.data.detection_utils import read_image
from xml.etree import ElementTree as ET
import os
import cv2

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
