import os
import random
import shutil
import json
from collections import defaultdict
import xml.etree.ElementTree as ET
from PIL import Image


def traverse_and_copy_json_file(src, dest):
    label_cnt = defaultdict(int)

    for root, dirs, files in os.walk(src):
        for file in files:
            if file.endswith('.json'):
                src_file_path = os.path.join(root, file)
                shutil.copy(src_file_path, dest)

                with open(src_file_path, 'r') as f:
                    data = json.load(f)
                    for obj in data.get('objects', []):
                        label = obj.get('label')
                        label_cnt[label] += 1

    return label_cnt


def write2file(file_list, filename):
    with open(filename, 'a') as f:
        for file in file_list:
            f.write(f'{file}\n')


def split_data_by_label(src, des, labels):
    label2files = defaultdict(list)

    for root, dirs, files in os.walk(src):
        for file in files:
            if file.endswith('.json'):
                src_file_path = os.path.join(root, file)
                img_file_path = os.path.join(root.replace('gtCoarse', 'leftImg8bit'), file.replace('_gtCoarse_polygons.json', '_leftImg8bit.png'))

                with open(src_file_path, 'r') as f:
                    data = json.load(f)
                    for obj in data.get('objects', []):
                        label = obj.get('label')
                        if label in labels:
                            label2files[label].append(file.replace('_gtCoarse_polygons.json', ''))
                            img = Image.open(img_file_path)
                            img.convert('RGB').save(os.path.join(des, file.replace('_gtCoarse_polygons.json', '.jpg')))
                            break

    # trainval = []
    # test = []
    #
    # for k, v in label2files.items():
    #     random.shuffle(label2files[k])
    #     trainval.extend(label2files[k][:150])
    #     test.extend(label2files[k][150:])
    #
    # write2file(trainval, '/home/wxq/od/DeFRCN/datasets/RDD/ImageSets/Main/trainval.txt')
    # write2file(test, '/home/wxq/od/DeFRCN/datasets/RDD/ImageSets/Main/test.txt')

    return


def transform_json_to_xml(rootdir):
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith('.json'):
                convert_annotation(os.path.join(root, file))


def position(pos):
    # 该函数用来找出xmin,ymin,xmax,ymax即bbox包围框
    x = []
    y = []
    nums = len(pos)
    for i in range(nums):
        x.append(pos[i][0])
        y.append(pos[i][1])
    x_max = max(x)
    x_min = min(x)
    y_max = max(y)
    y_min = min(y)
    return [x_min, y_min, x_max, y_max]


def convert_annotation(json_file_name):
    name = json_file_name.replace('_gtCoarse_polygons.json', '')

    tree = ET.parse('template.xml')
    root = tree.getroot()

    filename = root.find('filename')
    filename.text = name.replace('/home/wxq/od/DeFRCN/datasets/LAF/Annotations/', '')

    with open(json_file_name, 'r') as load_f: # 导入json标签的地址
        load_dict = json.load(load_f)

        # keys=tuple(load_dict.keys())
        w = load_dict['imgWidth']  # 原图的宽，用于归一化
        h = load_dict['imgHeight']

        size = root.find('size')
        width = size.find('width')
        height = size.find('height')

        width.text = str(w)
        height.text = str(h)

        objects = load_dict['objects']
        nums = len(objects)

        object_template = root.find('object')
        root.remove(object_template)

        for i in range(0, nums):
            labels = objects[i]['label']
            # print(i)
            if labels in ['02', '04', '05', '20', '26', '36', '16', '28', '32']:
                # print(labels)
                pos = objects[i]['polygon']
                b = position(pos)

                new_obj = ET.Element('object')
                label_name = ET.SubElement(new_obj, 'name')
                label_name.text = labels

                pose = ET.SubElement(new_obj, 'pose')
                pose.text = 'Unspecified'

                truncated = ET.SubElement(new_obj, 'truncated')
                truncated.text = '0'

                difficult = ET.SubElement(new_obj, 'difficult')
                difficult.text = '0'

                bndbox = ET.SubElement(new_obj, 'bndbox')

                xmin = ET.SubElement(bndbox, 'xmin')
                xmin.text = str(b[0])

                ymin = ET.SubElement(bndbox, 'ymin')
                ymin.text = str(b[1])

                xmax = ET.SubElement(bndbox, 'xmax')
                xmax.text = str(b[2])

                ymax = ET.SubElement(bndbox, 'ymax')
                ymax.text = str(b[3])

                root.append(new_obj)

        tree.write(name + '.xml', encoding='utf-8')


def rename_file(src):
    for filename in os.listdir(src):
        if '_leftImg8bit' in filename:
            newname = filename.replace('_leftImg8bit', '')
            os.rename(os.path.join(src, filename), os.path.join(src, newname))


def update_txt_file(dic):
    for filename in os.listdir(dic):
        with open(os.path.join(dic, filename), 'r') as file:
            lines = file.readlines()
        with open(os.path.join(dic, filename), 'w') as file:
            for line in lines:
                newline = line.replace('.json', '')
                file.write(newline)


src = '/home/wxq/od/laf/gtCoarse'
des = '/home/wxq/od/DeFRCN/datasets/RDD/JPEGImages/'
#
# label_cnt = traverse_and_copy_json_file(src, '/home/wxq/od/DeFRCN/datasets/LAF/Annotations/')
# for label, cnt in label_cnt.items():
# #     print(f'label:{label} cnt:{cnt}')
#
split_data_by_label(src, des, ['05', '20', '26', '36'])
# transform_json_to_xml('/home/wxq/od/DeFRCN/datasets/LAF/Annotations/')

# rename_file('Annotations')
# rename_file('JPEGImages')
# update_txt_file('ImageSets/Main')