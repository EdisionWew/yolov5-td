#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
"""
@author: WEW
@contact: wangerwei@tju.edu.cn
"""


import os
from pathlib import Path
from pascal_voc_writer import Writer



def check_dir(dir):
    if os.path.exists(dir):
        return dir
    else:
        os.makedirs(dir)
    return dir



def coarse_class(root, names, boxes, path):
    import cv2
    img = cv2.imread(path)
    path = Path(path)
    h, w, _ = img.shape

    if boxes.shape[0]==0:
        src_dir =  check_dir(os.path.join(root, "empty"))
    else:
        src_dir = check_dir(os.path.join(root, names[int(boxes[0][-1].item())].split('-')[0]))

    cp_jpg_str = "cp " + str(path) + ' ' + src_dir.replace(' ', '\ ')
    os.system(cp_jpg_str)

    writer = Writer(path.name, w, h)

    for box in boxes:
        writer.addObject(names[int(box[-1].item())], box[0].item(), box[1].item(), box[2].item(), box[3].item())

    if boxes.shape[0] !=0:
        writer.save(os.path.join(src_dir, path.name[:-3]+'xml'))
    else:
        pass




