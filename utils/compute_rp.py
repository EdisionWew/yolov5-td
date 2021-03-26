#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
"""
@author: WEW
@contact: wangerwei@tju.edu.cn
"""
import pandas as pd
import numpy as np


def per_brand_recall_precison(target_label, pre_label, brand_names, csv_file = "./"):
    sorted(brand_names)
    target_label = np.array(target_label)
    pre_label = np.array(pre_label)
    print("| {:<20} | {:<8} | {:<8} | {:<8} | {:<10} | {:<10} | {:<10} |".format("brand_name", "recall", 'precision', 'f1-score','predictions', 'pos-targets', 'targets'))
    print('|' + '----------------------|'+'----------|' * 3+'-----------|'*3)
    pycharts_data = {}
    for i, brand_name in enumerate(brand_names):
        per_target_label = target_label==i
        per_pre_label = pre_label==i

        #compute presion
        if sum(per_pre_label) != 0.0:
            precision = sum(per_target_label & per_pre_label) / sum(per_pre_label)
        else:
            precision = 0.0
        # compute recall
        if sum(per_target_label) != 0.0:
            recall = sum(per_target_label & per_pre_label) / sum(per_target_label)
        else:
            recall = 0.0
        if precision + recall == 0:
            f1_score = 0
        else:
            f1_score = 2*(precision*recall)/(precision+recall)
        print("| {:<20} | {:<8} | {:<8} | {:<8} | {:<10} | {:<10} | {:<10} |".format(brand_name, round(recall, 3), round(precision, 3), round(f1_score, 3), sum(per_pre_label), sum(per_target_label & per_pre_label), sum(per_target_label)))

        pycharts_data.update({brand_name: [round(recall, 3), round(precision, 3)]})
    print(pycharts_data)
    pd.DataFrame(pycharts_data).to_csv(csv_file)
    return


def compute_recall_precision(target_label, pre_label):
    #compute precison
    target_label = np.array(target_label)
    pre_label = np.array(pre_label)
    print("len(target_label): ", len(target_label))

    # pre is true
    print("white samples: ", sum(target_label==-1))
    print("black samples: ", sum(target_label!=-1))
    tpfp = pre_label != -1
    alls = target_label == pre_label

    A = sum(tpfp & alls)
    precisions =  A / sum(tpfp)

    #compute recall
    recall = A / sum(target_label!=-1)
    if recall + precisions == 0:
        f1_score = 0.0
    else:
        f1_score = 2*(precisions*recall)/(recall+precisions)
    return recall, precisions, f1_score





