# coding=utf-8
import numpy as np
import operator

def classify0(inX, data_set, labels, k):
    data_set_size = data_set.shape[0]
    diff_matrix = np.tile(inX, (data_set_size, 1)) - data_set
    sq_diff_matrix = diff_matrix**2
    sq_distance = sq_diff_matrix.sum(axis=1)
    distance = sq_distance**0.5
    sorted_distance_indicies = distance.argsort()
    class_count = {}
    for i in range(k):
        voted_label = labels[sorted_distance_indicies[i]]
        class_count[voted_label] = class_count.get(voted_label, 0) +1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]
