from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import argparse
import sys
import tempfile
import numpy as np
import pandas as pd
import pickle as pickle
import re
import time
from threading import Thread
import datetime
import random

def load_data(fname):
    f = open(fname, encoding='utf8')
    data = f.read().strip().split('\n')
    f.close()
    return np.array(data)


positives = load_data("positiveTriplets_unique_large.txt")
negatives = load_data("negativeTriplets_more_large.txt")
np.random.shuffle(positives)
np.random.shuffle(negatives)

pos_list = np.array_split(positives, 6)
pos_test_dev = pos_list.pop(0)


fout = "training_large.txt"
fo = open(fout, "w", encoding='utf8')
for a in pos_list:
    for l in np.nditer(a):
        fo.write(np.array_str(l))
        fo.write('\n')

pos_test_dev_list = np.array_split(pos_test_dev, 6)
pos_dev = pos_test_dev_list.pop(0)
fout2 = "test_large.txt"
fo2 = open(fout2, "w", encoding='utf8')
for a in pos_test_dev_list:
    for l in np.nditer(a):
        fo2.write(np.array_str(l))
        fo2.write('\n')
        fo2.write(np.array_str(negatives[0]))
        fo2.write('\n')
        negatives = np.delete(negatives, 0, 0)

fout3 = "../processed/dev_large.txt"
fo3 = open(fout3, "w", encoding='utf8')
for a in np.nditer(pos_dev):
    fo3.write(np.array_str(a))
    fo3.write('\n')
    fo3.write(np.array_str(negatives[0]))
    fo3.write('\n')
    negatives = np.delete(negatives, 0, 0)



