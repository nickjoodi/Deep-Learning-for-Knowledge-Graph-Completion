from __future__ import print_function
import pickle as pickle
import numpy as np
import re

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

clean_dic = {}
with open(r"word_vectors.pkl", "rb") as input_file:
    e = pickle.load(input_file)
    for k,v in e.items():
        new_k = ''.join(w for w in re.split(r"\W", k) if w)
        clean_dic[new_k.lower()] = v

save_object(clean_dic,'word_vectors_clean.pkl')


