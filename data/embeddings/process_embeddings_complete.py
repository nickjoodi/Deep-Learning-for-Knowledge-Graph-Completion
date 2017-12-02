from __future__ import print_function
import pickle as pickle
import numpy as np
import re

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

clean_dic = {}
#f1  = open(r"word_vectors_clean_complete.pkl", "rb")
f2  = open(r"../processed/entities_to_strings_clean_map_large.pkl", "rb")
#wv = pickle.load(f1)
wv = {}
entities = pickle.load(f2)
count=0
unknown = set()
#print(len(wv))
for k,v in entities.items():
	for s in v:
		#if s not in wv:
		unknown.add(s)
print(len(unknown))

if len(unknown) > 1:
	for e in unknown:
		#print(e)
		r = .0001
		word_vector = np.random.random(( 300)) * 2 * r - r
		wv[e] = word_vector

print(len(wv))


save_object(wv,'random_init_word_vectors_clean_complete_large.pkl')