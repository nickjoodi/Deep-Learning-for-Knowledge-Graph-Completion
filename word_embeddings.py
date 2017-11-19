from __future__ import print_function
from gensim.models import KeyedVectors
import pickle as pickle


# downloaded pretrained word embeddings from here:
# https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md

en_model = KeyedVectors.load_word2vec_format('~/Downloads/wiki.en.vec')

words = []
for word in en_model.vocab:
    words.append(word)

with open("unique.txt","r", encoding='utf8') as f:
    content = f.readlines()

content = [x.strip() for x in content] 

print("Number of Tokens: {}".format(len(words)))

print("Vector components of a word: {}".format(
    en_model[words[0]]
))

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

fout='word_vectors.txt'
fo = open(fout, "w", encoding='utf-8')
word_vecs = {}
for k in content:
    print(k.encode('utf-8'))
    if k in en_model.wv.vocab:
        word_vecs[k] =en_model[k]
    elif k.lower() in en_model.wv.vocab:
        word_vecs[k] =en_model[k.lower()]
    elif k.upper() in en_model.wv.vocab:
        word_vecs[k] =en_model[k.upper()]

save_object(word_vecs,'word_vectors.pkl')




