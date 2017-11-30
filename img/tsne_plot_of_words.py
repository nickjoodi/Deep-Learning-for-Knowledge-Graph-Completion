from __future__ import print_function
import pickle as pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from gensim.models import KeyedVectors
import numpy as np

with open(r"../data/embeddings/random_init_word_vectors_clean_complete.pkl", "rb") as input_file:
    e = pickle.load(input_file)

    print(e['the'])

def plot_with_labels(low_dim_embs, labels, filename='tsne_words_random.png'):
    plt.figure(figsize=(18, 18)) 
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
    plt.savefig(filename)

words = []
embedding = np.array([])
limit=500
vector_dim = 300
i=0
for k, v in e.items():
    if i == limit: break
    words.append(k)
    embedding = np.append(embedding, v)
    i += 1

# Reshaping embedding
embedding = embedding.reshape(limit, vector_dim)

tsne = TSNE(perplexity=30.0, n_components=2, init='pca', n_iter=5000)

low_dim_embedding = tsne.fit_transform(embedding)

plot_with_labels(low_dim_embedding, words)