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
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

preds = {'P26':'spouse',
    'P40':'child',
    'P22':'father',
    'P25':'mother',
    'P3373':'sibling'}
    
# Implementation of the Neural tensor network in tensorflow 1.3
# Used the following works to develop the architecture:
# Original code provided by the authors of the NTN: http://www-nlp.stanford.edu/~socherr/codeDeepDB.zip
# A python implementation using numpy and Scipy: https://github.com/siddharth-agrawal/Neural-Tensor-Network
# An implementation in Tensorflow 0.5.0: https://github.com/dddoss/tensorflow-socher-ntn
# The original article: https://nlp.stanford.edu/~socherr/SocherChenManningNg_NIPS2013.pdf

def load_word_vecs():
    f = open(r"../data/embeddings/large_set/random_init_word_vectors_clean_complete_large.pkl", "rb") 
    w = pickle.load(f)
    f.close
    return w

def load_entities_to_words():
    f = open(r"../data/processed/large_set/entities_to_strings_clean_map_large.pkl", "rb") 
    w = pickle.load(f)
    f.close
    return w


def load_data(fname):
    f = open(fname, encoding='utf8')
    training_data = [l.split() for l in f.read().strip().split('\n')]
    return np.array(training_data)

def create_dic(file_name):
    file_object = open(file_name, 'r', encoding='utf8')
    data = file_object.read().splitlines()
    dictionary = {}
    index = 0
    for entity in data:
        dictionary[entity] = index
        index += 1
    return dictionary


def create_dic_from_word_vec(word_vecs):
    words_dic ={}
    index = 0
    for k,v in word_vecs.items():
        words_dic[k] = index
        index += 1
    return words_dic


def index_data(training_data,entity_dic,pred_dic):
    indexed_data = [(entity_dic[training_data[i][0]], pred_dic[training_data[i][1]], entity_dic[training_data[i][2]]) for i in range(len(training_data))]
    return indexed_data

def index_testing_data(training_data,entity_dic,pred_dic):
    indexed_data = [(entity_dic[training_data[i][0]], pred_dic[training_data[i][1]], entity_dic[training_data[i][2]], float(training_data[i][3])) for i in range(len(training_data))]
    return indexed_data



def create_indexed_embeds(word_vecs, entities_to_words, words_dic, entity_dic):
    indexed_word_vecs = [[0.0 for j in range(300)] for i in range(len(words_dic))]
    for k,v in words_dic.items():
        i = 0
        for e in word_vecs[k]:
            indexed_word_vecs[v][i] = e
            i+=1

    indexed_entities =[None]*len(entity_dic)
    for k,v in entity_dic.items():
        indexed_entities[v] = []
        for s in entities_to_words[k]:
            indexed_entities[v].append(words_dic[s])
    return (indexed_word_vecs,indexed_entities)

training_data = load_data("../data/processed/large_set/training_large.txt")
testing_data = load_data("../data/processed/large_set/test_large.txt")
dev_data = load_data("../data/processed/large_set/dev_large.txt")
pred_dic = create_dic("../data/processed/predicates.txt")
entity_dic = create_dic("../data/processed/large_set/entityIds_large.txt")
word_vecs = load_word_vecs()
entities_to_words = load_entities_to_words()
words_dic = create_dic_from_word_vec(word_vecs)
indexed_word_vecs,indexed_entities  =create_indexed_embeds(word_vecs, entities_to_words, words_dic, entity_dic)
indexed_data = index_data(training_data,entity_dic,pred_dic)
indexed_dev_data = index_testing_data(dev_data,entity_dic,pred_dic)
indexed_test_data = index_testing_data(testing_data,entity_dic,pred_dic)
indexed_train_data = index_testing_data(training_data,entity_dic,pred_dic)

num_iters = 200
batch_size=10000
corrupt_size = 10
slice_size = 3
num_entities = len(entity_dic)
num_preds = len(pred_dic)
embedding_size = 300
reg = 0.0001

def calculate_loss(predictions):
    max_with_margin_sum =tf.reduce_sum(tf.maximum(tf.subtract(predictions[1, :], predictions[0, :]) + 1, 0))
    l2 = tf.sqrt(sum([tf.reduce_sum(tf.square(var)) for var in tf.trainable_variables()]))
    return max_with_margin_sum + (reg * l2)

def fill_feed_dict(batches, batch_placeholders):
    feed_dict = {}
    for i in range(len(batch_placeholders)):
        feed_dict[batch_placeholders[i]] = batches[i]
    feed_dict[flip_placeholder] = bool(random.getrandbits(1))
    return feed_dict

def get_batch( data):
    random_indices = random.sample(range(len(data)), batch_size)
    batch = [(data[i][0], data[i][1], data[i][2], random.randint(0, num_entities-1))\
        for i in random_indices for j in range(corrupt_size)]
    return batch

def distribute_batch(data_batch):
    batches = [[] for i in range(num_preds)]
    for e1,r,e2,e3 in data_batch:
        batches[r].append((e1,e2,e3))
    return batches


def distribute_testing_data(data):
    batches = [[] for i in range(num_preds)]
    labels = [[] for i in range(num_preds)]
    for e1,r,e2,label in data:
        batches[r].append((e1,e2))
        labels[r].append(label)
    return batches,labels

def fill_dev_feed_dict(dev_batch, test_batch_placeholders):
    feed_dict = {}
    for i in range(len(batch_placeholders)):
        feed_dict[test_batch_placeholders[i]] = dev_batch[i]
    return feed_dict

def compute_ideal_thresholds(min_score, max_score, predictions_list, dev_labels,dev_data):
    best_thresholds = np.zeros([num_preds, 1])
    best_accuracies = np.zeros([num_preds, 1])
    for i in range(num_preds):
        best_thresholds[i][0] = min_score
        best_accuracies[i][0] =-1

    score = min_score
    increment = 0.01
    while(score <= max_score):
        for i in range(num_preds):
            predictions = (predictions_list[i] > score) * 2 - 1
            accuracy = np.mean((predictions == dev_labels[i]))
            if(accuracy > best_accuracies[i, 0]):
                best_accuracies[i, 0] = accuracy
                best_thresholds[i, 0] = score
        score += increment
    return best_thresholds

def classify(predictions_list,best_thresholds):
    classifications = [[] for i in range(num_preds)]
    for i in range(num_preds):
        for test_score in predictions_list[i]:
            if(test_score > best_thresholds[i, 0]):
                classifications[i].append(1)
            else:
                classifications[i].append(-1)

    return classifications

def determine_predictions_for_training(ent_to_word_indices,flip_placeholder,batch_placeholders, E_tensor, W_tensor, V_tensor,U_tensor,b_tensor):
    flip = tf.random_uniform([1], minval=0, maxval=1, dtype=tf.float32)
    entity_emb= tf.stack([tf.reduce_mean(tf.gather(E_tensor, ent_word_indice), 0) for ent_word_indice in ent_to_word_indices])
    predictions = list()
    for p in range(num_preds):
        e1, e2, e3 = tf.split( tf.cast(batch_placeholders[p], tf.int32),3,1)
        e1v = tf.transpose(tf.squeeze(tf.gather(entity_emb, e1)))
        e2v = tf.transpose(tf.squeeze(tf.gather(entity_emb, e2)))
        e3v = tf.transpose(tf.squeeze(tf.gather(entity_emb, e3)))
        e1v_pos = e1v
        e2v_pos = e2v
        e1v_neg,e2v_neg = tf.cond(flip_placeholder, lambda: (e1v,e3v), lambda: (e3v,e2v))
        num_pred_i = tf.expand_dims(tf.shape(e1v_pos)[1], 0)
        preactivation_pos = list()
        preactivation_neg = list()

        for slice in range(slice_size):
            preactivation_pos.append(tf.reduce_sum(e1v_pos*tf.matmul(W_tensor[p][:,:,slice], e2v_pos), 0))
            preactivation_neg.append(tf.reduce_sum(e1v_neg*tf.matmul( W_tensor[p][:,:,slice], e2v_neg), 0))

        preactivation_pos = tf.stack(preactivation_pos)
        preactivation_neg = tf.stack(preactivation_neg)
        preactivation_pos = preactivation_pos+tf.matmul( V_tensor[p],tf.concat( [e1v_pos, e2v_pos],0))+b_tensor[p]
        preactivation_neg = preactivation_neg+tf.matmul( V_tensor[p],tf.concat([e1v_neg, e2v_neg],0))+b_tensor[p]
        activation_pos = tf.tanh(preactivation_pos)
        activation_neg = tf.tanh(preactivation_neg)
        score_pos = tf.reshape(tf.matmul(U_tensor[p], activation_pos), num_pred_i)
        score_neg = tf.reshape(tf.matmul(U_tensor[p], activation_neg), num_pred_i)
        predictions.append(tf.stack([score_pos, score_neg]))
    predictions = tf.concat( predictions,1)
    return predictions



def predict(ent_to_word_indices, test_batch_placeholders,E_tensor, W_tensor, V_tensor,U_tensor,b_tensor):
    entity_emb= tf.stack([tf.reduce_mean(tf.gather(E_tensor, ent_word_indice), 0) for ent_word_indice in ent_to_word_indices])
    predictions_list = list()
    for p in range(num_preds):

        e1, e2 = tf.split( tf.cast(test_batch_placeholders[p], tf.int32),2,1)
        e1v = tf.transpose(tf.squeeze(tf.gather(entity_emb, e1)))
        e2v = tf.transpose(tf.squeeze(tf.gather(entity_emb, e2)))
        num_pred_i = tf.expand_dims(tf.shape(e1v)[1], 0)
        preactivation_pos = list()
        for slice in range(slice_size):
            preactivation_pos.append(tf.reduce_sum(e1v*tf.matmul(W_tensor[p][:,:,slice], e2v), 0))

        preactivation_pos = tf.stack(preactivation_pos)+tf.matmul( V_tensor[p],tf.concat( [e1v, e2v],0))+b_tensor[p]
        activation_pos = tf.tanh(preactivation_pos)
        score_pos = tf.reshape(tf.matmul(U_tensor[p], activation_pos), num_pred_i)
        predictions_list.append(tf.stack(score_pos))
    min_score = tf.reduce_min(tf.concat(predictions_list, 0))
    max_score = tf.reduce_max(tf.concat(predictions_list, 0))
    return min_score, max_score, predictions_list

g = tf.Graph()
with g.as_default():
    print('create graph...')
    ent_to_word_indices = [tf.constant(entity_i) for entity_i in indexed_entities]
    flip_placeholder = tf.placeholder(tf.bool)
    batch_placeholders = [tf.placeholder(tf.int32, shape=(None, 3)) for i in range(num_preds)]
    test_batch_placeholders = [tf.placeholder(tf.int32, shape=(None, 2)) for i in range(num_preds)]
    E_tensor = tf.Variable(indexed_word_vecs, name='E_tensor')
    W_tensor = [tf.Variable(tf.truncated_normal([embedding_size, embedding_size, slice_size])) for p in range(num_preds)]
    V_tensor = [tf.Variable(tf.zeros([slice_size,2*embedding_size])) for p in range(num_preds)]
    U_tensor = [tf.Variable(tf.zeros([1,slice_size])) for p in range(num_preds)]
    b_tensor = [tf.Variable(tf.zeros([slice_size,1])) for p in range(num_preds)]
    
    predictions_for_training = determine_predictions_for_training(ent_to_word_indices,flip_placeholder,batch_placeholders, E_tensor, W_tensor,  V_tensor,U_tensor,b_tensor)
    loss = calculate_loss(predictions_for_training)
    
    predictions = predict(ent_to_word_indices,test_batch_placeholders, E_tensor, W_tensor, V_tensor,U_tensor,b_tensor)
    
    train_step = tf.train.AdagradOptimizer(0.01).minimize(loss)
    with tf.Session(graph=g) as sess:
        
        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()
        iter_list = []
        loss_list = []
        print('Begin training...')
        for i in range(1, num_iters):
            print(str(datetime.datetime.now())+" - iteration "+str(i))
            data_batch = get_batch(indexed_data)
            pred_batches = distribute_batch(data_batch)
            feed_dict = fill_feed_dict(pred_batches, batch_placeholders)
            _, iter_loss = sess.run([train_step,loss],feed_dict=feed_dict)
            iter_list.append(i)
            loss_list.append(iter_loss)
            print('loss at current iteration = ' )
            print(iter_loss)
        print('Calculate thresholds for each predicate')
        dev_batch,dev_labels = distribute_testing_data( indexed_dev_data )
        feed_dev_dict = fill_dev_feed_dict(dev_batch, test_batch_placeholders)
        print('Determine ideal thresholds')
        min_score, max_score, predictions_list = sess.run(predictions, feed_dict=feed_dev_dict)
        best_thresholds = compute_ideal_thresholds(min_score, max_score, predictions_list,dev_labels,dev_batch)
        
        print('Test model')
        test_batch,test_labels = distribute_testing_data( indexed_test_data )
        feed_testing_dict = fill_dev_feed_dict(test_batch, test_batch_placeholders)
        print('Perform predictions')
        min_score, max_score, predictions_list = sess.run(predictions, feed_dict=feed_testing_dict)
        classifications = classify(predictions_list,best_thresholds)
        print(test_labels)
        for i in range(num_preds):
            accuracy = sum(1 for x,y in zip(test_labels[i],classifications[i]) if x == y) / len(test_labels[i])
            print('pred')
            print(i)
            print(accuracy)
        flattened_test_labels = np.concatenate(test_labels).ravel()
        flattened_predictions = np.concatenate(predictions_list).ravel()
        flattened_classifications = np.concatenate(classifications).ravel()

        print('plot ROC')
        fpr, tpr , _ = metrics.roc_curve(flattened_test_labels, flattened_predictions, pos_label=1)
        aucROC = metrics.auc(fpr, tpr)
        print('auc')
        print(aucROC)
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(fpr, tpr, lw=2, color='darkorange', label="AUC:{:.3f}".format(aucROC))
        ax1.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC({!s})".format('ntn'))
        plt.legend(loc="lower right")
        filename = 'img/{!s}_ROC_{!s}.pdf'.format('ntn', 'all')
        plt.savefig(filename)

        accuracy = sum(1 for x,y in zip(flattened_test_labels,flattened_classifications) if x == y) / len(flattened_test_labels)
        print('overall accuracy')
        print(accuracy)

        word_vectors = E_tensor.eval()
        embedding = np.array([])
        words = [None] * len(words_dic)
        for k,v in words_dic.items():
            words[v] = k
        word_vectors = word_vectors[:500,:]
        words = words[:500]

        def plot_with_labels(low_dim_embs, labels, filename='img/tsne_words_ntn.png'):
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

        embedding = np.array(word_vectors)
        limit=500
        vector_dim = 300

        embedding = embedding.reshape(limit, vector_dim)
        print('TSNE')
        tsne = TSNE(perplexity=30.0, n_components=2, init='pca', n_iter=5000)

        low_dim_embedding = tsne.fit_transform(embedding)
        print('Plot semantic space')
        plot_with_labels(low_dim_embedding, words)

        print('Plot loss per iteration')
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(iter_list, loss_list, lw=2, color='darkorange')
        plt.xlabel("Iteration #")
        plt.ylabel("Loss")
        plt.title("Loss per Iteration of Training")
        filename = 'img/_loss_.pdf'
        plt.savefig(filename)



