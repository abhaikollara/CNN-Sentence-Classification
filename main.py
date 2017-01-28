import os
import model
import tensorflow as tf
import data_utils
import configuration

data, labels, w2idx = data_utils.get_data(configuration.config['paths'])

configuration.config['n_words'] = len(w2idx)+1

with tf.Session() as sess:
    net = model.CNN(configuration.config, sess)
    net.train(data, labels)