import os
import model
import tensorflow as tf
import data_utils
config = {
    'n_epochs' : 20,
    'kernel_sizes' : [3, 4, 5],
    'dropout_rate' : 0.5,
    'val_split' : 0.1,
    'edim' : 300,
    'n_words' : None,
    'std_dev' : 0.1,
    'sentence_len' : 54,
    'n_filters'  : 100,
    'batch_size' : 50,
    'paths' : ['data/rt-polarity.pos', 'data/rt-polarity.neg']
}

data, labels, w2idx = data_utils.get_data(config['paths'])

config['n_words'] = len(w2idx)+1

with tf.Session() as sess:
    net = model.CNN(config, sess)
    net.train(data, labels)