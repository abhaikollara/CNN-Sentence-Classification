import tensorflow as tf
import numpy as np

config = {
    'n_epochs' : 5,
    'kernel_sizes' : [3, 4, 5],
    'dropout_rate' : 0.5,
    'val_split' : 0.1,
    'edim' : 300,
    'n_words' : 18763,
    'std_dev' : 0.05,
    'sentence_len' : 54,
    'n_kernels'  : 100,
}

class CNN(object):
    
    
    def __init__(self, config):
        self.n_epochs = config['n_epochs']
        self.kernel_sizes = config['kernel_sizes']
        self.n_kernels = config['n_kernels']
        self.dropout_rate = config['dropout_rate']
        self.val_split = config['val_split']
        self.edim = config['edim']
        self.n_words = config['n_words']
        self.std_dev = config['std_dev']
        self.input_len = config['sentence_len']
        self.inp = tf.placeholder(shape=[None, self.input_len], dtype='int32')
        self.labels = tf.placeholder(shape=[None,], dtype='int32')
        self.loss = None
#         self.session = sess
        
    def build_model(self):
        word_embedding = tf.Variable(tf.random_normal([self.n_words, self.edim], stddev=self.std_dev))
        x = tf.nn.embedding_lookup(word_embedding, self.inp)
        x_conv = tf.expand_dims(x, -1)
        #Filters
        F1 = tf.Variable(tf.random_normal([self.kernel_sizes[0], self.edim ,1, self.n_kernels]))
        F2 = tf.Variable(tf.random_normal([self.kernel_sizes[1], self.edim, 1, self.n_kernels]))
        F3 = tf.Variable(tf.random_normal([self.kernel_sizes[2], self.edim, 1, self.n_kernels]))
        #Weight for final layer
        W = tf.Variable(tf.random_normal([3*self.n_kernels, 2]))
        b = tf.Variable(tf.random_normal([1,2]))
        #Convolutions
        C1 = tf.nn.relu(tf.nn.conv2d(x_conv, F1, [1,1,1,1], padding='VALID'))
        C2 = tf.nn.relu(tf.nn.conv2d(x_conv, F2, [1,1,1,1], padding='VALID'))
        C3 = tf.nn.relu(tf.nn.conv2d(x_conv, F2, [1,1,1,1], padding='VALID'))
        #Max pooling
        maxC1 = tf.nn.max_pool(C1, [1,C1.get_shape()[1],1,1] , [1,1,1,1], padding='VALID')
        maxC1 = tf.squeeze(maxC1, [1,2])
        maxC2 = tf.nn.max_pool(C2, [1,C2.get_shape()[1],1,1] , [1,1,1,1], padding='VALID')
        maxC2 = tf.squeeze(maxC2, [1,2])
        maxC3 = tf.nn.max_pool(C3, [1,C3.get_shape()[1],1,1] , [1,1,1,1], padding='VALID')
        maxC3 = tf.squeeze(maxC3, [1,2])
        z = tf.concat(1, [maxC1, maxC2, maxC3])
        zd = tf.dropout(z, self.dropout_rate)
        # Fully connected layer
        y = tf.add(tf.matmul(z,W), b)
        
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(y, self.labels)
        self.loss = tf.reduce_mean(losses)
    def train(self, data):
    	pass