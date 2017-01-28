import tensorflow as tf
import numpy as np
from math import ceil
import data_utils
import sys


class CNN(object):
    
    
    def __init__(self, config, sess):
        self.n_epochs = config['n_epochs']
        self.kernel_sizes = config['kernel_sizes']
        self.n_kernels = config['n_kernels']
        self.dropout_rate = config['dropout_rate']
        self.val_split = config['val_split']
        self.edim = config['edim']
        self.n_words = config['n_words']
        self.std_dev = config['std_dev']
        self.input_len = config['sentence_len']
        self.batch_size = config['batch_size']

        self.inp = tf.placeholder(shape=[None, self.input_len], dtype='int32')
        self.labels = tf.placeholder(shape=[None,], dtype='int32')
        self.loss = None
        self.session = sess
        
    def build_model(self):
        word_embedding = tf.Variable(tf.random_normal([self.n_words, self.edim], stddev=self.std_dev))
        x = tf.nn.embedding_lookup(word_embedding, self.inp)
        x_conv = tf.expand_dims(x, -1)
        #Filters
        F1 = tf.Variable(tf.random_normal([self.kernel_sizes[0], self.edim ,1, self.n_kernels] ,stddev=self.std_dev),dtype='float32')
        F2 = tf.Variable(tf.random_normal([self.kernel_sizes[1], self.edim, 1, self.n_kernels] ,stddev=self.std_dev),dtype='float32')
        F3 = tf.Variable(tf.random_normal([self.kernel_sizes[2], self.edim, 1, self.n_kernels] ,stddev=self.std_dev),dtype='float32')
        #Weight for final layer
        W = tf.Variable(tf.random_normal([3*self.n_kernels, 2], stddev=self.std_dev),dtype='float32')
        b = tf.Variable(tf.random_normal([1,2] ,stddev=self.std_dev),dtype='float32')
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
        zd = tf.nn.dropout(z, self.dropout_rate)
        print zd.get_shape()
        # Fully connected layer
        y = tf.add(tf.matmul(zd,W), b)
        print y.get_shape()
        print self.labels.get_shape()
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(y, self.labels)
        self.loss = tf.reduce_mean(losses)
        self.optim = tf.train.AdamOptimizer()
        self.train_op = self.optim.minimize(self.loss)

    def train(self, data, labels):
        self.build_model()
        n_batches = int(ceil(data.shape[0]/self.batch_size))
    	tf.initialize_all_variables().run()
        for epoch in range(1,self.n_epochs+1):
            t_data, t_labels, v_data, v_labels = data_utils.generate_split(data, labels, self.val_split)
            train_cost = 0
            for batch in range(1,n_batches+1):
                X, y = data_utils.generate_batch(data, labels, self.batch_size)
                f_dict = {
                    self.inp : X,
                    self.labels : y,
                }    

                _, cost = self.session.run([self.train_op, self.loss], feed_dict=f_dict)
                train_cost += cost
                sys.stdout.write('Cost  :   %f - Batch %d of %d     \r' %(cost ,batch ,n_batches))
                sys.stdout.flush()

            print
            print "Epoch train cost", train_cost/n_batches
            print

            self.test(v_data, v_labels)
    
    def test(self,data,labels):
        n_batches = int(ceil(data.shape[0]/self.batch_size))
        test_cost = 0
        for batch in range(1,n_batches+1):
            X, y = data_utils.generate_batch(data, labels, self.batch_size)
            f_dict = {
                self.inp : X,
                self.labels : y,
            }    
            cost = self.session.run([self.loss], feed_dict=f_dict)
            test_cost += cost[0]
            sys.stdout.write('Cost  :   %f - Batch %d of %d     \r' %(cost[0] ,batch ,n_batches))
            sys.stdout.flush()
        print
        print "Test cost", test_cost/n_batches
        print