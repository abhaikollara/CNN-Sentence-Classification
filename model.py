import tensorflow as tf
import numpy as np
from math import ceil
import data_utils
import sys


class CNN(object):
    
    
    def __init__(self, config, sess):
        self.n_epochs = config['n_epochs']
        self.kernel_sizes = config['kernel_sizes']
        self.n_filters = config['n_filters']
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
        F1 = tf.Variable(tf.random_normal([self.kernel_sizes[0], self.edim ,1, self.n_filters] ,stddev=self.std_dev),dtype='float32')
        F2 = tf.Variable(tf.random_normal([self.kernel_sizes[1], self.edim, 1, self.n_filters] ,stddev=self.std_dev),dtype='float32')
        F3 = tf.Variable(tf.random_normal([self.kernel_sizes[2], self.edim, 1, self.n_filters] ,stddev=self.std_dev),dtype='float32')
        FB1 = tf.Variable(tf.constant(0.1, shape=[self.n_filters]))
        FB2 = tf.Variable(tf.constant(0.1, shape=[self.n_filters]))
        FB3 = tf.Variable(tf.constant(0.1, shape=[self.n_filters]))
        #Weight for final layer
        W = tf.Variable(tf.random_normal([3*self.n_filters, 2], stddev=self.std_dev),dtype='float32')
        b = tf.Variable(tf.random_normal([1,2] ,stddev=self.std_dev),dtype='float32')
        #Convolutions
        C1 = tf.add(tf.nn.conv2d(x_conv, F1, [1,1,1,1], padding='VALID'), FB1)
        C2 = tf.add(tf.nn.conv2d(x_conv, F2, [1,1,1,1], padding='VALID'), FB2)
        C3 = tf.add(tf.nn.conv2d(x_conv, F3, [1,1,1,1], padding='VALID'), FB3)

        C1 = tf.nn.relu(C1)
        C2 = tf.nn.relu(C2)
        C3 = tf.nn.relu(C3)

        #Max pooling
        maxC1 = tf.nn.max_pool(C1, [1,C1.get_shape()[1],1,1] , [1,1,1,1], padding='VALID')
        maxC1 = tf.squeeze(maxC1, [1,2])
        maxC2 = tf.nn.max_pool(C2, [1,C2.get_shape()[1],1,1] , [1,1,1,1], padding='VALID')
        maxC2 = tf.squeeze(maxC2, [1,2])
        maxC3 = tf.nn.max_pool(C3, [1,C3.get_shape()[1],1,1] , [1,1,1,1], padding='VALID')
        maxC3 = tf.squeeze(maxC3, [1,2])
        z = tf.concat(1, [maxC1, maxC2, maxC3])
        zd = tf.nn.dropout(z, self.dropout_rate)
        # Fully connected layer
        self.y = tf.add(tf.matmul(zd,W), b)
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(self.y, self.labels)
        self.loss = tf.reduce_mean(losses)
        self.optim = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op = self.optim.minimize(self.loss)

    def train(self, data, labels):
        self.build_model()
        n_batches = int(ceil(data.shape[0]/self.batch_size))
    	tf.initialize_all_variables().run()
        t_data, t_labels, v_data, v_labels = data_utils.generate_split(data, labels, self.val_split)
        for epoch in range(1,self.n_epochs+1):
            train_cost = 0
            for batch in range(1,n_batches+1):
                X, y = data_utils.generate_batch(t_data, t_labels, self.batch_size)
                f_dict = {
                    self.inp : X,
                    self.labels : y,
                }    

                _, cost = self.session.run([self.train_op, self.loss], feed_dict=f_dict)
                train_cost += cost
                sys.stdout.write('Epoch %d Cost  :   %f - Batch %d of %d     \r' %(epoch, cost ,batch ,n_batches))
                sys.stdout.flush()

            print

            self.test(v_data, v_labels)
    
    def test(self,data,labels):
        n_batches = int(ceil(data.shape[0]/self.batch_size))
        test_cost = 0
        preds = []
        ys = []
        for batch in range(1,n_batches+1):
            X, Y = data_utils.generate_batch(data, labels, self.batch_size)
            f_dict = {
                self.inp : X,
                self.labels : Y,
            }    
            cost, y = self.session.run([self.loss,self.y], feed_dict=f_dict)
            test_cost += cost
            sys.stdout.write('Cost  :   %f - Batch %d of %d     \r' %(cost ,batch ,n_batches))
            sys.stdout.flush()

            preds.extend(np.argmax(y,1))
            ys.extend(Y)

        print
        print "Accuracy", np.mean(np.asarray(np.equal(ys, preds), dtype='float32'))*100