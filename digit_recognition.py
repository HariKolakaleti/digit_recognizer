#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 22:19:29 2017

@author: harik
"""

""""
    model architecture
    ------------------
    # inputs
    # layer_1
        - conv1
        - relu1
        - maxpool1
    # layer_2
        - conv2
        - relu2
        - maxpool2
    # layer_3
        - conv3
        - relu3    
        - dropout
    # output
        - fc1
        - relu4
        - softmax[1:5]
"""

#%%

import pickle
import random
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

#%%

# mnist dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True) 

#%%

# import SVHN database

print('Loading pickled data...')

pickle_file = 'svhn_data/SVHN.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    X_train = save['train_dataset']
    y_train = save['train_labels']
    X_test = save['test_dataset']
    y_test = save['test_labels']
    del save  
    print('Training data shape:', X_train.shape)
    print('Training label shape:',y_train.shape)
    print('Test data shape:', X_test.shape)
    print('Test label shape:', y_test.shape)

print('Data successfully loaded!')

#%%

def display_samples(num_samples=1):
    for i in range(num_samples):        

        idx = random.choice(range(X_train.shape[0]))
        print('Display sample train image:', idx)
        plt.imshow(X_train[idx].reshape(32,32), interpolation='nearest')
        plt.show()
        
        idx = random.choice(range(X_test.shape[0]))
        print('Display sample test image:', idx)
        plt.imshow(X_test[idx].reshape(32,32), interpolation='nearest')
        plt.show()

display_samples()

#%%

# params
batch_size = 1         # batch size
img_size   = 32        # image size 32x32
in_chan    = 1         # grey scale

# conv1
c1_patch   = 5         # patch size 5x5
c1_stride  = 1         # stride 1
c1_depth   = 16        # 16 features (out channels)
c1_padding = 'VALID'   # padding valid
c1_stride  = [1,1,1,1] # stride 1x1

# maxpool1
p1_padding = 'VALID'   # padding valid
p1_patch   = [1,2,2,1] # patch size 2x2
p1_stride  = [1,2,2,1] # stride 2x2

# conv2
c2_patch   = 5         # patch size 5x5
c2_depth   = 32        # 16 features (out channels)
c2_padding = 'VALID'   # padding valid
c2_stride  = [1,1,1,1] # stride 1x1

# maxpool2
p2_padding = 'VALID'   # padding valid
p2_patch   = [1,2,2,1] # patch size 2x2
p2_stride  = [1,2,2,1] # stride 2x2

# conv3
c3_patch   = 5         # patch size 5x5
c3_depth   = 96        # 16 features (out channels)
c3_padding = 'VALID'   # padding valid
c3_stride  = [1,1,1,1] # stride 1x1

# fc
keep_prob  = 0.8       # dropout rate 0.2
fc_nodes   = 64        # hidden layer

# output
out_digits = 6         # up to 5 digits [1-5]
out_labels = 11        # 10 (detect 0-9)

#%%

# cnn model architecture

graph = tf.Graph()

with graph.as_default():
    
    def init_weight(name, shape, init='conv2d'):
        if init == 'conv2d':
            initializer = tf.contrib.layers.xavier_initializer_conv2d()
        else:
            initializer = tf.contrib.layers.xavier_initializer()
        return tf.get_variable(shape=shape, name=name, initializer=initializer)
                
    def init_bias(name, shape):
        return tf.Variable(tf.constant(1.0, shape=shape), name=name)

    X = tf.placeholder(tf.float32, shape=[batch_size, img_size, img_size, in_chan])
    Y = tf.placeholder(tf.int32, shape=[batch_size, out_digits])

    tf_test_dataset = tf.constant(X_test)

    b_C1 = init_bias(name='b_C1', shape=[c1_depth])
    b_C2 = init_bias(name='b_C2', shape=[c2_depth])
    b_C3 = init_bias(name='b_C3', shape=[c3_depth])
    b_FC = init_bias(name='b_FC', shape=[fc_nodes])
        
    W_C1 = init_weight(name='W_C1', shape=[c1_patch, c1_patch, in_chan,  c1_depth])
    W_C2 = init_weight(name='W_C2', shape=[c2_patch, c2_patch, c1_depth, c2_depth])
    W_C3 = init_weight(name='W_C3', shape=[c3_patch, c3_patch, c2_depth, c3_depth])
    W_FC = init_weight(name='W_FC', shape=[c3_depth, fc_nodes])
        
    b_Y1 = init_bias(name='b_Y1', shape=[out_labels])
    b_Y2 = init_bias(name='b_Y2', shape=[out_labels])
    b_Y3 = init_bias(name='b_Y3', shape=[out_labels])
    b_Y4 = init_bias(name='b_Y4', shape=[out_labels])
    b_Y5 = init_bias(name='b_Y5', shape=[out_labels])
    b_Y  = [b_Y1, b_Y2, b_Y3, b_Y4, b_Y5]
        
    W_Y1 = init_weight(name='W_Y1', shape=[fc_nodes, out_labels])
    W_Y2 = init_weight(name='W_Y2', shape=[fc_nodes, out_labels])
    W_Y3 = init_weight(name='W_Y3', shape=[fc_nodes, out_labels])
    W_Y4 = init_weight(name='W_Y4', shape=[fc_nodes, out_labels])
    W_Y5 = init_weight(name='W_Y5', shape=[fc_nodes, out_labels])
    W_Y  = [W_Y1, W_Y2, W_Y3, W_Y4, W_Y5]

    def model(X, keep_prob):
        with tf.name_scope('layer_1'):
            c1_out = tf.nn.conv2d(X, W_C1, c1_stride, padding=c1_padding)
            r1_out = tf.nn.relu(c1_out + b_C1)
            p1_out = tf.nn.max_pool(r1_out, p1_patch, p1_stride, padding=p1_padding)
        
        with tf.name_scope('layer_2'):
            c2_out = tf.nn.conv2d(p1_out, W_C2, c2_stride, padding=c2_padding)
            r2_out = tf.nn.relu(c2_out + b_C2)
            p2_out = tf.nn.max_pool(r2_out, p2_patch, p2_stride, padding=p2_padding)
        
        with tf.name_scope('layer_3'):
            c3_out = tf.nn.conv2d(p2_out, W_C3, c3_stride, padding=c3_padding)
            r3_out = tf.nn.relu(c3_out + b_C3)
            d1_out = tf.nn.dropout(r3_out, keep_prob)
        
        with tf.name_scope('fc_layer'):
            shape   = d1_out.get_shape().as_list()
            reshape = tf.reshape(d1_out, [shape[0], shape[1] * shape[2] * shape[3]])
            fc_out  = tf.nn.relu(tf.matmul(reshape, W_FC) + b_FC)
        
        with tf.name_scope('softmax'):                
            y1 = tf.matmul(fc_out, W_Y[0]) + b_Y[0]
            y2 = tf.matmul(fc_out, W_Y[1]) + b_Y[1]
            y3 = tf.matmul(fc_out, W_Y[2]) + b_Y[2]
            y4 = tf.matmul(fc_out, W_Y[3]) + b_Y[3]
            y5 = tf.matmul(fc_out, W_Y[4]) + b_Y[4]
            
        return [y1, y2, y3, y4, y5]

    [y1, y2, y3, y4, y5] = model(X, keep_prob)

    with tf.name_scope("cross_entropy"):        
        cross_entropy = \
            tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y1, Y[:, 1])) + \
            tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y2, Y[:, 2])) + \
            tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y3, Y[:, 3])) + \
            tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y4, Y[:, 4])) + \
            tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y5, Y[:, 5]))
        tf.summary.scalar("cross_entropy", cross_entropy)

    # optimizer
    alpha = 0.05; 
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(alpha, global_step, 10000, 0.96)
            
    optimizer  = tf.train.AdagradOptimizer(learning_rate)
    train_step = optimizer.minimize(cross_entropy, global_step=global_step)

    def softmax_combine(X):
        train_pred = tf.pack([
            tf.nn.softmax(model(X, 1.0)[0]),
            tf.nn.softmax(model(X, 1.0)[1]),
            tf.nn.softmax(model(X, 1.0)[2]),
            tf.nn.softmax(model(X, 1.0)[3]),
            tf.nn.softmax(model(X, 1.0)[4])])
        return train_pred

    train_pred = softmax_combine(X)
    test_pred  = softmax_combine(tf_test_dataset)

    '''Save Model (will be initiated later)'''
    saver = tf.train.Saver()

    # weight histogram
    tf.summary.histogram("W_C1", W_C1)
    tf.summary.histogram("W_C2", W_C2)
    tf.summary.histogram("W_C3", W_C3)
    tf.summary.histogram("W_FC", W_FC)
    tf.summary.histogram("W_Y1", W_Y1)
    tf.summary.histogram("W_Y2", W_Y2)
    tf.summary.histogram("W_Y3", W_Y3)
    tf.summary.histogram("W_Y4", W_Y4)
    tf.summary.histogram("W_Y5", W_Y5)

    tf.summary.histogram("b_C1", b_C1)
    tf.summary.histogram("b_C2", b_C2)
    tf.summary.histogram("b_C3", b_C3)
    tf.summary.histogram("b_FC", b_FC)
    tf.summary.histogram("b_Y1", b_Y1)
    tf.summary.histogram("b_Y2", b_Y2)
    tf.summary.histogram("b_Y3", b_Y3)
    tf.summary.histogram("b_Y4", b_Y4)
    tf.summary.histogram("b_Y5", b_Y5)

    print('Graph done!')

#%%

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 2).T == labels)
            / predictions.shape[1] / predictions.shape[0])

def get_offset(step, batch_size, data):
    offset = (step * batch_size) % (data.shape[0] - batch_size)
    return offset

with tf.Session(graph=graph) as sess:
    writer = tf.summary.FileWriter("log", sess.graph)
    merged = tf.summary.merge_all()

    sess.run(tf.global_variables_initializer())
    
    num_steps = 2
    for step in range(num_steps):
        offset  = get_offset(step, batch_size, y_train)
        batch_X = X_train[offset:(offset + batch_size), :, :, :]
        batch_Y = y_train[offset:(offset + batch_size), :]
        train_data = {X: batch_X, Y: batch_Y}
        _, l, pred, summary = sess.run([train_step, cross_entropy, train_pred, merged], feed_dict=train_data)

        writer.add_summary(summary)
        if (step % 500 == 0):
            print(('Minibatch loss at step {}: {}').format(step, l))
            print(('Minibatch accuracy: {}%'.format(accuracy(pred, batch_Y[:,1:6]))))

    print(
    ('Test accuracy: {}%'.format(accuracy(test_pred.eval(), y_test[:,1:6]))))

    save_path = saver.save(sess, "digit_recognizer.ckpt")
    print('Model saved to file: {}'.format(save_path))


print('Tensorboard: tensorboard --logdir=log')
