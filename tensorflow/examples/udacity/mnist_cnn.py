from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10
IMG_SIZE = 28
IMG_WIDTH = 28
IMG_HEIGHT = 28


def train():

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    sess = tf.InteractiveSession()

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape=[None, IMG_WIDTH * IMG_HEIGHT], name='x-input')
        y_ = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES], name='y-input')

    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, IMG_HEIGHT, IMG_WIDTH, 1])
        tf.image_summary('input', image_shaped_input, NUM_CLASSES)
    
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        """ returns a vanila 1-stride, 0-padded conv """
        return tf.nn.conv2d(x, W, 
                            strides=[1,1,1,1],  # stride=1
                            padding='SAME')

    def max_pool_2x2(x):
        """ vanila 2x2 max pool """
        return tf.nn.max_pool(x, 
                            ksize=[1, 2, 2, 1], 
                            strides=[1, 2, 2, 1],
                            padding='SAME')

    def variable_summaries(var): 
        """ Attach summaries to variables """
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            #tf.summary.histogram('histogram', var)
    
    def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
        ''' Resuable code ot make a simple nn layer. 
            
            Does matrix multiple, bias add, and then relu to non-linearize 
            Also does name scoping
        '''
        with tf.name_scope(layer_name):
            # Weights
            with tf.name_scope('weights'):
                weights = weight_variable([input_dim, output_dim])
                variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = bias_variable([output_dim])
                variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
                #tf.summary.histogram('pre_activations', preactivate)
            # apply activation
            activations = act(preactivate, name='activation')
            #tf.summary.histogram('activations', activations)
            return activations

    # create 1st hidden layer
    hidden1 = nn_layer(x, 784, 256, 'layer1')

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_prob', keep_prob)
        # apply dropout
        dropped = tf.nn.dropout(hidden1, keep_prob)
    
    # Dont apply softmax activation yet
    # Dropout layer
    y = nn_layer(dropped, 256, NUM_CLASSES, 'layer2', act=tf.identity)

    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(y, y_)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
    # add cross entropy
    tf.summary.scalar('cross_entropy', cross_entropy)

    ## Train step: 
    with tf.name_scope('train'):
        learning_rate = 0.5
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    
    # accuracy
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # Merge all summaries and write them out
    merged = tf.summary.merge_all()
    train_writer = tf.train.SummaryWriter('/tmp/train')
    test_writer  = tf.train.SummaryWriter('/tmp/test')

    # init
    tf.initialize_all_variables().run()

    def feed_dict(train):
        if train:
            xs, ys = mnist.train.next_batch(100)
            k = 0.8  # i.e. dropout = 20% 
        else:
            xs, ys = mnist.test.images, mnist.test.labels
            k = 1.0 # no dropout on Test
        return {x: xs, y_: ys, keep_prob: k}
    
    # Run the model
    for i in range(2001):   # 
        if i % 10 == 0:    # record summaries and test-set acc
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary( summary, i)
            print('Acc at step %s: %s' % (i, acc))
        else: 
            summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
            train_writer.add_summary(summary, i)
    
    train_writer.close()
    test_writer.close()

if __name__ == '__main__':
    train()

