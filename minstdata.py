from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data #imports training data

import tensorflow as tf

FLAGS = None #not sure what this line does

def main(_):
    #import data 
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot = True)

    #creating the model
    x = tf.placeholder(tf.float32, [None, 784]) #creates a placeholder. 
    #the none indicates that we can include as many images as we want
    #the 784 is the representation of 28 x 28 pictures into a 1 dimensional array
    W = tf.Variable(tf.zeros([784, 10]))
    print("W", W)
    b = tf.Variable(tf.zeros([10]))
    print("b", b)
    y = tf.matmul(x, W) + b


    #defining loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y)
    )
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    #the shit above basically attempts to reduce "entropy"
    #cross entrpy - how well data describes truth
    #the gradient descent optimizer takes the gradient and moves in that direction
    #the 0.5 is the step count

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    ###TRAINING###

    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100) #training in batches. for some reason, less computationally intensive
        sess.run(train_step, feed_dict = {x: batch_xs, y_:batch_ys})

    #testing the trained mdoel

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    print(sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels}))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
