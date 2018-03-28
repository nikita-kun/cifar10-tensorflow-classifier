#convolutional nn, adam training

import numpy as np
import tensorflow as tf
import os
# class written to replicate input_data from tensorflow.examples.tutorials.mnist for CIFAR-10
import cifar10_read


# location of the CIFAR-10 dataset
#CHANGE THIS PATH TO THE LOCATION OF THE CIFAR-10 dataset on your local machine
data_dir = '/media/wasp/EXFAT/tensorflow/unzip/cifar-10-batches-py/'

# read in the dataset
print('reading in the CIFAR10 dataset')
dataset = cifar10_read.read_data_sets(data_dir, distort_train=True, one_hot=True, reshape=False)   

using_tensorboard = True

##################################################
# PHASE 1  - ASSEMBLE THE GRAPH

# 1.1) define the placeholders for the input data and the ground truth labels

# x_input can handle an arbitrary number of input vectors of length input_dim = d 
# y_  are the labels (each label is a length 10 one-hot encoding) of the inputs in x_input
# If x_input has shape [N, input_dim] then y_ will have shape [N, 10]

input_dim = 32*32*3    # d
x_input = tf.placeholder(tf.float32, shape = [None, 32, 32, 3])
y_ = tf.placeholder(tf.float32, shape = [None, 10])
m = 100 #hidden nodes
learning_adam_rate = 0.001
nF = 40
f = 5 #filter size

# variables
F = tf.Variable(tf.truncated_normal([f, f, 3, nF], stddev=.01))
b = tf.Variable(tf.constant(0.1, shape=[nF]))
W1 = tf.Variable(tf.truncated_normal([nF*16*16, m], stddev=.01))
b1 = tf.Variable(tf.constant(0.1, shape=[m]))
W2 = tf.Variable(tf.truncated_normal([m, 10], stddev=.01))
b2 = tf.Variable(tf.constant(0.1, shape=[10]))

S = tf.nn.conv2d(x_input, F, strides=[1,1,1,1], padding='SAME') + b # Convolutions (22), (23)
X1 = tf.nn.relu(S) # (24)

H = tf.nn.max_pool(X1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME') # (25)
vecH = tf.reshape(H, [-1, int(16*16*nF)])

s1 = tf.matmul(vecH, W1) + b1 # (26)
x1 = tf.nn.relu(s1) #(27)
y = tf.matmul(x1, W2) + b2 # (28)

# 1.4) define the loss funtion 
# cross entropy loss: 
# Apply softmax to each output vector in y to give probabilities for each class then compare to the ground truth labels via the cross-entropy loss and then compute the average loss over all the input examples
# Obsolete (replace with something newer)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# Training with AdamOptimizer
train_step = tf.train.AdamOptimizer(learning_adam_rate).minimize(cross_entropy)

# definition of accuracy, count the number of correct predictions where the predictions are made by choosing the class with highest score
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  

# 1.6) Add an op to initialize the variables.
init = tf.global_variables_initializer()

##################################################


# If using TENSORBOARD
if using_tensorboard:
    # keep track of the loss and accuracy for the training set
    tf.summary.scalar('training loss', cross_entropy, collections=['training'])
    tf.summary.scalar('training accuracy', accuracy, collections=['training'])
    # merge the two quantities
    tsummary = tf.summary.merge_all('training')
    
    # keep track of the loss and accuracy for the validation set
    tf.summary.scalar('validation loss', cross_entropy, collections=['validation'])
    tf.summary.scalar('validation accuracy', accuracy, collections=['validation'])
    # merge the two quantities
    vsummary = tf.summary.merge_all('validation')

##################################################


##################################################
# PHASE 2  - PERFORM COMPUTATIONS ON THE GRAPH

n_iter = 1000000

# 2.1) start a tensorflow session
with tf.Session() as sess:

    ##################################################
    # If using TENSORBOARD
    if using_tensorboard:
        # set up a file writer and directory to where it should write info + 
        # attach the assembled graph
        summary_writer = tf.summary.FileWriter('network3/results3', sess.graph)
    ##################################################

    # 2.2)  Initialize the network's parameter variables
    # Run the "init" op (do this when training from a random initialization)
    sess.run(init) 


    # 2.3) loop for the mini-batch training of the network's parameters
    for i in range(n_iter):
        	
        # grab a random batch (size nbatch) of labelled training examples
        nbatch = 200
        batch = dataset.train.next_batch(nbatch)

        # create a dictionary with the batch data 
        # batch data will be fed to the placeholders for inputs "x_input" and labels "y_"
        batch_dict = {
            x_input: batch[0], # input data
            y_: batch[1], # corresponding labels
         }
        
        # run an update step of mini-batch by calling the "train_step" op 
        # with the mini-batch data. The network's parameters will be updated after applying this operation
        sess.run(train_step, feed_dict=batch_dict)
	
	print 'validation start'
        # periodically evaluate how well training is going
        if i % 500 == 0:

            # compute the performance measures on the training set by
            # calling the "cross_entropy" loss and "accuracy" ops with the training data fed to the placeholders "x_input" and "y_"
            
            tr = sess.run([cross_entropy, accuracy], feed_dict = {x_input:dataset.train.images[1:5000], y_: dataset.train.labels[1:5000]})

            # compute the performance measures on the validation set by
            # calling the "cross_entropy" loss and "accuracy" ops with the validation data fed to the placeholders "x_input" and "y_"

            val = sess.run([cross_entropy, accuracy], feed_dict={x_input:dataset.validation.images, y_:dataset.validation.labels})            

            info = [i] + tr + val
            print(info)

            ##################################################
            # If using TENSORBOARD
            if using_tensorboard:

                # compute the summary statistics and write to file
                summary_str = sess.run(tsummary, feed_dict = {x_input:dataset.train.images[1:5000], y_: dataset.train.labels[1:5000]})
                #summary_str = sess.run(tsummary, feed_dict = {x_input:dataset.train.images, y_: dataset.train.labels})
                summary_writer.add_summary(summary_str, i)

                summary_str1 = sess.run(vsummary, feed_dict = {x_input:dataset.validation.images, y_: dataset.validation.labels})
                summary_writer.add_summary(summary_str1, i)
            ##################################################

        print 'validation end'
    # evaluate the accuracy of the final model on the test data
    test_acc = sess.run(accuracy, feed_dict={x_input: dataset.test.images, y_: dataset.test.labels})
    final_msg = 'test accuracy:' + str(test_acc)
    print(final_msg)

##################################################
