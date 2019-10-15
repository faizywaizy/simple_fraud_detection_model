# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 19:48:00 2019

Splits creditcard.csv data
builds model
trains model
optimizes and minimizes cross entropy, y_train vs y_predict

@author: Faizan
"""

import pandas as pd
import numpy as np

#import ant store the data set
credit_card_data = pd.read_csv('creditcard.csv')
#print(credit_card_data)

#To split the data into useable dataframes:
#1. Shuffle or randomize the data, helps break up clumps / bias
#2. One-hot encoding// #easier to feed data
#3. Normalize our datac convert values into 0 < x < 1
#4. Splitting X/y values - y is legit
#5. Convert into the dataframes into numpy arrays (float32)
#6. Splitting the final data into X/y train/test

#shuffle and randomize data
rand_data = credit_card_data.sample(frac=1)
one_hot_data = pd.get_dummies(rand_data, columns = ['Class'])
#change class into [01] and [10] for legit and fraud transactions respectively
formatted_data = (one_hot_data - one_hot_data.min()) / (one_hot_data.max() - one_hot_data.min())
dataframe_X = formatted_data.drop(['Class_0', 'Class_1'], axis = 1)
dataframe_y = formatted_data[['Class_0','Class_1']]

# convert data frames into numpy arrays
array_X, array_y  = np.asarray(dataframe_X.values, dtype = 'float32'), np.asarray(dataframe_y.values, dtype = 'float32')

# Allocate data for 20% testing and 80% training
train_length = int(0.8 * len(array_X))
(raw_X_train, raw_y_train) = (array_X[:train_length], array_y[:train_length])
(raw_X_test, raw_y_test) = (array_X[train_length:], array_y[train_length:])

# finding the percentage of fraud data provided from the data set, very small so I must weight it heavier
count_legit, count_fraud = np.unique(credit_card_data['Class'], return_counts = True)[1]
fraud_ratio = float(count_fraud / (count_legit + count_fraud))

print('Percent of fraudulent transactions: ', fraud_ratio)

# adding logical weighting to training datasets due to low fraudulent transactions
# common when dealing with unbalanced datasets, pay close attention to what I need

weighting = 1/fraud_ratio
raw_y_train[:, 1] = raw_y_train[:, 1] * weighting

import tensorflow as tf

input_dimensions = array_X.shape[1]
output_dimensions = array_y.shape[1]

num_layer_1_cells = 100
num_layer_2_cells = 150

# inputs into model
X_train_node = tf.placeholder(tf.float32, [None, input_dimensions], name = 'X_train')
y_train_node = tf.placeholder(tf.float32, [None, output_dimensions], name = 'y_train')

X_test_node = tf.constant(raw_X_test, name = 'X_test')
y_test_node = tf.constant(raw_y_test, name = 'y_test')

#the output of one layer is the input of the next layer
# how do we choose the layers ? A whole lot!! of trial and error
## zeros() -> creates zero matrices

weight_1_node = tf.Variable(tf.zeros([input_dimensions, num_layer_1_cells]), name = 'weight_1')
biases_1_node = tf.Variable(tf.zeros([num_layer_1_cells]), name = 'biases_1')

weight_2_node = tf.Variable(tf.zeros([num_layer_1_cells, num_layer_2_cells]), name = 'weight_2')
biases_2_node = tf.Variable(tf.zeros([num_layer_2_cells]), name = 'biases_2')

weight_3_node = tf.Variable(tf.zeros([num_layer_2_cells, output_dimensions]), name = 'weight_3')
biases_3_node = tf.Variable(tf.zeros([output_dimensions]), name = 'biases_3')

# optimize loss by adjusting the zeros in layers 


# funtion runs through layers and returns new tensor
## nn -> neural netework
## matmul -> matrix multiplication
## sigmoid() -> best graph fitting model
## dropout() -> prevents nn from being too lazy when recieving verified input successes
## softmax() -> works well with one_hot_data output

def network(input_tensor):
    layer1 = tf.nn.sigmoid(tf.matmul(input_tensor, weight_1_node) + biases_1_node)
    layer2 = tf.nn.dropout(tf.nn.sigmoid(tf.matmul(layer1, weight_2_node) + biases_2_node), 0.85)
    layer3 = tf.nn.softmax(tf.matmul(layer2, weight_3_node) + biases_3_node)
    return layer3

# prediction variables for y test and y test
# these are placeholders, will get values at runtime

y_train_prediction = network(X_train_node)
y_test_prediction = network(X_test_node)

# use softmax for loss calcs because I am using one_hot_data
cross_entropy = tf.losses.softmax_cross_entropy(y_train_node, y_train_prediction)

#optimizer, gradient descent to minimize, learning rate of 0.005
optimizer = tf.train.AdamOptimizer(0.005).minimize(cross_entropy)

# create fucntion to predict accuracy of output

def calculate_accuracy(actual, predicted):
    actual = np.argmax(actual, 1)
    predicted = np.argmax(predicted, 1)
    return (100*(np.sum(np.equal(predicted, actual)) / predicted.shape[0]))

#number of times to run model
num_epochs = 100

import time

with tf.Session() as session:
    tf.global_variables_initializer().run()
    for epoch in range(num_epochs):
        
        start_time = time.time()
        # feed dictionary as inputs
        _, cross_entropy_score = session.run([optimizer, cross_entropy], 
                                             feed_dict={X_train_node: raw_X_train, y_train_node: raw_y_train})
        if epoch % 10 == 0:
            timer = time.time() - start_time
            print("Epoch: {}".format(epoch), 'Current loss: {0:0.4f}'.format(cross_entropy_score),
                  'Elapsed time: {0:.2f}'.format(timer))
        final_y_test = y_test_node.eval()
        final_y_test_prediction = y_test_prediction.eval()
        final_accuracy = calculate_accuracy(final_y_test, final_y_test_prediction)
        print("Epoch accuracy: {0:.2f}".format(final_accuracy))
        
    # from one h
    final_y_test = y_test_node.eval()
    final_y_test_prediction = y_test_prediction.eval()
    final_accuracy = calculate_accuracy(final_y_test, final_y_test_prediction)
    print("Current accuracy: {0:.2f}%".format(final_accuracy))
    
final_fraud_y_test = final_y_test[final_y_test[:,1] == 1]
final_fraud_y_test_prediction = final_y_test_prediction[final_y_test[:,1] == 1]
final_fraud_accuracy = calculate_accuracy(final_fraud_y_test, final_fraud_y_test_prediction)
print('Final fraud specific accuracy: {0:.2f}%'.format(final_fraud_accuracy))
    

    
    










 