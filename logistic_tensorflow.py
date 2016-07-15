from __future__ import print_function

import tensorflow as tf
import numpy as np
import csv
import pandas
from pandas import DataFrame as df
from sklearn import metrics
import matplotlib.pyplot as plt
import time

np.random.seed(3)

class Options(object):
    learning_rate=0.5
    batch_size = 16
    max_epochs = 300
    test_portion = 0.2
    validation_portion = 0.1
    k=6
    dim_features = 4**k

class LogitModel():
    def __init__(self,is_training):
        options = Options()
        dim_features = options.dim_features
        batch_size = options.batch_size

        self.inputs = tf.placeholder(tf.int32, shape=[None, dim_features], name="inputs")
        self.labels = tf.placeholder(tf.int32, shape=[None, 2], name="labels")

        W = 0.01 * np.random.rand(dim_features, 2)
        b = np.random.rand(2, 1)
        with tf.variable_scope("logit"):
            W = tf.get_variable('W', shape=[dim_features, 2],initializer=tf.constant_initializer(W), dtype=tf.float32)
            b = tf.get_variable('b', shape=[2],initializer=tf.constant_initializer(b), dtype=tf.float32)

        self.softmax_probabilities = tf.nn.softmax(tf.matmul(tf.cast(self.inputs, dtype=tf.float32),W) + b)
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.cast(self.labels,dtype=tf.float32)* tf.log(self.softmax_probabilities), reduction_indices=[1]))
        self.predictions = tf.argmax(self.softmax_probabilities, dimension=1)
        self.num_correct_predictions = tf.reduce_sum(
            tf.cast(tf.equal(self.predictions, tf.argmax(self.labels, 1)), dtype=tf.float32))
        self.train_step = tf.train.GradientDescentOptimizer(options.learning_rate).minimize(self.cross_entropy)

        self.num_correct_prediction = tf.equal(tf.argmax(self.softmax_probabilities, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.num_correct_prediction, tf.float32))




def load_data(path):
    options = Options()
    data = pandas.read_csv(path, sep='\t')
    kmer_counts = np.array(data.ix[:,'AAAAAA':'TTTTTT']).astype(np.int32)
    def labels_to_int_map(input):
        if input=='B':
            return 1
        else:
            return 0

    cell_type_dict={}
    cell_type_dict['MCF_7_labels'] = map(labels_to_int_map, data['MCF-7']) # 0 bounds
    cell_type_dict['IMR-90'] = map(labels_to_int_map, data['IMR-90']) # 5 bounds
    cell_type_dict['K562'] = map(labels_to_int_map, data['K562'])  # 5 bounds
    cell_type_dict['HepG2'] = map(labels_to_int_map, data['HepG2']) # 4 bounds
    cell_type_dict['HeLa-S3'] = map(labels_to_int_map, data['HeLa-S3']) # 4 bounds
    cell_type_dict['H1-hESC'] = map(labels_to_int_map, data['H1-hESC']) # 0 bounds
    cell_type_dict['A549'] = map(labels_to_int_map, data['A549']) # 5 bounds

    return kmer_counts, cell_type_dict

def split_data(features, labels):
    # returns a list of tuples (features, labels)
    print("Splitting data into train, validation, test")
    options = Options()

    data=zip(features,labels)

    dim_features = features.shape[1]
    num_sequences = features.shape[0]
    random_data_indices =np.random.permutation(num_sequences)
    # permuting the data
    data=[data[i] for i in random_data_indices]

    num_testing_sequences = int(num_sequences*options.test_portion)
    num_train_sequences = num_sequences - num_testing_sequences
    num_validation_sequences = int(num_train_sequences * options.validation_portion)
    num_train_sequences = num_train_sequences - num_validation_sequences

    print("num train sequences: %d"%num_train_sequences)
    print("num validation sequences: %d"%num_validation_sequences)
    print("num test sequences: %d"%num_testing_sequences)

    train = data[:num_train_sequences]
    validation = data[num_train_sequences:num_train_sequences+num_validation_sequences]
    test = data[num_train_sequences+num_validation_sequences:]

    return train, validation, test

def prepare_data(list_of_tuples):
    features = [list_of_tuples[i][0] for i in range(len(list_of_tuples))]
    labels = [list_of_tuples[i][1] for i in range(len(list_of_tuples))]
    labels = binary_one_hot(labels)

    features= np.asarray(features).astype('int32')
    labels = np.asarray(labels).astype('int32')

    return features, labels

def binary_one_hot(x):
    try:
        if type(x).__module__ == np.__name__:
            dim0 = x.shape[0]
        elif isinstance(x, list):
            dim0 = len(x)
        else:
            raise TypeError
    except TypeError:
        print("Expecting input type to be one of {list, numpy.ndarray}. Received %s" % type(x))

    dim1 = 2
    output = np.zeros((dim0, dim1))
    for i in range(dim0):
        output[i, x[i]] = 1
    return output

def main():
    path = "/Users/Derrick/DREAM-challenge/CTCF_subsample.tsv"
    options = Options()
    learning_rate = options.learning_rate
    batch_size = options.batch_size
    dim_features = 4 ** options.k

    kmer_counts, cell_type_dict = load_data(path)
    train, validation, test = split_data(kmer_counts, cell_type_dict['K562'])

    train_features, train_labels = prepare_data(train)
    validation_features, validation_labels = prepare_data(validation)
    testing_features, testing_labels = prepare_data(test)

    num_train_sequences = train_features.shape[0]
    num_validation_sequences = validation_features.shape[0]
    num_testing_sequences = testing_features.shape[0]
    num_sequences = num_testing_sequences + num_validation_sequences + num_testing_sequences

    num_train_batches = num_train_sequences // batch_size
    num_validation_batches = num_validation_sequences // batch_size
    num_testing_batches = num_testing_sequences // batch_size

    with tf.variable_scope("model"):
        m_train = LogitModel(is_training=True)
    with tf.variable_scope("model",reuse=True):
        m_validation = LogitModel(is_training=False)
        m_test = LogitModel(is_training=False)

    sess=tf.Session()
    sess.run(tf.initialize_all_variables())
    start_time = time.time()
    for i in range(options.max_epochs):
        _, cross_entropy= sess.run([m_train.train_step, m_train.cross_entropy],
                                    feed_dict={m_train.inputs: train_features, m_train.labels: train_labels})
        print("Epoch %d cost: %.6f"%(i+1,cross_entropy))
    total_time = time.time()-start_time
    print("total time: ",total_time)
    softmax_probabilities = sess.run(m_train.softmax_probabilities,
                                     feed_dict={m_train.inputs: testing_features, m_train.labels: testing_labels})

    y_scores = softmax_probabilities[:,1]
    y_labels = np.argmax(testing_labels,axis=1)
    print("scores: ",y_scores)
    print("Area under the curve: ", metrics.roc_auc_score(y_labels, y_scores))
    fpr, tpr, thresholds = metrics.roc_curve(y_labels, y_scores, pos_label=1)
    print("num positives in train set: ", np.sum(train_labels[:, 1]))
    print("num positives in test set: ", np.sum(testing_labels[:, 1]))
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlim([-0.1, 1.05])
    plt.ylim([-0.1, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()




if __name__ == '__main__':
    main()







