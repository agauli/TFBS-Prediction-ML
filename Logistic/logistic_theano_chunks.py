from __future__ import print_function

import theano
import theano.tensor as T
import numpy as np
import csv
import pandas
from pandas import DataFrame as df
from sklearn import metrics
import matplotlib.pyplot as plt
import time
import multiprocessing
import Queue

np.random.seed(3)

class Options(object):
    num_workers=4
    chunksize=10000
    learning_rate=0.5
    batch_size = 1000
    max_epochs = 300
    validation_portion = 0.2
    k=6

class LogisticRegression(object):
    """
    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=0.01 * np.random.rand(n_in, n_out).astype(theano.config.floatX), name='W', borrow=True)
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=np.random.rand(n_out).astype(theano.config.floatX), name='b', borrow=True)

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_prediction = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """
        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_prediction
        if y.ndim != self.y_prediction.ndim:
            raise TypeError(
                'y should have the same shape as self.y_prediction',
                ('y', y.type, 'y_prediction', self.y_prediction.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.sum(T.neq(self.y_prediction, y))
        else:
            raise NotImplementedError()

# not used right now
def generate_mini_batch_indices(total_num_sequences, num_batches):
    options = Options()
    batch_size = options.batch_size
    indices = np.random.permutation(total_num_sequences)
    list_of_indices = [indices[batch_size * i : batch_size *(i+1)] for i in range(num_batches)]
    return list_of_indices

def get_data_reader_iterator(path):
    options=Options()
    reader = pandas.read_table(path, sep='\t',chunksize=options.chunksize)
    return reader

def load_data(data):
    options = Options()
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
    del data
    return kmer_counts, cell_type_dict

def split_data(features, labels):
    # returns a list of tuples (features, labels)
    print("Splitting data into train, validation")
    options = Options()

    data=zip(features,labels)

    dim_features = features.shape[1]
    num_sequences = features.shape[0]
    np.random.seed(3)
    random_data_indices =np.random.permutation(num_sequences)
    # permuting the data
    data=[data[i] for i in random_data_indices]

    num_validation_sequences = int(num_sequences*options.validation_portion)
    num_train_sequences = num_sequences - num_validation_sequences

    print("num train sequences: %d"%num_train_sequences)
    print("num validation sequences: %d"%num_validation_sequences)

    train = data[:num_train_sequences]
    validation = data[num_train_sequences:]

    return train, validation

def prepare_data(list_of_tuples):
    def share_data(data_xy, borrow=True):
        # data_xy is a list of tuples
        """ Function that loads the dataset into shared variables
        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x = [data_xy[i][0] for i in range(len(data_xy))]
        data_y = [data_xy[i][1] for i in range(len(data_xy))]

        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')
    train_features, train_labels = share_data(list_of_tuples)
    return train_features, train_labels


class myThread(multiprocessing.Process):
    def __init__(self, threadID, data_iterator,queue):
        multiprocessing.Process.__init__(self)
        self.threadID = threadID
        self.data_iterator = data_iterator
        self.queue = queue

    def run(self):
        options = Options()
        for mini_batch_number, data in enumerate(self.data_iterator):
            if mini_batch_number % options.num_workers == self.threadID:
                print("thread %d mini batch number: %d" % (self.threadID,mini_batch_number))
                self.queue.put(data)


def main():
    options=Options
    path = "~/biodata/chr10.tsvfinal"
    threads = []
    manager = multiprocessing.Manager()
    queue = manager.Queue(maxsize=6)

    data_iterator=[]
    for i in range(options.num_workers):
        data_iterator.append(get_data_reader_iterator(path))

    for threadID in range(options.num_workers):
        thread = myThread(threadID, data_iterator[threadID],queue)
        thread.daemon=True
        threads.append(thread)
        thread.start()

    while True:
        try:
            print("queue size ", queue.qsize())
            data = queue.get(block=False)
            train_model(data)
        except:
            pass

    for t in threads:
        t.join()

    print("Finished all threads")


def train_model(data):
    options = Options()
    learning_rate = options.learning_rate
    batch_size = options.batch_size
    dim_features = 4 ** options.k

    kmer_counts, cell_type_dict = load_data(data)
    train, validation = split_data(kmer_counts,cell_type_dict['K562'])

    train_features, train_labels = prepare_data(train)
    validation_features, validation_labels = prepare_data(validation)

    num_train_sequences = train_features.get_value(borrow=True).shape[0]
    num_validation_sequences = validation_features.get_value(borrow=True).shape[0]
    num_sequences = num_train_sequences + num_validation_sequences

    num_train_batches =num_train_sequences // batch_size
    num_validation_batches = num_validation_sequences // batch_size

    print("%d training sequences "%num_train_sequences)
    print("%d validation sequences " %num_validation_sequences)

    # Symbolic Inputs
    x = T.matrix('x')
    y = T.ivector('y')
    index = T.lscalar()

    classifier = LogisticRegression(input=x, n_in= dim_features, n_out=2)
    cost = classifier.negative_log_likelihood(y)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    train_model = theano.function(
        inputs=[index],
        outputs=[cost],
        updates=updates,
        givens={
            x: train_features[batch_size * index : batch_size * (index+1)],
            y: train_labels[batch_size * index : batch_size * (index+1)]
        }
    )

    validation_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: validation_features[index * batch_size: (index + 1) * batch_size],
            y: validation_labels[index * batch_size: (index + 1) * batch_size]
        }
    )
    """
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: testing_features[index * batch_size: (index + 1) * batch_size],
            y: testing_labels[index * batch_size: (index + 1) * batch_size]
        }
    )
    """
    scores = theano.function(
        inputs=[],
        outputs=classifier.p_y_given_x,
        givens={
            x:validation_features
        }
    )


    epoch = 0
    max_epochs = options.max_epochs
    try:
        start_time = time.time()
        while epoch < max_epochs:
            epoch += 1
            train_sequences_seen = validation_sequcnes_seen = testing_sequences_seen = 0
            cost=0
            for i in range(num_train_batches):
                cost += train_model(i)[0]
            if epoch % 50 == 0:
                print("Epoch %d cost %.5f" %(epoch,cost))

    except KeyboardInterrupt:
        print("\nTraining interrupted")
    total_time = time.time()-start_time
    print("total time :",total_time)
    num_validation_errors = 0
    for i in range(num_validation_batches):
        num_validation_errors += validation_model(i)
    validation_accuracy = 1 - num_validation_errors / num_validation_sequences
    print("Validation accuracy: %.8f" % validation_accuracy)
    #print("Scores for the testing set:", scores()[:,1])

    y_scores = scores()[:,1]
    y_labels = validation_labels.eval()

    print("number of 1's in the training labels: ", sum(train_labels.eval()))
    print("number of 1's in the validation labels: ", sum(y_labels))
    try:
        print("Area under the curve: %.5f \n" %metrics.roc_auc_score(y_labels,y_scores))
    except:
        pass
    #fpr, tpr, thresholds = metrics.roc_curve(y_labels, y_scores, pos_label=1)
    """
    plt.figure()
    plt.plot(fpr,tpr)
    plt.xlim([-0.1,1.05])
    plt.ylim([-0.1,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()
    """




if __name__ == '__main__':
    main()










