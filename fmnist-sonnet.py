# Import necessary libraries
import numpy as np
import sonnet as snt
import tensorflow as tf

import time  # To time each epoch

# Needed to download Fashion-MNIST (FMNIST) dataset without much hassle.
from tensorflow.keras import datasets


def get_fashion_MNIST_data():
    """ Download Fashion MNIST dataset. """

    # Step 1: Get the Data
    train_data, test_data = datasets.fashion_mnist.load_data()  # Download FMNIST

    # Step 2: Preprocess Dataset
    """ Centering and Normalization
        Perform centering by mean subtraction, and normalization by dividing with 
        the standard deviation of the training dataset.
    """
    train_data_mean = np.mean(train_data[0])
    train_data_stdev = np.std(train_data[0])
    train_data = ((train_data[0] - train_data_mean) /
                  train_data_stdev, train_data[1])
    test_data = ((test_data[0] - train_data_mean) /
                 train_data_stdev, test_data[1])

    return train_data, test_data


# Step 3: Build the model

# Multi-Layer Perceptron
class FMNISTMLPClassifier(snt.AbstractModule):
    """ Model for Classifying FMNIST dataset based on a Multi-Layer Perceptron """

    def __init__(self, name='fmnist_mlp_classifier'):
        """ Initialize the MLP based classifier.

            We have only initialized the module name to keep things simple. In 
            practice we should make our models more configurable, like including 
            a parameter to choose the number of hidden layers in our MLP.
        """

        super().__init__(name=name)

    def _build(self, inputs):
        """ Build the model stack.

            Stacks the necessary modules/operations needed to build the model.

            Model Stack:            
            1. BatchFlatten: Flattens the image tensor into a 1-D tensor so that 
                each pixel of the image has a corresponding input neuron.
            2. MLP: A configurable MLP module that comes with Sonnet. We'll 
                configure it as a two layer MLP with 128 neurons in the hidden 
                layer and 10 neurons in the output layer.        
        """

        outputs = snt.BatchFlatten()(inputs)  # Input layer with 784 neurons
        outputs = snt.nets.MLP(  # MLP module from Sonnet
            output_sizes=[128, 10],
            name='fmnist_mlp'
        )(outputs)

        return outputs


# Convolutional Neural Network
class FMNISTConvClassifier(snt.AbstractModule):
    """ Model for Classifying FMNIST dataset based on a Multi-Layer Perceptron """

    def __init__(self, name='fmnist_conv_classifier'):
        """ Initialize the conv net based classifier.

            We have only initialized the module name to keep things simple. In 
            practice we should make our models more configurable, like including 
            a parameter to choose the number of convolutional layers in the Conv net.
        """

        super().__init__(name=name)

    def _build(self, inputs):
        """ Build the model stack.

            Stacks the necessary modules/operations needed to build the model.

            Model Stack:            
            1. expand_dims: Tensorflow op to expand dimensions of the input 
                data to match the shape required by the Conv net.
            2. ConvNet2D: A configurable 2D Convolutional Neural Network module 
                that comes with Sonnet. We'll configure it to have two 
                convolutional layers. For more details on conv nets refer the 
                CS231n stanford course notes (http://cs231n.github.io/convolutional-networks/)
            3. BatchFlatten: Flattens the output tensor from the conv net into 
                a 1-D tensor so that each pixel of the image has a corresponding 
                input neuron.
            4. MLP: the fully connected network module that comes with Sonnet 
                (essentially an MLP). We'll configure it as a two layer MLP with 
                64 neurons in the hidden layer and 10 neurons in the output layer.  
        """

        # Shape: (BATCH_SIZE, 28, 28, 1)
        inputs = tf.expand_dims(inputs, axis=-1)
        outputs = snt.nets.ConvNet2D(
            output_channels=[64, 32],  # Two Conv layers
            kernel_shapes=[5, 5],
            strides=[2, 2],
            paddings=[snt.SAME],
            # By default final layer activation is disabled.
            activate_final=True,
            name='convolutional_module'
        )(inputs)
        outputs = snt.BatchFlatten()(outputs)  # Input layer for FC network
        outputs = snt.nets.MLP(  # Fully Connected layer
            output_sizes=[64, 10],
            name='fully_connected_module'
        )(outputs)

        return outputs


def get_model(model_name='mlp'):
    """ Helper function to make model configurable """

    if model_name == 'mlp':
        return FMNISTMLPClassifier()
    if model_name == 'conv':
        return FMNISTConvClassifier()
    raise Exception('Invalid Model')


def train(model_name, batch_size=1000, epoch=5):
    """ Training the model """

    train_data, test_data = get_fashion_MNIST_data()

    train_images, train_labels = train_data
    test_images, test_labels = test_data

    """ Now that we're going to start building our DataFlow graph, its a good 
        practice to reset the graph before starting to add ops.
    """
    tf.reset_default_graph()

    """ Since the dataset is huge, directly creating the `tf.data.Dataset` object from 
        a tensor would result in a constant op getting added to the graph. Since constant 
        ops store data in-graph, the size of the graph blows us. To avoid this we need 
        to use a placeholder.
    """
    # Training dataset placeholders
    train_images_op = tf.placeholder(
        shape=train_images.shape, dtype=tf.float32, name='train_images_ph')
    train_labels_op = tf.placeholder(
        shape=train_labels.shape, dtype=tf.int64, name='train_labels_ph')

    # Test dataset placeholders
    test_images_op = tf.placeholder(
        shape=test_images.shape, dtype=tf.float32, name='test_images_ph')
    test_labels_op = tf.placeholder(
        shape=test_labels.shape, dtype=tf.int64, name='test_labels_ph')

    # placeholder for the batch size, we need it to change the
    # batch size when we switch between datasets.
    batch_size_op = tf.placeholder(dtype=tf.int64)

    """ Create the input pipeline from the training and test data placeholders.
        1. Get data from data placeholder
        2. Shuffle the data
        3. Batch it with size batch_size_op
    """
    train_dataset_op = tf.data.Dataset.from_tensor_slices(
        (train_images_op, train_labels_op))
    train_dataset_op = train_dataset_op.shuffle(buffer_size=10000)
    train_dataset_op = train_dataset_op.batch(batch_size_op)

    test_dataset_op = tf.data.Dataset.from_tensor_slices(
        (test_images_op, test_labels_op))
    test_dataset_op = test_dataset_op.shuffle(buffer_size=10000)
    test_dataset_op = test_dataset_op.batch(batch_size_op)

    """ Create a reinitializable iterator. This type of iterator can be 
        initialized with any dataset that contains records with similar type 
        and shape. Since the records in the training and test dataset share 
        similar type and shape we can use this iterator to swap them.
    """
    iterator_op = tf.data.Iterator.from_structure(
        train_dataset_op.output_types,
        train_dataset_op.output_shapes)
    next_batch_images_op, next_batch_labels_op = iterator_op.get_next()

    training_init_op = iterator_op.make_initializer(train_dataset_op)
    testing_init_op = iterator_op.make_initializer(test_dataset_op)

    # Obtain the desired model to be trained from a configuration parameter.
    model = get_model(model_name)

    # Step 4: Setup the training apparatus
    prediction_op = model(next_batch_images_op)  # Forward pass Op
    loss_op = tf.losses.sparse_softmax_cross_entropy(
        next_batch_labels_op, prediction_op)  # Loss Op
    optimizer = tf.train.AdamOptimizer()  # Optimizer Op
    sgd_step = optimizer.minimize(loss_op)  # Gradient Descent step Op.

    """ Op to evaluate training and test accuracy every epochs. 
        `tf.metrics.accuracy` returns two ops, `acc_op` which simple perform a 
        calculation to give the current accuracy, and `acc_update_op` which 
        actually perform the evaluation and keeps tracks of the counts needed 
        to evaluate accuracy.
    """
    acc_op, acc_update_op = tf.metrics.accuracy(
        labels=next_batch_labels_op,
        predictions=tf.argmax(prediction_op, 1),
        name='accuracy_metric'
    )

    """ Since we want to evaluate both test and training accuracy we need to 
        make sure that the count variables in the accuracy op is reset after 
        each evaluation. In order to do that we need to create an initializer 
        just for the variables associated with the accuracy op. When this 
        initializer is called after (or before) every evaluation we can ensure 
        that the count variables are reset.
    """
    # Get initializer for accuracy vars to reset them after each epoch.
    accuracy_running_vars = tf.get_collection(
        tf.GraphKeys.LOCAL_VARIABLES, scope="accuracy_metric")
    accuracy_vars_initializer = tf.variables_initializer(
        var_list=accuracy_running_vars)

    # Pre-defined Feed dicts to sent data to placeholders.
    train_feed_dict = {train_images_op: train_images,
                       train_labels_op: train_labels, batch_size_op: batch_size}
    train_eval_feed_dict = {train_images_op: train_images,
                            train_labels_op: train_labels,
                            batch_size_op: len(train_labels)}
    test_feed_dict = {test_images_op: test_images,
                      test_labels_op: test_labels,
                      batch_size_op: len(test_labels)}

    # Step 5: Train
    with tf.Session() as sess:
        # Initialize local and global variables
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        # Epochs
        for idx in range(epoch):
            start = time.time()  # Taking time to track training duration for epoch

            # Initialize accuracy count variable for evaluating training accuracy
            sess.run(accuracy_vars_initializer)

            # Initalize the datset iterator with the training dataset.
            sess.run(training_init_op, feed_dict=train_feed_dict)

            # Loop until the SGD is performed over the entire dataset.
            while True:
                try:
                    # After an SGD step perform training accuracy evaluation for the current batch
                    sess.run([sgd_step, acc_update_op],
                             feed_dict=train_feed_dict)
                except tf.errors.OutOfRangeError:  # No more data in the input pipeline
                    break
            # Calculate training time for epoch.
            train_time = time.time()-start

            print("Epoch {:d} ::: Training Time: {:.2f}s,".format(
                idx+1, train_time), end=' ')

            # Evaluate the training loss with the current model
            sess.run(training_init_op, feed_dict=train_eval_feed_dict)
            print("Training Loss: {:.5f},".format(
                sess.run(loss_op, feed_dict=train_eval_feed_dict)), end=' ')

            print("Training Accuracy: {:.5f},".format(
                sess.run(acc_op)), end=' ')

            # Initialize accuracy count variable for evaluating test accuracy
            sess.run(accuracy_vars_initializer)

            # Initialize the dataset iterator with the test dataset.
            sess.run(testing_init_op, feed_dict=test_feed_dict)

            # Perform accuracy evaluation for the entire test dataset in one go.
            sess.run(acc_update_op, feed_dict=test_feed_dict)
            print("Test Accuracy: {:.5f}".format(sess.run(acc_op)))


train('mlp', batch_size=200, epoch=10)
