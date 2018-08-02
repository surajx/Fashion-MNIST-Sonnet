# Import necessary libraries
import numpy as np
import sonnet as snt
import tensorflow as tf

import time  # To time each epoch

# Needed to download Fashion-MNIST (FMNIST) dataset without much hassle.
from tensorflow.keras import datasets


# Step 1: Get the Data
def get_fashion_MNIST_data():
    """ Download FMNIST Dataset. """

    train_data, test_data = datasets.fashion_mnist.load_data()

    # Pre-process the data
    train_data = (normalize_fmnist_images(train_data[0]), train_data[1])
    test_data = (normalize_fmnist_images(test_data[0]), test_data[1])

    return train_data, test_data


# Step 2: Preprocess Dataset
def normalize_fmnist_images(images):
    """ Perform max-min normalization of images in FMNIST. """

    return images / 255.0


# Step 3: Build the model

# Multi-Layer Perceptron
class FMNISTMLPClassifier(snt.AbstractModule):
    """ Model for Classifying FMNIST dataset based on a Multi-Layer Perceptron """

    def __init__(self, name='fmnist_mlp_classifier'):
        """ Initialize the MLP Calssifier """
        super().__init__(name=name)

    def _build(self, inputs):
        """ Builds the model """
        # Stack necessary modules to build model.
        outputs = snt.BatchFlatten()(inputs)  # Input layer with 784 neurons
        outputs = snt.nets.MLP(  # MLP module from Sonnet
            output_sizes=[128, 10],
            name='fmnist_mlp'
        )(outputs)

        return outputs


# Convolutional Neural Network
class FMNISTConvClassifier(snt.AbstractModule):
    def __init__(self, name='fmnist_conv_classifier'):
        super().__init__(name=name)

    def _build(self, inputs):
        inputs = tf.expand_dims(inputs, axis=-1)  # (BATCH_SIZE, 28, 28, 1)
        outputs = snt.nets.ConvNet2D(
            output_channels=[64, 32],  # 3 Conv layers
            kernel_shapes=[5, 5],
            strides=[2, 2],
            paddings=[snt.SAME],
            name='convolutional_module'
        )(inputs)
        outputs = snt.BatchFlatten()(outputs)
        outputs = tf.nn.relu(outputs)
        outputs = snt.nets.MLP(
            output_sizes=[64, 10],
            name='fully_connected_module'
        )(outputs)

        return outputs


def get_model(model_name='mlp'):
    """ Helper function to make model configurablel """

    if model_name == 'mlp':
        return FMNISTMLPClassifier()
    if model_name == 'conv':
        return FMNISTConvClassifier()
    raise Exception('Invalid Model')


def train(model_name, batch_size=1000, epoch=5):
    train_data, test_data = get_fashion_MNIST_data()

    train_images, train_labels = train_data
    test_images, test_labels = test_data

    tf.reset_default_graph()

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

    batch_size_op = tf.placeholder(dtype=tf.int64)

    # Create tf.Datasets from training as test data
    train_dataset_op = tf.data.Dataset.from_tensor_slices(
        (train_images_op, train_labels_op))
    train_dataset_op = train_dataset_op.shuffle(buffer_size=10000)
    train_dataset_op = train_dataset_op.batch(batch_size_op)

    test_dataset_op = tf.data.Dataset.from_tensor_slices(
        (test_images_op, test_labels_op))
    test_dataset_op = test_dataset_op.shuffle(buffer_size=10000)
    test_dataset_op = test_dataset_op.batch(batch_size_op)

    # Reinitializable iterator
    iterator_op = tf.data.Iterator.from_structure(
        train_dataset_op.output_types,
        train_dataset_op.output_shapes)
    next_batch_images_op, next_batch_labels_op = iterator_op.get_next()

    training_init_op = iterator_op.make_initializer(train_dataset_op)
    testing_init_op = iterator_op.make_initializer(test_dataset_op)

    # Get the un-trained model
    model = get_model(model_name)

    # Set-up model training apparatus
    prediction_op = model(next_batch_images_op)
    loss_op = tf.losses.sparse_softmax_cross_entropy(
        next_batch_labels_op, prediction_op)
    optimizer = tf.train.AdamOptimizer()
    sgd_step = optimizer.minimize(loss_op)  # Gradient descent step

    # Evaluate test accuracy every epochs
    acc_op, acc_update_op = tf.metrics.accuracy(
        labels=next_batch_labels_op,
        predictions=tf.argmax(prediction_op, 1),
        name='accuracy_metric'
    )

    # Get initializer for accuracy vars to reset them after each epoch.
    accuracy_running_vars = tf.get_collection(
        tf.GraphKeys.LOCAL_VARIABLES, scope="accuracy_metric")
    accuracy_vars_initializer = tf.variables_initializer(
        var_list=accuracy_running_vars)

    train_feed_dict = {train_images_op: train_images,
                       train_labels_op: train_labels, batch_size_op: batch_size}
    train_eval_feed_dict = {train_images_op: train_images,
                            train_labels_op: train_labels,
                            batch_size_op: len(train_labels)}
    test_feed_dict = {test_images_op: test_images,
                      test_labels_op: test_labels,
                      batch_size_op: len(test_labels)}
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        for idx in range(epoch):
            start = time.time()
            sess.run(accuracy_vars_initializer)
            sess.run(training_init_op, feed_dict=train_feed_dict)
            while True:
                try:
                    sess.run([sgd_step, acc_update_op],
                             feed_dict=train_feed_dict)
                except tf.errors.OutOfRangeError:
                    break
            train_time = time.time()-start

            print("Epoch {:d} ::: Training Time: {:.2f}s,".format(
                idx+1, train_time), end=' ')

            sess.run(training_init_op, feed_dict=train_eval_feed_dict)
            print("Training Loss: {:.5f},".format(
                sess.run(loss_op, feed_dict=train_eval_feed_dict)), end=' ')

            print("Training Accuracy: {:.5f},".format(
                sess.run(acc_op)), end=' ')

            sess.run(accuracy_vars_initializer)
            sess.run(testing_init_op, feed_dict=test_feed_dict)
            sess.run(acc_update_op, feed_dict=test_feed_dict)
            print("Test Accuracy: {:.5f}".format(sess.run(acc_op)))


train('mlp', batch_size=200, epoch=10)
