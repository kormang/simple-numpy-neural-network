#! /usr/bin/python3

from data_loader import *
from dnn import CrossEntropyClassifier
from layers import DenseLayer, SoftmaxCrossEntropyLayer
from activations import ReLUActivation
from cost import softmax_cross_entropy
from utils import compute_classification_accuracy
import pylab


# grad_check_classifier = CrossEntropyClassifier([
#     DenseLayer(4, 6, ReLUActivation()),
#     SoftmaxCrossEntropyLayer(6, 4)
# ], softmax_cross_entropy)
# grad_check_classifier.grad_check(np.array([100, 130, 50, 300]), np.array([0, 1, 0, 0]))


training_images, training_labels = loadMNIST("train", "../datasets/mnist/")
test_images, test_labels = loadMNIST("t10k", "../datasets/mnist/")

training_labels = to_hot_encoding(training_labels)
test_labels = to_hot_encoding(test_labels)

training_images = training_images.reshape((
    training_images.shape[0],
    training_images.shape[1] * training_images.shape[2]))

test_images = test_images.reshape((
    test_images.shape[0],
    test_images.shape[1] * test_images.shape[2]))

training_images *= 1.0/255.0
test_images *= 1.0/255.0

# print (training_images.shape)
# print (training_labels.shape)
# print (test_images.shape)
# print (test_labels.shape)

classifier = CrossEntropyClassifier([
    DenseLayer(training_images.shape[1], 128, ReLUActivation()),
    SoftmaxCrossEntropyLayer(128, training_labels.shape[1])
], softmax_cross_entropy)
classifier.train(training_images, training_labels, max_iter=10, alpha=0.05, target_cost=0.1)

pylab.figure()
pylab.plot(classifier.cost_history)
pylab.title("Learning curve")
pylab.savefig("learning_curve.png")

predictions = classifier.predict(training_images)
accuracy = compute_classification_accuracy(predictions, training_labels)
print("Training accuracy {0:.2f}%".format(100 * accuracy))

predictions = classifier.predict(test_images)
accuracy = compute_classification_accuracy(predictions, test_labels)
print("Test accuracy {0:.2f}%".format(100 * accuracy))

print ("Done")
