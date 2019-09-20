#! /usr/bin/python3

from data_loader import *
from dnn import Classifier
from layers import DenseLayer, SoftmaxCrossEntropyLayer
from activations import ReLUActivation
from cost import softmax_cross_entropy
from utils import compute_classification_accuracy
import pylab
import warnings
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-d", "--dir", dest="datadir", default="../datasets/mnist/",
                    help="path to directory containing data")
parser.add_argument("-s", "--hidden-layer-sizes",
                    dest="sizes", default="128",
                    help="sizes of hidden layers separated by commas, e.g. '128,256'")
parser.add_argument("-l", "--learning-rate",
                    dest="learning_rate", default="0.05",
                    help="learning rate")
parser.add_argument("-m", "--max-iter",
                    dest="max_iter", default="50",
                    help="maximum number of iterations")
parser.add_argument("-b", "--batch-size",
                    dest="batch_size", default="600",
                    help="batch size")

args = parser.parse_args()

datadir = args.datadir
sizes = [int(s) for s in args.sizes.split(",")]
learning_rate = float(args.learning_rate)
max_iter = int(args.max_iter)
batch_size = int(args.batch_size)

# hide GLibc warning caused by matplotlib/pylab
warnings.simplefilter("ignore")

# grad_check_classifier = Classifier([
#     DenseLayer(4, 6, ReLUActivation()),
#     SoftmaxCrossEntropyLayer(6, 4)
# ], softmax_cross_entropy)
# grad_check_classifier.grad_check(np.array([100, 130, 50, 300]), np.array([0, 1, 0, 0]))


training_images, training_labels = loadMNIST("train", datadir)
test_images, test_labels = loadMNIST("t10k", datadir)

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

np.random.seed(1345134)

# training_images = training_images[0:1]
# training_labels = training_labels[0:1]

# print (training_images.shape)
# print (training_labels.shape)
# print (test_images.shape)
# print (test_labels.shape)

layers = [DenseLayer(training_images.shape[1], sizes[0], ReLUActivation())]
i = 1
while i < len(sizes):
    layers.append(DenseLayer(sizes[i-1], sizes[i], ReLUActivation()))
    i += 1
layers.append(SoftmaxCrossEntropyLayer(sizes[i-1], training_labels.shape[1]))

classifier = Classifier(layers, softmax_cross_entropy)
classifier.train(training_images, training_labels,
    max_iter=max_iter,
    learning_rate=learning_rate,
    target_acc=0.99,
    batch_size=batch_size)

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
