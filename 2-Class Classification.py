##################################################################################################################################################################
#  In this project i have implemented the Logistic Regression, and the optimization is done using Gradient Descent, Momentum and Nesterov's Acelerated Gradient  #
##################################################################################################################################################################

from sklearn.datasets import make_blobs
import pickle
import matplotlib.pyplot as plt
import numpy as np
import math
from timeit import default_timer as timer

# Sigmoid Function
def predict(weight_vector,datapoint):
    return 1/(1+np.exp(weight_vector.dot(datapoint)))

# Negative Log likelihood = -(1/m)*Sigma(ylog(p(y=1|x,w)) + (1-y)log(p(y=0|x,w)))
#Gradient = Sigma(x(y-p(y=1|x,w)))
def calculateLogLossAndGradient(weight_vector,data_points,labels):
    log_loss = 0
    gradient = 0
    for i in range(len(labels)):
        h = predict(weight_vector,data_points[i])
        if h == 1:
            h = 0.99999999 # Addresses the log 0 case.
        elif h == 0:
            h = 1 - 0.99999999
        log_loss = log_loss + labels[i] * math.log(h) + (1 - labels[i]) * math.log(1 - h)
        gradient = gradient + data_points[i].dot(labels[i] - h)
    return log_loss,gradient

#Plots the positive and negative class points
def plotPoints(dataset):
    # Divide the dataset to positive and negative class for visualization
    positive_index = [i for (i, x) in enumerate(dataset[1]) if x == 1]
    negative_index = [i for (i, x) in enumerate(dataset[1]) if x == 0]

    positive_class_points = dataset[0][positive_index]
    negative_class_points = dataset[0][negative_index]

    plt.plot(positive_class_points, 'bo')
    plt.plot(negative_class_points, 'ro')
    plt.show()

#Calculated the accuracy of the classifier
def claculateAccuracy(weight_vector,test_points,test_labels):
    count = 0
    for i in range(len(test_labels)):
        prediction = predict(weight_vector,test_points[i])
        prediction = 1 if prediction > 0.5 else 0
        if prediction == test_labels[i]:
            count += 1
    return count, count*100/len(test_labels)

def printPerformance(loss_function,algorithm,start,end,loss_list):
    print 'Time took to complete the task using {} loss function and {} is {}'.format(loss_function, algorithm, end - start)
    plt.plot(loss_list)
    plt.show()

def gradientDescent(weightVector, train_points, train_labels,learning_rate = 0, reg_lambda = 0, momentum_step = 0):
    v = np.zeros(np.shape(weightVector))
    previous_loss = float('inf')
    loss_list = []
    count = 0
    while True:
        count += 1
        (loss, gradient) = calculateLogLossAndGradient(weightVector, train_points, train_labels)
        loss_list.append(0 - loss)
        v = momentum_step * v - learning_rate * gradient
        weightVector = weightVector + v + reg_lambda * weightVector
        delta = abs(previous_loss - loss)
        if delta < 1:
            break
        previous_loss = loss
        if (count % 250 == 0):
            print 'Iter number {}. Cost is {}'.format(count, loss)
    return weightVector,loss_list

def NAGGradientDescent(weightVector, train_points, train_labels,learning_rate = 0, reg_lambda = 0, momentum_step = 0):
    # Nestorov's Accelerated Gradient module
    v = np.zeros(np.shape(weightVector))
    previous_loss = float('inf')
    loss_list = []
    count = 0
    while True:
        count += 1
        v_prev = v
        (loss, gradient) = calculateLogLossAndGradient(weightVector, train_points, train_labels)
        loss_list.append(0 - loss)
        v = momentum_step * v - learning_rate * gradient
        weightVector = weightVector + v + reg_lambda * weightVector + momentum_step * (v - v_prev)
        delta = abs(previous_loss - loss)
        if delta < 10 ** -6:
            break
        previous_loss = loss
        if (count % 250 == 0):
            print 'Iter number {}. Cost is {}'.format(count, loss)
        return weightVector, loss_list

dataset = make_blobs(n_samples=100, n_features=2)
#dataset_file = open('dataset.pickle','wb')
#pickle.dump(dataset,dataset_file)
#dataset_file = open('dataset.pickle','rb')
#dataset = pickle.load (dataset_file)
#dataset_file.close()

plotPoints(dataset)

#Divide Data to Training and Testing set
shuffle = np.random.permutation(len(dataset[0]))

train_points = dataset[0][shuffle[:int(.8*len(shuffle))]]
train_labels = dataset[1][shuffle[:int(.8*len(shuffle))]]

test_points = dataset[0][shuffle[int(.8*len(shuffle)):]]
test_labels = dataset[1][shuffle[int(.8*len(shuffle)):]]

#Add bias term
train_points = [np.append([1],train_points[i]) for i in range(len(train_points))]
test_points = [np.append([1],test_points[i]) for i in range(len(test_points))]

loss_list = []

learning_rate = 0.01
reg_lambda = 0.01
momentum_step = 0.01

# Gradient Descent
start = timer()
#Random Weighs initilization
weightVector = np.random.rand(3)
weightVector,loss_list = gradientDescent(weightVector,train_points,train_labels,learning_rate=learning_rate,reg_lambda=reg_lambda)
end = timer()
printPerformance('Negative Log Likelihood','L2 Regularization',start,end,loss_list)
print 'The accuracy for Gradient Descent is {}'.format(claculateAccuracy(weightVector, test_points, test_labels))

# Gradient Descent with Momentum Update
start = timer()
#Random Weighs initilization
weightVector = np.random.rand(3)
weightVector,loss_list = gradientDescent(weightVector, train_points, train_labels, learning_rate=learning_rate, reg_lambda=reg_lambda, momentum_step=momentum_step)
end = timer()
printPerformance('Negative Log Likelihood','Momentum Update',start,end,loss_list)
print 'The accuracy for Gradient Descent using Momentum Update is {}'.format(claculateAccuracy(weightVector, test_points, test_labels))

# Gradient Descent with NAG Update
start = timer()
#Random Weighs initilization
weightVector = np.random.rand(3)
weightVector,loss_list = NAGGradientDescent(weightVector, train_points, train_labels, learning_rate=learning_rate, reg_lambda=reg_lambda, momentum_step=momentum_step)
end = timer()
printPerformance('Negative Log Likelihood',"Nesterov's Accelerated Gradient",start,end,loss_list)
print "The accuracy for Gradient Descent using Nesterov's Accelerated Gradient is {}".format(claculateAccuracy(weightVector, test_points, test_labels))