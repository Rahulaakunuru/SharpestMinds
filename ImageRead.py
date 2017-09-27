import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import scipy.io as sio
import cv2
import os
import numpy as np
import pickle
import random
import math
from sklearn.model_selection import train_test_split


def getDataSet():
    max_width = float('-Inf')
    max_height = float('-Inf')
    dataset = list()
    cats = np.array(os.listdir('Dataset/Images'))
    df_with_dummies = pd.get_dummies(cats)
    for cat in cats:
        cat_list = os.listdir('Dataset/Images/'+cat)
        for cat_image in cat_list:
            img = cv2.imread('Dataset/Images/'+cat+'/'+cat_image,0)
            print 'Dataset/Images/'+cat+'/'+cat_image
            img_height, img_width = img.shape
            max_width = max(img_width, max_width)
            max_height = max(img_height, max_height)
            file_number = cat_image.split('_')[1].split('.')[0]
            box = sio.loadmat('Dataset/Annotations/'+cat+'/annotation_'+file_number)
            one_hot = np.array(df_with_dummies[cat].values)
            dataset.append([img,np.array(box['box_coord']).reshape(4),one_hot])
    random.shuffle(dataset)
    images = resizeImages(zip(*dataset)[0],max_width, max_height)
    bounding_box = zip(*dataset)[1]
    labels = zip(*dataset)[2]
    return (images, bounding_box, labels, max_width, max_height)


def resizeImages(images, width, height):
    resizedImages = []
    for i in range(len(images)):
        image = images[i]
        img_height, img_width = image.shape
        if width > img_width:
            width_padding = np.zeros((width - img_width) * img_height)
            width_padding = width_padding.reshape(img_height, (width - img_width))
            image = np.append(image, width_padding, 1)
        if height > img_height:
            height_padding = np.zeros((height-img_height)*width)
            height_padding = height_padding.reshape((height-img_height), width)
            image = np.append(image, height_padding, 0)
        image = image.reshape(300,300,1)
        resizedImages.append(image)
    return resizedImages


def getNextBatch(dataset, bounding_box, labels, offset, batch_size):
    if(offset > len(dataset)):
        return None
    else:
        end_index = min(len(dataset),offset+batch_size)
        return (dataset[offset:offset+batch_size], bounding_box[offset:offset+batch_size], labels[offset:offset+batch_size])


def drawImageWithBox(image, bounding_box = None):
    fig, ax = plt.subplots(1)
    ax.imshow(image,cmap='gray')
    if bounding_box != None:
        rect = patches.Rectangle((bounding_box[2], bounding_box[0]), bounding_box[3] - bounding_box[2], bounding_box[1] - bounding_box[0], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()


def getTrainTestData(dataset, bounding_box, labels, test_size=0.4):
    index = np.random.permutation(len(dataset))
    train_index = index[:int(len(index) - len(index)*test_size)]
    test_index = index[int(len(index) - len(index)*test_size):]
    x_train = [dataset[i] for i in range(len(dataset)) if i in train_index]
    x_test = [dataset[i] for i in range(len(dataset)) if i in test_index]
    box_train = [bounding_box[i] for i in range(len(bounding_box)) if i in train_index]
    box_test = [bounding_box[i] for i in range(len(bounding_box)) if i in test_index]
    label_train = [labels[i] for i in range(len(labels)) if i in train_index]
    label_test = [labels[i] for i in range(len(labels)) if i in test_index]
    return x_train, x_test, box_train, box_test, label_train, label_test