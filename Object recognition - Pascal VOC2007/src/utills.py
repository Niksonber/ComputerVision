import tensorflow as tf
import numpy as np
import sys

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path )
import cv2
import random

NUM_CLASSES = 20
shape = (150,150)
input_shape = (150,150,1)


classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow',
           'diningtable', 'dog', 'horse', 'motorbike', 'person',
           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

# response: 0- names, 1- response
def read_file_names_or_responses(path, respose):
    with tf.gfile.GFile(path) as f:
        lines = f.readlines()
    #return list(map(lambda x: x.strip().split()[0], lines))
    return [line.strip().split()[respose] for line in lines]

def read_responses(path, set_name):
    y = []
    for name in classes:
        y.append(read_file_names_or_responses(path+name+'_'+set_name+'.txt', 1))
    y = np.array(y, dtype = np.float32)
    return y.transpose()

def read_imgs(path, names):
    imgs = []
    for indx, name in enumerate(names):
        if indx % 100 == 0:
            print('image %d of %d' % (indx, len(names)))
        img = cv2.imread(path+name+'.jpg', cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, shape)
        imgs.append(img)
    return np.array(imgs)

def load_data(dir='data/VOC2007/', percent_of_train = .9):
    names_examples = read_file_names_or_responses(dir + '/ImageSets/Main/trainval.txt',0)
    #names_train = read_file_names_or_responses(dir + 'ImageSets/Main/train.txt',0)
    #names_val = read_file_names_or_responses(dir + 'ImageSets/Main/val.txt',0)

    print('Reading dataset.')
    x = read_imgs(dir+'JPEGImages/', names_examples)
    y = read_responses(dir+'ImageSets/Main/', 'trainval')

    random.seed(10)
    mask = np.ones(len(names_examples), dtype=bool)
    bound = int(percent_of_train*len(names_examples))
    mask[bound:] = False
    random.shuffle(mask)

    x_train = x[mask]
    y_train = y[mask]

    x_val = x[~mask]
    y_val = y[~mask]

    '''
    x_train = read_imgs(dir+'JPEGImages/', names_train)
    y_train = read_responses(dir+'ImageSets/Main/', 'train')

    print('Reading test data')
    x_val = read_imgs(dir+'JPEGImages/', names_val)
    y_val = read_responses(dir+'ImageSets/Main/', 'val')
    '''
    return (x_train, y_train),(x_val, y_val)
