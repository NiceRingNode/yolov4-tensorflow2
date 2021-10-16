from yolov4 import yolov4_net
from tensorflow.keras import Input
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv
import tensorflow.keras.backend as K
import tensorflow as tf

#inputs = Input([416,416,3])
#model = yolov4_net(inputs,3,80)
#model.summary()
#for i,layer in enumerate(model.layers):
#    print(i,layer.name)


grid_shape = tf.constant([13,13])
num_anchors = 3
print([K.arange(0,stop=grid_shape[0])])
print(tf.shape([K.arange(0,stop=grid_shape[0])] * grid_shape[1] * num_anchors))
#print(tf.shape(tf.reshape([K.arange(0,stop=grid_shape[0])] * grid_shape[1] * num_anchors,
#    (num_anchors,grid_shape[0],grid_shape[1]))))
"""

grid_shape = tf.constant([13,13])
num_anchors = 3
print([K.arange(0,stop=grid_shape[0])])
print(tf.shape(tf.tile([K.arange(0,stop=grid_shape[0])],[grid_shape[1] * num_anchors,1])))
#print(tf.shape([K.arange(0,stop=grid_shape[0])] * grid_shape[1] * num_anchors))
print(tf.shape(tf.reshape(tf.tile([K.arange(0,stop=grid_shape[0])],[grid_shape[1] * num_anchors,1]),
    (num_anchors,grid_shape[0],grid_shape[1]))))
"""
x_origin = tf.cast(tf.expand_dims(tf.transpose(tf.reshape(tf.tile(
    [K.arange(0,stop=grid_shape[0])],[grid_shape[1] * num_anchors,1]),
    (num_anchors,grid_shape[0],grid_shape[1])),(1,2,0)),axis=0),dtype=tf.float32)
y_origin = tf.transpose(x_origin,(0,2,1,3))
#print(x_origin)
#print(y_origin)


'''
def get_anchors(file_path):
    with open(file_path) as f:
        anchors = f.readline() # 只有一行
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1,2)

line = './VOCdevkit/VOC2007/JPEGImages/000162.jpg 306,227,380,299,19 196,143,309,369,14'.split()
bboxes = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
#print(line)
print(bboxes.shape)
#print(bboxes[:,0:2][bboxes[:,0:2] > 0])
bboxes_w = bboxes[:,2] - bboxes[:,0]
bboxes_h = bboxes[:,3] - bboxes[:,1]
bboxes = bboxes[np.logical_and(bboxes_w > 416,bboxes_h < 416)]
#print(np.logical_and(bboxes_w > 416,bboxes_h < 416))
# 原理是这样的，首先bboxes_w和bboxes_h维度要一样，然后对每个一个单体进行逻辑与比较，比如
# bboxes_w[0] > 416 & bboxes_h[0] < 416,因为他有两个w和h，所以还有
# bboxes_w[0] > 416 & bboxes_h[0] < 416，就会有两个结果，[False,False]
# 然后就是与的语义，> 1的意思是所有bbox宽高都要大于1，不然就是废的
#print(bboxes)

def rand(a:float=0.0,b:float=1.0):
    # np.random.rand()产生一个服从0,1均匀分布的值，# 这样写跟原来没区别
    return np.random.rand() * (b - a) + a

w,h = 416,416
jitter = 0.3
#fig,ax = plt.subplots(3,3)
for i in range(9):
    new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
        #print(nh,nw)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)
        #print(nh,nw)

    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))
    #print(dx,dy)
    image = Image.open('./VOCdevkit/VOC2007/JPEGImages/000162.jpg')
    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', (w, h), (128,128,128))
    new_image.paste(image, (dx, dy))
    #plt.figure()
    #plt.imshow(new_image)
    #plt.show()

image = Image.open('./VOCdevkit/VOC2007/JPEGImages/000162.jpg')
print(np.array(image).shape)
image_hsv = cv.cvtColor(np.array(image,np.float32) / 255.,cv.COLOR_RGB2HSV)
print(image_hsv.shape)

input_shape = np.array((416,416),dtype='float32')
grid_shapes = [input_shape // {0:32,1:16,2:8}[layer]  for layer in range(3)]
print(grid_shapes)

import os
print(os.path.exists(path='./model_data/yolov4_weight.h5'))
'''

input_shape = tf.constant([512,512])
image_shape = tf.constant([720,1280])
print(tf.reduce_min(input_shape / image_shape))