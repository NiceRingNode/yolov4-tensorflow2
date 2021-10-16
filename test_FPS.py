import os,time
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import backend as K
from utils import letterbox_image
from application import yolov4_application

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

'''
该FPS测试不包括前处理（归一化与resize部分）、绘图。
包括的内容为：网络推理、得分门限筛选、非极大抑制。
使用'img/street.jpg'图片进行测试，该测试方法参考库https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch

video.py里面测试的FPS会低于该FPS，因为摄像头的读取频率有限，
而且处理过程包含了前处理和绘图部分。
'''
class yolov4_fps(yolov4_application):
    def __init__(self):
        super().__init__()

    def get_fps(self,image,test_interval):
        if self.letterbox_image:
            boxed_image = letterbox_image(image,
                (self.image_shape[1],self.image_shape[0]))
        else:
            boxed_image = image.convert('RGB')
            boxed_image = boxed_image.resize(
                (self.image_shape[1],self.image_shape[0]),Image.BICUBIC)
        #color_image = boxed_image
        gray = boxed_image.convert('L')
        boxed_image = gray.point(lambda x:255 if x > 128 else 0)
        image_data = np.array(boxed_image,dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data,axis=0) # 增加batch_size维度

        input_image_shape = np.expand_dims(np.array([image.size[1],image.size[0]],
            dtype='float32'),axis=0)
        eval_coords,eval_scores,eval_classes = self.get_prediction(image_data,
            input_image_shape) # eval_coords:(valid_num,4),(ymin,xmin,ymax,xmax)
        
        start = time.time()
        for _ in range(test_interval):
            input_image_shape = np.expand_dims(np.array([image.size[1],
                image.size[0]],dtype='float32'),axis=0)
            eval_coords,eval_scores,eval_classes = self.get_prediction(image_data,
                input_image_shape) # eval_coords:(valid_num,4),(ymin,xmin,ymax,xmax)
        end = time.time()
        detect_time = (end - start) / test_interval

        return detect_time

def main():
    fps_test = yolov4_fps()
    test_interval = 100 
    image = Image.open('./img/street.jpg')
    detect_time = fps_test.get_fps(image,test_interval)
    print(str(detect_time) + ' seconds, ' + str(1 / detect_time) + 'fps')

if __name__ == '__main__':
    main()
