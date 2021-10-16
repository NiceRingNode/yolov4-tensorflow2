import tensorflow as tf
from PIL import Image
from application import yolov4_application
import argparse

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

yolov4 = yolov4_application()

parser = argparse.ArgumentParser()
parser.add_argument('--img_root',default='./img/street.jpg')
opt = parser.parse_args()

try:
    image = Image.open(opt.img_root)
except:
    print('Open failed! Try again!')
    #continue
else:
    #pred_image = yolov4.draw_bounding_boxes(image)
    pred_image = yolov4.detect_image(image)
    pred_image.show()