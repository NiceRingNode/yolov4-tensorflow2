import time
import numpy as np
import cv2 as cv
import tensorflow as tf
from PIL import Image
from application import yolov4_application

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

yolov4 = yolov4_application()
#capture = cv.VideoCapture('./video/road.mp4') # 0是摄像头
capture = cv.VideoCapture(0)
fourcc = cv.VideoWriter_fourcc(*'XVID')
saveroot = './video/output.avi'
writer = cv.VideoWriter(saveroot,fourcc,15.0,(640,480))
fps = 0.0

while True:
    ret,frame = capture.read()
    # 格式转变，BGR2RGB
    if ret == False:
        break
    frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    frame = Image.fromarray(np.uint8(frame))

    start = time.time()
    frame = np.array(yolov4.detect_image(frame))
    frame = cv.cvtColor(frame,cv.COLOR_RGB2BGR) # RGB2BGR满足opencv显示格式

    fps = 1. / (time.time() - start)
    print(f'fps = {fps}')
    frame = cv.putText(frame,'fps = %.2f' % fps,(0,40),cv.FONT_HERSHEY_SIMPLEX,
        1,(0,255,0),2)
    #frame = cv.cvtColor(frame,cv.COLOR_RGB2BGR)
    cv.imshow('video',frame)
    writer.write(frame)
    c = cv.waitKey(30) & 0xff
    if c == 27:
        capture.release()
        break    
