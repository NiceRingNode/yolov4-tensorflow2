import xml.etree.ElementTree as ET
from os import getcwd # 返回当前工作目录,get current work directory

'''
他这个生成的规则是4+1，4个坐标(xmin,ymin,xmax,ymax,class_id),然后5个5个这样排列，
他先设定了20个类，20个类以外的类不记录，比如000162，他有tvmonitor,person,hand,foot,head,
但是因为hand,foot,head不在20个类里面，所以他只记录tvmonitor,person的信息
'''

sets = [('2007','train'),('2007','val'),('2007','test')]
#sets = [('2012','train'),('2012','val'),('2012','test')]
classes = ["aeroplane","bicycle","bird","boat","bottle","bus","car","cat",
           "chair","cow","diningtable","dog","horse","motorbike","person",
           "pottedplant","sheep","sofa","train","tvmonitor"]
# 这里的类别是voc的类，如果是自己的数据集需要自己写classes
'''
sets = [('2012','train'),('2012','val'),('2012','test')]
classes = ['person','bicycle','car','motorbike','aeroplane','bus','train',
        'truck','boat','traffic light','fire hydrant','stop sign','parking meter',
        'bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra',
        'giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis',
        'snowboard','sports ball','kite','baseball bat','baseball glove','skateboard',
        'surfboard','tennis racket','bottle','wine glass','cup','fork','knife',
        'spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot',
        'hot dog','pizza','donut','cake','chair','sofa','pottedplant','bed',
        'diningtable','toilet','tvmonitor','laptop','mouse','remote','keyboard',
        'cell phone','microwave','oven','toaster','sink','refrigerator','book',
        'clock','vase','scissors','teddy bear','hair drier','toothbrush']
'''        
def convert_annotations(year,image_id,output_file):
    with open('./VOCdevkit/VOC%s/Annotations/%s.xml' % (year,image_id),
              encoding='utf-8') as input_file:
        tree = ET.parse(input_file)
        root = tree.getroot()
        for obj in root.iter('object'):
            difficult = 0
            if obj.find('difficult') != None:
            # 这个是VOC数据集的一个特点，会标出是不是困难样本，如果是difficult就是1
                difficult = obj.find('difficult').text
            class_name = obj.find('name').text # 这个是类名，比如horse
            if class_name not in classes or int(difficult) == 1: continue
            class_id = classes.index(class_name)

            xmlbox = obj.find('bndbox')
            # 一张图里面可能有很多个object所以会有很多个boundingboxes，
            # 但是这些只是坐标，在图片上并没把框标出来，(xmin,ymin,xmax,ymax)
            bboxes = (int(xmlbox.find('xmin').text)),\
                     (int(xmlbox.find('ymin').text)),\
                     (int(xmlbox.find('xmax').text)),\
                     (int(xmlbox.find('ymax').text))
            output_file.write(' ' + ','.join([str(each) for each in bboxes]) + ','
                            + str(class_id))

working_directory = ''
for year,image_set in sets:
    image_ids = open('./VOCdevkit/VOC%s/ImageSets/Main/my_%s.txt' %
        (year,image_set)).read().strip().split()
        # 这个就是在Main那里的那些train和val的txt，我们在voc2yolov4那里
        # 叫做my_xx.txt，这里要加个my，因为上一个文件的val和test是没数据的，所以现在也没有
    with open('%s_%s.txt' % (year,image_set),'w') as output_file:
        # 这个的文件名就比如是VOC2007_train.txt
        for image_id in image_ids:
            output_file.write('./VOCdevkit/VOC%s/JPEGImages/%s.jpg' %
                            (year,image_id))
            convert_annotations(year,image_id,output_file)
            output_file.write('\n')