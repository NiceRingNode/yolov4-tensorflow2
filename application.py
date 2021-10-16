import colorsys,copy,os
from timeit import default_timer as timer
import numpy as np
import tensorflow as tf
from PIL import Image,ImageDraw,ImageFont
from tensorflow.keras.layers import Input,Lambda
from tensorflow.keras.models import Model,load_model
from yolov4 import yolov4_net
from utils import letterbox_image
from evaluate import evaluate
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# 使用自己训练好的模型预测要改两个参数，model_path和classes_path

class yolov4_application:
    '''
        这个权重，太狗了，他根本就是已经训练好可以直接预测的权重！！！
    '''
    default_args = {
        #'model_path':'./model_data/yolov4_weight.h5', # 预训练权重
        #'model_path':'./logs/voc_with_cosine.h5',
        'model_path':'./model_data/yolov4_coco_weights.h5',
        'anchors_path':'./model_data/yolo_anchors.txt',
        'classes_path':'./model_data/coco_classes.txt',
        #'classes_path':'./model_data/coco_classes.txt',
        'score_threshold':0.5,
        'iou_threshold':0.3,
        'max_bboxes':100,
        'image_shape':(416,416), # 叫image_size可能会好一点，(h,w)
        'letterbox_image':False 
        # 该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #不用这个函数直接resize效果更好，letterbox是源码的名字，可能是说像一个信封
    }

    @classmethod
    def get_default_args(cls,n:str): # cls代表类自身，self代表对象实例自身
        if n in cls.default_args:
            return cls.default_args[n]
        else:
            raise AttributeError("Unrecognized attribute name '" + n + "'")

    def __init__(self,**kwargs):
        self.__dict__.update(self.default_args) # 这里使得参数变成了自己属性
        self.classes_names = self.get_classes()
        self.anchors = self.get_anchors()
        self.num_classes = len(self.classes_names)
        self.generate()

    def get_classes(self):
        classes_path = os.path.expanduser(self.classes_path) 
        # 一般用于linux，将~变成根目录
        with open(classes_path) as f:
            classes_names = f.readlines()
        classes_names = [c.strip() for c in classes_names]
        return classes_names

    def get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline() # 只有一行
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1,2)

    def generate(self): # 载入模型
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'),'model or weights must be a .h5 file'
        num_anchors = len(self.anchors)

        self.yolo_model = yolov4_net(Input(shape=(None,None,3)),num_anchors // 3, 
            self.num_classes)
        self.yolo_model.load_weights(self.model_path,by_name=True,
            skip_mismatch=True) 
        # 预训练权重
        print(f'{model_path} model,anchors and classes loaded')

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.classes_names),1.,1.) for x in range(
            len(self.classes_names))]
        self.colors = list(map(lambda x:colorsys.hsv_to_rgb(*x),hsv_tuples))
        self.colors = list(map(lambda x:(int(x[0] * 255),int(x[1] * 255),
            int(x[2] * 255)),self.colors))
        np.random.seed(100)
        np.random.shuffle(self.colors) # 打乱颜色
        np.random.seed(None)

        # 下面是构建预测模型，然后进行预测
        self.input_image_shape = Input([2,],batch_size=1) # 其实就是留个位置
        inputs = [*self.yolo_model.output,self.input_image_shape]
        outputs = Lambda(evaluate,output_shape=(1,),name='yolov4_evaluate',
            arguments={'anchors':self.anchors,'num_classes':self.num_classes,
            'image_shape':self.image_shape,'score_threshold':self.score_threshold,
            'eager':True,'max_bboxes':self.max_bboxes,'letterbox_image':
            self.letterbox_image})(inputs)
        # 同样的操作，将evaluate作为最后一层输出
        '''
            他这里的操作得益于tf.keras.models.Model的数据输入不是贪心的，
            而是够了就停，所以input_image_shape位置上的参数可以完美传到evaluate()
            里面去
        '''
        self.yolo_model = Model([self.yolo_model.input,self.input_image_shape],
            outputs)

    @tf.function
    def get_prediction(self,image_data,input_image_shape):
        # input_image_shape:(1330,1330,3)
        eval_coords,eval_scores,eval_classes = self.yolo_model([image_data,
            input_image_shape],training=False) # training是Model自带的方法
        return eval_coords,eval_scores,eval_classes
    
    def draw_bounding_boxes(self,image): # image是个PIL对象
        '''
        Args:
            y_output:从final_pred_one输出来的，
            dict,key:obj_class_on_img,value:[confs,xmin,ymin,xmax,ymax],
            也就是说，这张图像上的类的个数，然后每个类只有最终一个框，
            不理想状态下就一个类有多个框
            axes:我们是在一张图上画这些框，用的是matplotlib的axes
        '''
        if self.letterbox_image:
            bboxed_image = letterbox_image(image,(self.image_shape[1],
                self.image_shape[0]))
        else: # 就是在这里resize成(416,416)的
            bboxed_image = image.convert('RGB')
            bboxed_image = bboxed_image.resize((self.image_shape[1],
                self.image_shape[0]),Image.BICUBIC)
        image_data = np.array(bboxed_image,dtype='float32') / 255.
        image_data = np.expand_dims(image_data,axis=0)

        input_image_shape = np.expand_dims(np.array([image.size[1],image.size[0]], 
            dtype='float32'),axis=0) # (1,h,w),这是PIL的顺序
        # 这里获取了输入图片的大小，也就是(1330,1330)，模型是可以对不同大小的图片进行预测的
        # 虽然他训练时用的是(416,416)或(608,608)，但是训练后他有能力应对不同大小的图片
        # 所以会出现eval_coords有548和950的出现
        eval_coords,eval_scores,eval_classes = self.get_prediction(image_data,
            input_image_shape) # eval_coords:(valid_num,4),(ymin,xmin,ymax,xmax)
        
        print(f'found {len(eval_coords)} boxes for image')

        fig = plt.imshow(image)
        plt.axis('off')
        colors = ['b','g','r','m','c']
        for i,c in list(enumerate(eval_classes)):
            # list(enumerate(x)):[(0,0),(1,1),(2,2),...]长这样
            cur_class = self.classes_names[c]
            cur_class_coords = eval_coords[i]
            cur_class_score = eval_scores[i]
            ymin,xmin,ymax,xmax = cur_class_coords # 他y轴向下，虽然并不影响
            ymin -= 5
            xmin -= 5
            ymax += 5
            xmax += 5 # 下面4行是典型的边缘控制
            ymin = max(0,np.floor(ymin + 0.5).astype('int32'))
            xmin = max(0,np.floor(xmin + 0.5).astype('int32'))
            ymax = min(image.size[1],np.floor(ymax + 0.5).astype('int32'))
            xmax = min(image.size[0],np.floor(xmax + 0.5).astype('int32'))
            color = colors[int(c) % len(colors)]
            rect = plt.Rectangle(xy=(xmin,ymin),width=xmax - xmin,height=ymax - ymin,
                fill=False,edgecolor=color,linewidth=2)
            fig.axes.add_patch(rect)

            text_color = 'w' if color == 'k' else 'k'
            label = f'{cur_class} {cur_class_score:.2f}'
            label = label.encode('utf-8')
            fig.axes.text(rect.xy[0],rect.xy[1],str(label,'UTF-8'),
                va='bottom',ha='left',fontsize=8,color=text_color,
                bbox=dict(facecolor=color,lw=0))
            # bbox是background box，给标题增加外框，boxstyle：方框外形，
            # facecolor：背景颜色，edgecolor：边框线条颜色，edgewidth：边框线条大小
            # str(label,'UTF-8')搞掉了那个'b'
        plt.show()
        return image

    def detect_image(self,image):
        # image是个PIL Image对象
        # 可以增加灰条，不失真地进行resize，也可以直接resize识别
        if self.letterbox_image:
            bboxed_image = letterbox_image(image,(self.image_shape[1],
                self.image_shape[0]))
        else: # 就是在这里resize成(416,416)的
            bboxed_image = image.convert('RGB')
            bboxed_image = bboxed_image.resize((self.image_shape[1],
                self.image_shape[0]),Image.BICUBIC)
        image_data = np.array(bboxed_image,dtype='float32') / 255.
        image_data = np.expand_dims(image_data,axis=0)

        input_image_shape = np.expand_dims(np.array([image.size[1],image.size[0]], 
            dtype='float32'),axis=0) # (1,h,w),这是PIL的顺序
        # 这里获取了输入图片的大小，也就是(1330,1330)，模型是可以对不同大小的图片进行预测的
        # 虽然他训练时用的是(416,416)或(608,608)，但是训练后他有能力应对不同大小的图片
        # 所以会出现eval_coords有548和950的出现
        eval_coords,eval_scores,eval_classes = self.get_prediction(image_data,
            input_image_shape) # eval_coords:(valid_num,4),(ymin,xmin,ymax,xmax)
        
        print(f'found {len(eval_coords)} boxes for image')

        font = ImageFont.truetype(font='./font/simhei.ttf',size=np.floor(3e-2 * 
            image.size[1] + 0.5).astype('int32'))
        thickness = max((image.size[0] + image.size[1]) // 300,1)

        for i,c in list(enumerate(eval_classes)):
            # list(enumerate(x)):[(0,0),(1,1),(2,2),...]长这样
            cur_class = self.classes_names[c]
            cur_class_coords = eval_coords[i]
            cur_class_score = eval_scores[i]
            top,left,bottom,right = cur_class_coords # 他y轴向下，虽然并不影响
            top -= 5
            left -= 5
            bottom += 5
            right += 5 # 下面4行是典型的边缘控制
            top = max(0,np.floor(top + 0.5).astype('int32'))
            left = max(0,np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1],np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0],np.floor(right + 0.5).astype('int32'))

            # 画框
            label = f'{cur_class} {cur_class_score:.2f}'
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label,font) 
            # 返回用这个字体显示这么长的串需要的图像尺寸
            label = label.encode('utf-8')
            print(label,top,left,bottom,right)

            '''
                这个就是将label文字的纵坐标放到框的上面，如果top比label的h高，就
                减就可以了，原始位置就在顶部，如果比label的小，就是框的最上面在图片的顶部，
                就要将label画在框顶下部（包在里面），否则画在框外
            '''
            if top - label_size[1] >= 0:
                text_origin_coord = np.array([left,top - label_size[1]])
            else:
                text_origin_coord = np.array([left,top + 1])
            for j in range(thickness):
                draw.rectangle([left + j,top + j,right - j,bottom - j],
                    outline=self.colors[c])
                # 如果没有这个循环，画出来的框就会很细，这个循环就是反复画框，然后使线变粗
            draw.rectangle([tuple(text_origin_coord),tuple(text_origin_coord 
                + label_size)],fill=self.colors[c])
            draw.text(text_origin_coord,str(label,'UTF-8'),fill=(0,0,0),font=font)
            del draw
        return image
