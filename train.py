import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,TensorBoard
from tensorflow.keras.layers import Input,Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from loss import loss
from yolov4 import yolov4_net
from utils import get_random_data_with_Mosaic,get_random_data,ModelCheckpoint,\
    WarmupCosineDecayScheduler
import os,argparse

tf.config.experimental_run_functions_eagerly(True)

def get_gt_classes(file_path):
    with open(file_path) as f:
        class_names = f.readlines()
    class_names = [i.strip() for i in class_names]
    return class_names

def get_anchors(file_path):
    with open(file_path) as f:
        anchors = f.readline() # 只有一行
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1,2)

def data_generator(annotation_lines,batch_size,input_shape,anchors,num_classes,
    mosaic=False,random=True):
    '''
        :param annotation_lines:就是2007_train.txt文件里的所有标注
        :param batch_size:2
        :param input_shape:(416,416)
        :param anchors:就是yolo_anchors.txt的数据，9*2
        :param num_classes:20
    '''
    num_annotations = len(annotation_lines)
    i,flag = 0,True
    while True:
        img_data,bboxes_data,img,bboxes = [],[],None,None
        for each_one in range(batch_size):
            if i == 0: np.random.shuffle(annotation_lines)
            if mosaic:
                if flag and (i + 4) < num_annotations:
                    img,bboxes = get_random_data_with_Mosaic(annotation_lines
                            [i:i + 4], input_shape)
                    i = (i + 4) % num_annotations
                else:
                    img,bboxes = get_random_data(annotation_lines[i],
                            input_shape,random=random)
                    i = (i + 1) % num_annotations
                flag = bool(1 - flag)
            else:
                img,bboxes = get_random_data(annotation_lines[i],
                    input_shape,random=random)
                i = (i + 1) % num_annotations
            img_data.append(img)
            bboxes_data.append(bboxes)
        img_data = np.array(img_data) # (batch_size,input_w,input_h,c)
        bboxes_data = np.array(bboxes_data)
        y_true = preprocess_true_bboxes(bboxes_data,input_shape,anchors,num_classes)
        yield [img_data,*y_true],np.zeros(batch_size)
        # [img_data,*y_true]:[(batch_size,input_w,input_h,c),(batch_size,13,13,3,85),
        # (batch_size,26,26,3,85),(batch_size,52,52,3,85)]

def preprocess_true_bboxes(gt_bboxes,input_shape,anchors,num_classes):
    '''
        读入xml文件
        :param gt_bboxes:真实图片的bounding boxes数据，是在get_random_data
        里面对实际图片进行随机处理过之后产生的数据，框的相对位置是没变的但是别的地方
        可能会变了，(batch_size,num_bboxes,5),(xmin,ymin,xmax,ymax,class_id)
        其中num_bboxes是一张图片上真实的bboxes数量，因为可能不止一个
        :param input_shape:(416,416)
        :param anchors:就是yolo_anchors.txt的数据，9*2
        :param num_classes:20
        :return y_true:[(batch_size,13,13,3,85),(batch_size,26,26,3,85),
            (batch_size,52,52,3,85)]
    '''
    assert (gt_bboxes[:,:,4] < num_classes).all(), \
        'class id must be less than the number of classes'
    
    num_layers = len(anchors) // 3
    # 13x13的特征层对应的anchor是[142,110],[192,243],[459,401]
    # 26x26的特征层对应的anchor是[36,75],[76,55],[72,146]
    # 52x52的特征层对应的anchor是[12,16],[19,36],[40,28]
    anchors_mask = [[6,7,8],[3,4,5],[0,1,2]]
    gt_bboxes = np.array(gt_bboxes,dtype='float32')
    input_shape = np.array(input_shape,dtype='int32')
    # 也就是h,w = 416,416
    # 计算gt的宽高和中心坐标，并归一化
    gt_bboxes_xy = (gt_bboxes[:,:,:2] + gt_bboxes[:,:,2:4]) // 2
    gt_bboxes_wh = gt_bboxes[:,:,2:4] - gt_bboxes[:,:,:2]
    gt_bboxes[:,:,:2] = gt_bboxes_xy / input_shape[::-1]
    gt_bboxes[:,:,2:4] = gt_bboxes_wh / input_shape[::-1]

    batch_size = gt_bboxes.shape[0]
    grid_shapes = [input_shape // {0:32,1:16,2:8}[layer] \
        for layer in range(num_layers)]
    y_true = [np.zeros((batch_size,grid_shapes[layer][0],grid_shapes[layer][1],
        len(anchors_mask[layer]),5 + num_classes),dtype='float32') for layer in
        range(num_layers)]
    '''
        grid_shapes:[array([13.,13.],dtype=float32),array([26.,26.],
            dtype=float32),array([52.,52.],dtype=float32)]
        y_true:[(batch_size,13,13,3,85),(batch_size,26,26,3,85),
            (batch_size,52,52,3,85)]
    '''
    anchors = np.expand_dims(anchors,axis=0) # (9,2)->(1,9,2)
    anchors_maxes = anchors / 2.
    anchors_mins = -anchors_maxes
    valid_mask = gt_bboxes_wh[:,:,0] > 0 # 长宽大于0才有效

    for each in range(batch_size):
        valid_wh = gt_bboxes_wh[each,valid_mask[each]] # 拿到宽高大于0的有效bboxes
        if len(valid_wh) == 0: continue
        valid_wh = np.expand_dims(valid_wh,axis=-2) # (num_bboxes,1,2),为了广播
        bboxes_maxes = valid_wh / 2.
        bboxes_mins = -bboxes_maxes

        '''
            这个的计算太骚了！
            我们本来用来求IOU的时候都是x_center-w选最大,x_center+w选最小，然后两者相减，
            x_center就抵消了，那么其实也就是对w和-w进行加减，他这里特殊的是，默认anchor
            是和gt_bboxes的中心点是一样的，否则不能抵消x_center
            shape为：
            lb:(num_bboxes,9,2) ub:(num_bboxes,9,2)
            intersection_area:(num_bboxes,9)
            gt_bboxes_area:(num_bboxes,1) anchors_area:(num_bboxes,9)
            IOU:(num_bboxes,9)
        '''
        lb = np.maximum(bboxes_mins,anchors_mins)
        ub = np.minimum(bboxes_maxes,anchors_maxes)
        intersection_wh = np.maximum(ub - lb,0.)
        intersection_area = intersection_wh[:,:,0] * intersection_wh[:,:,1]
        gt_bboxes_area = valid_wh[:,:,0] * valid_wh[:,:,1]
        anchors_area = anchors[:,:,0] * anchors[:,:,1] # (1,9,2)
        IOU = intersection_area / (gt_bboxes_area + anchors_area - intersection_area)
        best_anchor = np.argmax(IOU,axis=-1) # (num_bboxes,)

        for each_bbox,single_anchor in enumerate(best_anchor):
            for layer in range(num_layers):
                if single_anchor in anchors_mask[layer]:
                    # i,j是gt_bboxes所属特征层对应的x,y坐标
                    i = np.floor(gt_bboxes[each,each_bbox,0] * 
                        grid_shapes[layer][1]).astype('int32')
                    j = np.floor(gt_bboxes[each,each_bbox,1] *
                        grid_shapes[layer][0]).astype('int32')
                    '''
                        这里要乘上13,26,52是因为，在前面gt_bboxes是归一化了的，
                        说明他的输出是(13,13,),(26,26,),(52,52,)这三个特征层的数据，
                        这里只是拿到他在13*13这种特征层下的xy坐标，方便记录下
                        这个小格子的xywh信息，但是要注意这个xywh信息是归一化的，用的
                        时候还要乘回去
                    '''
                    # anchor_idx是当前这个anchor在当前特征层所有anchor里面的下标
                    anchor_idx = anchors_mask[layer].index(single_anchor)
                    # class_id是这个真实框的类别，作者把anchor叫做先验框
                    class_id = gt_bboxes[each,each_bbox,4].astype('int32')
                    y_true[layer][each,j,i,anchor_idx,:4] = \
                        gt_bboxes[each,each_bbox,:4]
                    y_true[layer][each,j,i,anchor_idx,4] = 1
                    y_true[layer][each,j,i,anchor_idx,5 + class_id] = 1
                    # 说明class_id是从0开始的
    return y_true 
    # [(batch_size,13,13,3,85),(batch_size,26,26,3,85),(batch_size,52,52,3,85)]

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

parser = argparse.ArgumentParser()
parser.add_argument('--classes_path',type=str,default='./model_data/voc_classes.txt',
    help='the path of datasets classes')
parser.add_argument('--pretrained_weights_path',type=str,
    default='./model_data/yolov4_coco_weights.h5')
parser.add_argument('--logs_path',type=str,default='./logs/')
parser.add_argument('--batch_size',type=int,default=2)
parser.add_argument('--epoch',type=int,default=100)
parser.add_argument('--freeze_epoch',type=int,default=50)
parser.add_argument('--lr_freeze',type=float,default=1e-3)
parser.add_argument('--lr_unfreeze',type=float,default=1e-4)
parser.add_argument('--img_h',type=int,default=416)
parser.add_argument('--img_w',type=int,default=416)
parser.add_argument('--cosine',type=bool,default=True,help='using cosine decay or not')
parser.add_argument('--mosaic',type=bool,default=False,help='using mosaic to augment'
    'data or not')
parser.add_argument('--label_smoothing_kesi',type=int,default=0,help='the value of '
    'argument label smoothing')
opt = parser.parse_args()

if not os.path.exists(opt.logs_path):
    os.mkdir(opt.logs_path)

def main():
    annotation_path = '2007_train.txt'
    log_dir = opt.logs_path # 模型保存的位置
    classes_path = opt.classes_path # 自己的数据集改这个
    anchors_path = './model_data/yolo_anchors.txt'
    pretrained_weights_path = opt.pretrained_weights_path
    input_shape = (opt.img_h,opt.img_w)
    class_names = get_gt_classes(classes_path)
    anchors = get_anchors(anchors_path)
    num_classes = len(class_names)
    num_anchors = len(anchors)

    '''
        Yolov4的tricks应用
        mosaic 马赛克数据增强 True or False
        实际测试时mosaic数据增强并不稳定，所以默认为False
        Cosine_scheduler 余弦退火学习率 True or False
        label_smoothing 标签平滑 0.01以下一般 如0.01、0.005
    '''
    mosaic,cosine_scheduler = opt.mosaic,opt.cosine
    label_smoothing_kesi = opt.label_smoothing_kesi

    # 下面创建yolo模型
    image_input = Input(shape=(None,None,3))
    h,w = input_shape # 416,416
    print('create yolov4 model with {} anchors and {} classes.'.
        format(num_anchors,num_classes))
    model_net = yolov4_net(image_input,num_anchors // 3,num_classes) 
    # //3是因为每一层只有3个anchor

    # 加载ImageNet预训练权重
    print('loading weights from {}'.format(pretrained_weights_path))
    model_net.load_weights(pretrained_weights_path,by_name=True,skip_mismatch=True)

    # 设置loss函数,[(13, 13, 3, 25), (26, 26, 3, 25), (52, 52, 3, 25)]
    y_true = [Input(shape=(h // {0:32,1:16,2:8}[layer],w // {0:32,1:16,2:8}[layer],\
        num_anchors // 3,num_classes + 5)) for layer in range(3)]
    loss_input = [*model_net.output,*y_true]
    model_loss = Lambda(loss,output_shape=(1,),name='yolov4_loss',
        arguments={'anchors':anchors,'num_classes':num_classes,
        'label_smoothing_kesi':label_smoothing_kesi})(loss_input)
    # 这就是格式
    model = Model([model_net.input,*y_true],model_loss)
    '''
        这个模型，跟我们之前的模型都不一样，之前都是直接给input然后输出y_pred,
        然后输入loss函数跟y_true求loss，但是这里的model都只是一个空壳，没有输入的，
        只是定义了输入和输出是什么，然后他是给图像输入和y_true输入，得到输出，
        Lambda只是一个层，相当于在最后，给了你一个计算并输出loss的层而已
    '''
    # 训练参数的设置
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 
        '/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        save_weights_only=True,save_best_only=False,period=1)
    early_stopping = EarlyStopping(min_delta=0,patience=10,verbose=1)
    '''
        logging表示tensorboard的保存地址
        checkpoint用于设置权值保存的细节，period用于修改多少epoch保存一次
        early_stopping用于设定早停，val_loss多次不下降自动结束训练，表示模型基本收敛
    '''

    # 划分训练集验证集，1:9
    val_ratio = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(100)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_ratio) 
    num_train = len(lines) - num_val

    '''
        冻结网络是因为，一开始有249层是训练ImageNet的主干网络，也就是进入SPP之前的
        CSPDarkNet53，他是有预训练权重的，所以可以先不训练，给点机会后续的网络
        说明他一开始的Model只是生成了一个壳，最终还是要使用model_net来进行训练
    '''
    freeze_layers = 249
    for i in range(freeze_layers):
        model_net.layers[i].trainable = False
    print('freezing the first {} layers of total {} layers'.format(freeze_layers,
        len(model_net.layers)))

    # 现在冻结了进行训练
    if True:
        init_epoch = 0 # 这些数据就是自定义是这样算而已
        freeze_epoch = opt.freeze_epoch # freeze_epoch就是在freeze状态下训练的epoch
        batch_size = opt.batch_size 
        # num_train在这里就是sample count，所以用来计算total_steps
        learning_rate_base = opt.lr_freeze

        # 如果使用余弦退火
        if cosine_scheduler:
            warmup_epoch = int((freeze_epoch - init_epoch) * 0.2) 
            # 预热的总epoch，只拿10个作为预热的epoch
            total_steps = int((freeze_epoch - init_epoch) * num_train / batch_size)
            warmup_steps = int(warmup_epoch * num_train / batch_size) 
            # 预热步长，公式warmup_epoch * sample_count / batch_size
            reduce_lr = WarmupCosineDecayScheduler(
                learning_rate_base=learning_rate_base,
                total_steps=total_steps, # 即Ti，总的训练的步数
                warmup_init_learning_rate=1e-4, # warmup步骤的初始化lr
                warmup_steps=warmup_steps, # warmup总共有多少步
                hold_base_rate_steps=num_train,
                min_learning_rate=1e-6 # nmin
            )
            model.compile(optimizer=Adam(),loss={'yolov4_loss':lambda y_true,
                y_pred:y_pred})
        else: 
            '''
                ReduceLROnPlateau是个回调函数，val_loss是内置的str
                参数：
                monitor:监测的值，可以是accuracy,val_loss,val_accuracy
                factor:缩放学习率的值，学习率将以lr *= factor的形式被减少
                patience:当patience个epoch之后而模型性能不提升时，开始学习率衰减
                mode:'auto','min','max'之一，默认'auto'
                epsilon:阈值，用来确定是否进入检测值的"平原区"
                cooldown:学习率减少后，会经过cooldown个epoch才重新进行正常操作
                min_lr:学习率最小值，能缩小到的下限
                verbose:详细信息模式，0或1
            '''
            reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.5,
                patience=3,verbose=1)
            model.compile(optimizer=Adam(learning_rate_base),
                loss={'yolov4_loss':lambda y_true,y_pred:y_pred})
        '''
            首先，我们知道的是yolov4_loss在model_loss定义的时候只是一个name,
            所以这里这种字典的形式估计也只是一个名字，就把这个Model对应的loss叫
            这个名字而已，因为发现没有'yolov4_loss'根本没区别，但是问题是，
            你不叫yolov4_loss就不行，会报这个错：
            Found unexpected keys that do not correspond to any Model output
            代码里也说使用定制的yolov4_loss层，所以有可能是这样：如果有字典的出现，
            那么就会自动去匹配字典的key是不是跟输出层的name（如果有name的话）相同，
            因为他认为‘嗯，我就要叫你这个名字的输出层来计算loss’，不同就报上面那个错
            如果是相同的，那么就用输出层的输出作为y_pred来计算loss，所以这个也对应了
            作者说用特制化的Lambda层来计算的说法，如果没有字典，
            就直接lambda y_true, y_pred: y_pred的话应该就是默认用模型的output
            来计算loss，只不过少了‘强制匹配’的步骤，而loss本身是个函数！是个函数！
            匿名函数也是函数，而这个函数怎么运作的就是下面说的

            因为在上面的时候，他将loss变成了Lambda一个层，然后定义Model的输出层为
            这个Lambda loss层，也就是跟平常输出预测结果不一样，是直接输出loss，
            是直接输出loss的
            而因为这是keras的Model，所以看keras是怎么写损失函数的，每个损失函数都是
            输入y_true,y_pred然后return某种loss，比如：
            def mean_squared_error(y_true, y_pred):
                return K.mean(K.square(y_pred - y_true), axis=-1)
            那么在Model里面他也肯定是这样的，有y_true,y_pred和返回的loss，但是！
            但是y_pred其实是模型的输出，正常来说是输出一个标签之类的，而这里是直接将模型
            的最后输出(即y_pred)设定为loss，也就是说我传进去的时候就是loss而不是
            y_pred了，所以返回y_pred就等于直接返回loss了！关于输入给keras定义的
            损失函数的y_true，个人理解就是y_train的(训练集label或者叫y_true)，
            而这里根本就不用管，就直接输出y_pred也就是loss了
    
            所以综合loss='yolov4_loss':lambda y_true,y_pred:y_pred来看，意思就是：
            我这里的loss要用这个函数来计算：
            def keras_style_loss(y_true,model_loss):
                return model_loss # model_loss为那个Lambda层的输出
            所以在这里他就会强制查找'yolov4_loss'这个name，原因是输出层就叫这个，
            那么找到了之后就使用model_loss这个Lambda层来计算loss，计算的原则就如上所说。

            实验证明，lambda y_true, y_pred: y_pred和
            {'yolov4_loss':lambda y_true, y_pred: y_pred}根本没差
        '''
        print('train on {} samples,validation on {} samples,with batch size {}.'.
            format(num_train,num_val,batch_size))
        '''
            verbose:日志展示，整数
                0:为不在标准输出流输出日志信息
                1:显示进度条
                2:每个epoch输出一行记录
            initial_epoch:开始训练的轮数，好像是跟epoch之前进行训练
            比如下面解冻后，因为之前训练了50个freeze_epoch，所以开始训练的epoch就是
            第50个，然后epoch设为100，也就是从第50个开始训练到第100个，总的来说就是
            冻结50个解冻50个epoch
        '''
        model.fit(
            data_generator(lines[:num_train],batch_size,input_shape,anchors,
                num_classes,mosaic=mosaic,random=True),
            steps_per_epoch=max(1,num_train // batch_size),
            validation_data=data_generator(lines[num_train:],batch_size,
                input_shape,anchors,num_classes,mosaic=mosaic,random=True),
            validation_steps=max(1,num_val // batch_size),
            epochs=freeze_epoch,
            initial_epoch=init_epoch,
            callbacks=[logging,checkpoint,reduce_lr,early_stopping]
        )
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

    # 解冻，然后训练
    for i in range(freeze_layers):
        model_net.layers[i].trainable = True

    if True:
        epoch = opt.epoch
        freeze_epoch = opt.freeze_epoch
        batch_size = opt.batch_size
        learning_rate_base = opt.lr_unfreeze

        # 如果使用余弦退火
        if cosine_scheduler:
            warmup_epoch = int((epoch - freeze_epoch) * 0.2) 
            # 预热的总epoch，此时的预热步骤已经是解冻后的训练过程了
            total_steps = int((epoch - freeze_epoch) * num_train / batch_size)
            warmup_steps = int(warmup_epoch * num_train / batch_size) # 预热步长
            reduce_lr = WarmupCosineDecayScheduler(
                learning_rate_base=learning_rate_base,
                total_steps=total_steps, # 即Ti，总的训练的步数
                warmup_init_learning_rate=1e-5, # warmup步骤的初始化lr
                warmup_steps=warmup_steps, # warmup总共有多少步
                hold_base_rate_steps=num_train // 2,
                min_learning_rate=1e-6 # nmin
            )
            model.compile(optimizer=Adam(),loss={'yolov4_loss':lambda y_true,
                y_pred:y_pred})
        else: 
            reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.5,
                patience=3,verbose=1)
            model.compile(optimizer=Adam(learning_rate_base),
                loss={'yolov4_loss':lambda y_true,y_pred:y_pred})
        print('train on {} samples,validation on {} samples,with batch size {}.'.
            format(num_train,num_val,batch_size))
        model.fit(
            data_generator(lines[:num_train],batch_size,input_shape,anchors,
                num_classes,mosaic=mosaic,random=True),
            steps_per_epoch=max(1,num_train // batch_size),
            validation_data=data_generator(lines[num_train:],batch_size,
                input_shape,anchors,num_classes,mosaic=mosaic,random=True),
            validation_steps=max(1,num_val // batch_size),
            epochs=epoch,
            initial_epoch=freeze_epoch,
            callbacks=[logging,checkpoint,reduce_lr,early_stopping]
        )
        model.save_weights(log_dir + 'last1.h5')

if __name__ == '__main__':
    main()