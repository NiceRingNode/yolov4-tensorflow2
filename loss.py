import tensorflow as tf
import numpy as np
from evaluate import *
import tensorflow.keras.backend as K
from utils import calc_IOU

def calc_CIOU(bboxes1,bboxes2):
    '''
        :param boxes1:(batch_size,h,w,num_bboxes,4),4:(x_center,y_center,w,h)
        :param boxes2:(batch_size,h,w,num_bboxes,4),4:(x_center,y_center,w,h)
        :return:CIOU:(batch_size,h,w,num_bboxes,1)
        毕竟还是IOU的一种，shape肯定还是IOU那个样子，本身的MSE也是输出的
        (batch_size,h,w,num_bboxes,1)，square会降维
        传进来的是pred_coords和y_true[:4]，xy是sigmoid后加上了xy_origin的然后再归一化的，
        wh是exp后乘anchor但是归一化了的，归一化是直接除以416的，因为他的anchor的大小是有459
        那么大的，所以也说明了下面那个coord_scale是2减去1*1以内的
        这样也解释了这个IOU是ok的
    '''
    IOU,bboxes1_xy,bboxes1_wh,bboxes2_xy,bboxes2_wh,bboxes1_xymin,bboxes1_xymax,\
        bboxes2_xymin,bboxes2_xymax = calc_IOU(bboxes1,bboxes2,for_CIOU=True)
    # 计算中心的距离
    '''
        bboxes1_xy:(batch_size,h,w,num_bboxes,2)
        a = np.arange(1*13*13*3*2).reshape(1,13,13,3,2)
        b = np.arange(1*13*13*3*2).reshape(1,13,13,3,2) - 10
        print((tf.square(a - b)).shape)
        这里的输出是(1, 13, 13, 3, 2)，说明会保持原状，
        所以直接reduce_sum(axis=-1)就会全部求和
    '''
    center_distance = tf.reduce_sum(tf.square(bboxes1_xy - bboxes2_xy),axis=-1)
    global_xymin = tf.minimum(bboxes1_xymin,bboxes2_xymin)
    global_xymax = tf.maximum(bboxes1_xymax,bboxes2_xymax)
    global_diagonal = tf.reduce_sum(tf.maximum(global_xymax - global_xymin,0.0),
        axis=-1)
    CIOU = IOU - 1.0 * (center_distance / tf.maximum(global_diagonal,1e-7))
    # CIOU:(batch_size,h,w,num_bboxes)
    v = 4.0 * tf.square(tf.math.atan2(bboxes1_wh[:,:,:,:,0],tf.maximum(
        bboxes1_wh[:,:,:,:,1],1e-7)) - tf.math.atan2(bboxes2_wh[:,:,:,:,0],
        tf.maximum(bboxes2_wh[:,:,:,:,1],1e-7))) / (np.pi ** 2)
    alpha = v / tf.maximum((1.0 - IOU + v),1e-7)
    CIOU -= alpha * v # 其实这里是负号了
    CIOU = tf.expand_dims(CIOU,axis=-1)
    '''
        tf.where(bool_mask,x,y)
        bool_mask是个bool数组，如果对应位置是True就返回x对应位置的东西，如果是False就返回y
        位置对应的东西，下面就是填充而已
    '''
    CIOU = tf.where(tf.math.is_nan(CIOU),tf.zeros_like(CIOU),CIOU)
    return CIOU

def label_smoothing(y_class_probs,kesi):
    # y_class_probs就是那20个或80个概率，(batch_size,13,13,3,20)
    # https://zhuanlan.zhihu.com/p/116466239
    num_classes = tf.cast(y_class_probs.shape[-1],dtype=tf.float32)
    kesi = tf.constant(kesi,dtype=tf.float32)
    return y_class_probs * (1.0 - kesi) + kesi / num_classes

def loss(args,anchors,num_classes,label_smoothing_kesi=0.1,ignore_threshold=0.6,
    normalize=True):
    '''
    y_true:[(batch_size,13,13,3,25),(batch_size,26,26,3,25),(batch_size,52,52,3,25)]
    y_pred:[(batch_size,13,13,3,25),(batch_size,26,26,3,25),(batch_size,52,52,3,25)]
    anchors:[9],不同的size分别对应有三个不同的框
    num_classes:20或80
    这里的y_pred就是作者的yolo_output,y_true就是y_true
    '''
    tot_loss,num_positive = 0,0 # 三层的loss
    num_layers = len(anchors) // 3 # 他读进来是[9],所以//3是对的
    y_true = args[num_layers:]
    y_pred = args[:num_layers]

    # 13x13的特征层对应的anchor是[142,110],[192,243],[459,401]
    # 26x26的特征层对应的anchor是[36,75],[76,55],[72,146]
    # 52x52的特征层对应的anchor是[12,16],[19,36],[40,28]
    anchors_mask = [[6,7,8],[3,4,5],[0,1,2]] if num_layers == 3 else \
        [[3,4,5],[1,2,3]]
    input_shape = tf.cast(tf.shape(y_true[0])[1:3] * 32,dtype=K.dtype(y_true[0]))
    # 也就是h,w = 416,416
    batch_size = tf.shape(y_pred[0])[0]
    batch_size_float = tf.cast(tf.shape(y_pred[0])[0],dtype=K.dtype(y_pred[0]))

    for layer in range(num_layers):
        '''
            这里是对(13,13),(26,26),(52,52)三个层按顺序求loss，正常来说y_true是需要
            除以32,16,8的，因为我们在图片提取出来的数据是真实图片416*416这个size下的数据
        '''
        '''
            将y_pred的单层输出还原到论文的图里的样子，就是sigmoid和加上xc、yc
            xy_origin:(13,13,1,2) 网格坐标
            raw_pred:(batch_size,13,13,3,85) 尚未处理的预测结果，也就是yolov4_net
                的raw输出,注意，这个是不需要写layer的
            pred_xy:(batch_size,13,13,3,2) 解码后的中心坐标，sigmoid后加上了x_origin的
            pred_wh:(batch_size,13,13,3,2) 解码后的宽高坐标，加上了y_origin的
        '''
        xy_origin,raw_pred,pred_xy,pred_wh = pred2origin(y_pred[layer],
            anchors[anchors_mask[layer]],num_classes,input_shape,calc_loss=True)
        pred_coords = tf.concat([pred_xy,pred_wh],axis=-1) # (batch_size,13,13,3,4)
        gt_coords = y_true[layer][:,:,:,:,:4]
        gt_confs = y_true[layer][:,:,:,:,4:5]  # 这个作者叫做object_mask
        # 必须在这里出现，计算mask用到
        IOU = calc_IOU(pred_coords,gt_coords) # 这个IOU只是用来计算best_iou_mask
        '''
            gt_areas:(batch_size,13,13,3)
            pred_areas:(batch_size,13,13,3)
            intersection_wh:(batch_size,h,w,num_bboxes,2)->(batch_size,13,13,3,2)
            intersection:(batch_size,h,w,num_bboxes)->(batch_size,13,13,3)
            IOU:(batch_size,h,w,num_bboxes)->(batch_size,13,13,3)
            这里的IOU是：所有13*13*3个预测的anchor_box与对应位置上gt的anchor_box之间的IOU
        '''
        best_iou_mask = tf.expand_dims(tf.equal(IOU,tf.reduce_max(
            IOU,axis=-1,keepdims=True)),axis=-1)
        # 因为IOU是(None, 13, 13, 3)，就自动压缩了，所以要扩维
        '''
            reduce_max那个是(batch_size,h,w,1),可能没有第五维的1了
            比较出来，best_iou_mask:(batch_size,h,w,num_bboxes)全是bool值
            他这里不设置阈值，如果是0.6有可能一个框都不到0.6，就以最大的框作为阈值
        '''
        best_iou_mask = tf.cast(best_iou_mask,tf.float32)
        mask = best_iou_mask * gt_confs
        '''
            gt_confs:(batch_size,h,w,num_bboxes)
            best_iou_mask:(batch_size,h,w,1)
            mask:(batch_size,h,w,num_bboxes,1)->(batch_size,13,13,3,1) 不用扩了！！！
            所以会广播，每个grid里面都选出了IoU最大那个，但是gt_confs只有几个有值，
            其他全是0,所以乘出来的结果是169个格子里面只有某几个格子的IoU最大的
            bbox有值是1，其他全是0
            在这里他其实就利用了responsible grid这个操作了
            而作者是先把1选出来，再去算IOU，再跟阈值进行比较
            相当于在代码作者的obj_mask上加了一层best_iou_mask，但是这样1-mask不符合
            原论文的意思，人家说的是noobj的地方，我们在标注的时候是看小格子的，只要这个
            小格子有，那么3个anchor box都有，但是best_iou_mask之后我们就相当于少了一
            些东西
        '''

        # 计算一个负样本的mask，将有一些预测出来特别准的框给筛掉，保留那些不怎么准的
        # 框的坐标，计算noobj_loss的时候就相当于忽略特别准的框了，合乎常理
        negative_mask = tf.TensorArray(K.dtype(y_true[layer][0]),size=1,
                                               dynamic_size=True)
        gt_confs_bool = tf.cast(gt_confs,dtype=tf.bool)

        def count_negative_example(each,negative_mask):
            gt_bboxes_have_obj = tf.boolean_mask(y_true[layer][each,:,:,:,:4],
                gt_confs_bool[each,:,:,:,0])
            iou = calc_IOU(pred_coords[each],gt_bboxes_have_obj,
                           is2_boolean_mask=True)
            '''
                pred_coords[each]:(13,13,3,4)
                gt_bboxes_have_obj:(valid_num_bboxes,4)
                iou:(13,13,3,valid_num_bboxes)
                best_iou:(13,13,3)
                negative_mask:(13,13,3) 单个，write之后是(batch_size,13,13,3)
            '''
            best_iou = tf.reduce_max(iou,axis=-1)
            negative_mask = negative_mask.write(each,tf.cast(
                best_iou < ignore_threshold,tf.float32))
            return each + 1,negative_mask

        _,negative_mask = tf.while_loop(lambda each,*args:each < batch_size,
            count_negative_example,[0,negative_mask])
        # while_loop(condition:bool,body:Function,arguments),所以一开始的lambda
        # 是循环判断条件，后面是循环体和初始参数，对每一张图片进行
        negative_mask = negative_mask.stack() # (batch_size,13,13,3)
        '''
            stack这里的作用是，把TensorArray里面的元素叠起来再输出，可能在一开始
            write的时候并不是(13,13,3)
        '''
        negative_mask = tf.expand_dims(negative_mask,axis=-1)

        # 下面开始计算loss
        coords_scale = 2 - y_true[layer][:,:,:,:,2:3] * y_true[layer][:,:,:,:,3:4]
        # y_true是因为在preprocess_true_bboxes那里归一化了，y_pred是pred2origin
        # 归一化了
        ciou = calc_CIOU(pred_coords,gt_coords)
        coords_loss = (1 - ciou) * coords_scale * mask
        # shape:(batch_size,13,13,3,1),1是因为扩维了，本身求和了是因为求CIOU的时候
        # square已经求和了

        # 置信度loss，分为有obj的loss和无obj的loss
        ## obj_loss
        pred_confs = raw_pred[:,:,:,:,4:5]
        obj_loss = mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=
            gt_confs,logits=pred_confs)
        noobj_loss = (1 - gt_confs) * tf.nn.sigmoid_cross_entropy_with_logits(labels=
            gt_confs,logits=pred_confs) * negative_mask
        # obj_loss和noobj_loss:(batch_size,13,13,3,1),BCE根本不改变shape

        # 类别损失
        gt_probs = y_true[layer][:,:,:,:,5:]  # (batch_size,13,13,3,20)
        pred_probs = raw_pred[:,:,:,:,5:]
        if label_smoothing_kesi:
            gt_probs = label_smoothing(gt_probs)
        class_probs_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=
             gt_probs,logits=pred_probs) * mask
        # BCE后，shape:(batch_size,13,13,3,20)

        #求和，这几个reduce_sum之后都剩下()，表示标量
        coords_loss = tf.reduce_sum(coords_loss,axis=[0,1,2,3,4])
        confs_loss = tf.reduce_sum(obj_loss,axis=[0,1,2,3,4]) + \
                     tf.reduce_sum(noobj_loss,axis=[0,1,2,3,4])
        class_probs_loss = tf.reduce_sum(class_probs_loss,axis=[0,1,2,3,4])
        tot_loss += (coords_loss + confs_loss + class_probs_loss)
        num_positive += tf.maximum(tf.reduce_sum(tf.cast(
            gt_confs,tf.float32)),1)

    tot_loss = tf.expand_dims(tot_loss,axis=-1)

    if normalize:
        tot_loss /= num_positive
    else:
        tot_loss /= batch_size_float
    return tot_loss # 就是reduce_mean