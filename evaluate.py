import tensorflow as tf
from utils import nms

# 下面这个函数将预测值变成真实值，就是作者的yolo_head
def pred2origin(y_pred,anchors,num_classes,input_shape,calc_loss=False):
    # y_pred:(batch_size,13,13,3,num_classes + 5)，他就是yolov4_net的输出
    # input_shape就是(416,416)
    '''
        input_shape是(416,416)的原因是他需要把13*13,26*26,52*52的特征图先放缩回416
        *416的大小，因为这是训练的图片大小，特征图和anchors都是基于这个大小的，然后除以
        416归一化成1，然后就可以给image_shape相乘，回复到原图大小下的正确的大小
    '''
    num_anchors = len(anchors)
    y_pred = tf.cast(tf.convert_to_tensor(y_pred),dtype=tf.float32)
    anchors = tf.cast(tf.reshape(tf.constant(anchors),[1,1,1,num_anchors,2]),
                    dtype=tf.float32) # 方便广播
    grid_shape = tf.shape(y_pred)[1:3]
    #h,w = y_pred.shape[1:3] # 就是宽高，这里说明有可能只有一张图
    x_origin = tf.cast(tf.expand_dims(tf.transpose(tf.reshape(tf.tile(
        [tf.range(0,grid_shape[0])],[grid_shape[1] * num_anchors,1]),
        (num_anchors,grid_shape[0],grid_shape[1])),(1,2,0)),axis=0),dtype=tf.float32)
    y_origin = tf.transpose(x_origin,(0,2,1,3))
    y_pred = tf.reshape(y_pred,[-1,grid_shape[0], grid_shape[1],
        num_anchors,num_classes + 5])
    # 下面是将预测值变成真实值
    # y_pred:(batch_size,13,13,3,num_classes + 5)
    y_pred_x = y_pred[:,:,:,:,0]
    y_pred_y = y_pred[:,:,:,:,1]
    # 除以h和w是归一化
    grid_shape = tf.cast(grid_shape,dtype=tf.float32)
    y_pred_x = tf.expand_dims(tf.cast(((tf.nn.sigmoid(y_pred_x) + x_origin) / 
        grid_shape[1]),dtype=y_pred.dtype),axis=-1)
    y_pred_y = tf.expand_dims(tf.cast(((tf.nn.sigmoid(y_pred_y) + y_origin) / 
        grid_shape[0]),dtype=y_pred.dtype),axis=-1)
    bboxes_xy = tf.concat([y_pred_x,y_pred_y],axis=-1)
    bboxes_wh = tf.cast(tf.exp(y_pred[:,:,:,:,2:4]) * anchors / input_shape[::-1],
                    dtype=y_pred.dtype)
    bboxes_confs = tf.nn.sigmoid(y_pred[:,:,:,:,4:5])
    bboxes_class_probs = tf.nn.sigmoid(y_pred[:,:,:,:,5:])

    # 在计算loss的时候返回xy_origin,y_pred,bboxes_xy,bboxes_wh
    # 在预测的时候返回box_xy, box_wh, box_confidence, box_class_probs
    if calc_loss == True:
        x_origin = tf.expand_dims(x_origin,axis=-1)
        y_origin = tf.expand_dims(y_origin,axis=-1)
        xy_origin = tf.concat([x_origin, y_origin],axis=-1)
        return xy_origin,y_pred,bboxes_xy,bboxes_wh
    return bboxes_xy,bboxes_wh,bboxes_confs,bboxes_class_probs

# 对bboxes进行调整，使其符合真实图片的样子
# 上面是y_pred预测出来的框(pred_bboxes)进行还原，这里是只对bboxes恢复到原图
def bboxes_in_image(bboxes_xy,bboxes_wh,input_shape,image_shape):
    '''
        image_shape:(图片的h,图片的w)，这里是(1330,1330)，每张图不一样
        input_shape:(416,416)
        所以他返回的，是在图片大小下的bounding boxes的坐标
    '''
    # 把y轴放前面是因为方便预测框和图像的宽高相乘
    bboxes_yx = bboxes_xy[:,:,:,:,::-1] # 这两个是归一化的
    bboxes_hw = bboxes_wh[:,:,:,:,::-1]
    input_shape = tf.cast(input_shape,dtype=bboxes_yx.dtype)
    image_shape = tf.cast(image_shape,dtype=bboxes_yx.dtype)
    new_shape = tf.round(image_shape * tf.reduce_min(input_shape / image_shape))
    # keras.round底层就是tf.round，是四舍六入五取偶，取最小的shape
    # 然后求出来图像有效区域相对于图像左上角的偏移情况
    offset = (input_shape - new_shape) / 2. / input_shape 
    # 这个也是要归一化，除以2是因为他是只在一半的方向上加这个offset
    scale = input_shape / new_shape
    bboxes_yx = (bboxes_yx - offset) * scale
    bboxes_hw *= scale
    # 下面是(ymin,xmin,ymax,xmax)
    bboxes_yxyx = tf.concat([(bboxes_yx[:,:,:,:,:1] - bboxes_hw[:,:,:,:,:1] / 2),
                             (bboxes_yx[:,:,:,:,1:2] - bboxes_hw[:,:,:,:,1:2] / 2),
                             (bboxes_yx[:,:,:,:,:1] + bboxes_hw[:,:,:,:,:1] / 2),
                             (bboxes_yx[:,:,:,:,1:2] - bboxes_hw[:,:,:,:,1:2] / 2)])
    bboxes_yxyx_inimg = bboxes_yxyx * tf.concat([image_shape,image_shape],axis=-1)
    return bboxes_yxyx_inimg

# 获取每个bounding box的坐标和它对应的得分
def acquire_coords_scores(y_pred,anchors,num_classes,input_shape,image_shape,
    letterbox_image):
    '''
        input_shape:(416,416)，用来给pred2origin得出归一化的xy中心点和宽高的
        image_shape:(1330,1330),是(h,w)不是(w,h)
        return:
        bboxes_coords:(batch_size*13*13*3,4)
        bboxes_scores:(batch_size*13*13*3,num_classes)
        他就是在这个函数里面，将坐标给搞到原图大小下也就是1300*1300的大小下
    '''
    input_shape = tf.cast(input_shape,dtype=tf.float32)
    bboxes_xy,bboxes_wh,bboxes_confs,bboxes_class_probs = \
        pred2origin(y_pred,anchors,num_classes,input_shape) 
    '''
        bboxes_xy:(batch_size,13,13,3,2),归一化后的
        bboxes_wh:(batch_size,13,13,3,2),归一化后的
        bboxes_confs:(batch_size,13,13,3,1)
        bboxes_class_probs:(batch_size,13,13,3,80)
        confs和probs不是1，gt是1
    '''
    if letterbox_image:
        bboxes_coords = bboxes_in_image(bboxes_xy,bboxes_wh,input_shape,image_shape)
        '''
            返回的是bounding boxes在原始图片上的坐标，不过是yxyx的格式，不是(xmin,
            ymin,xmax,ymax)而是(ymin,xmin,ymax,xmax)
            其实这个跟下面else的部分几乎是一样的，只是因为他有灰条，所以要特殊放缩处理，
            结果都是返回的在图片上的bboxes的yxyx坐标
        '''
    else:
        bboxes_yx = bboxes_xy[:,:,:,:,::-1]
        bboxes_hw = bboxes_wh[:,:,:,:,::-1]
        bboxes_yxmin = bboxes_yx - (bboxes_hw / 2.)
        bboxes_yxmax = bboxes_yx + (bboxes_hw / 2.)
        image_shape = tf.cast(image_shape,dtype=image_shape.dtype)

        bboxes_coords = tf.concat([
                bboxes_yxmin[:,:,:,:,0:1] * image_shape[0], # ymin
                bboxes_yxmin[:,:,:,:,1:2] * image_shape[1], # xmin
                bboxes_yxmax[:,:,:,:,0:1] * image_shape[0], # ymax
                bboxes_yxmax[:,:,:,:,1:2] * image_shape[1], # xmax
            ],axis=-1)
    # 获取最终得分和框的位置
    bboxes_coords = tf.reshape(bboxes_coords,[-1,4])
    bboxes_scores = bboxes_confs * bboxes_class_probs
    bboxes_scores = tf.reshape(bboxes_scores,[-1,num_classes])
    return bboxes_coords,bboxes_scores

def evaluate(y_pred,anchors,num_classes,image_shape,max_bboxes=20,
    score_threshold=0.6,iou_threshold=0.5,eager=False,letterbox_image=True):
    '''
        y_pred:[(batch_size,13,13,3,25),(batch_size,26,26,3,25),
            (batch_size,52,52,3,25),(?,?)] (?,?)->(1330,1330)
        letterbox_image:就是将图像嵌入到灰度图里面变成正方形又保证不会失真的处理方法
    '''
    if eager: # 这里image_shape被重塑成(1330,1330)了！不是eager的话就不重塑，所以不能不传
        image_shape = tf.reshape(y_pred[-1],[-1]) # 这里的image_shape应该是(h,w)
        num_layers = len(y_pred) - 1 # 减去最后图片shape那一维
    else:
        num_layers = len(y_pred) # 说明没有把image_shape传进来
    # 13x13的特征层对应的anchor是[142,110],[192,243],[459,401]
    # 26x26的特征层对应的anchor是[36,75],[76,55],[72,146]
    # 52x52的特征层对应的anchor是[12,16],[19,36],[40,28]
    anchors_mask = [[6,7,8],[3,4,5],[0,1,2]]
    input_shape = tf.shape(y_pred[0])[1:3] * 32 # (416,416)
    bboxes_coords,bboxes_scores = [],[]

    for layer in range(num_layers):
        cur_coords,cur_scores = acquire_coords_scores(y_pred[layer],
            anchors[anchors_mask[layer]],num_classes,input_shape,
            image_shape,letterbox_image)
        bboxes_coords.append(cur_coords) # cur_coords和cur_scores都是2维的，具体看上面
        bboxes_scores.append(cur_scores)
    bboxes_coords = tf.concat(bboxes_coords,axis=0)
    bboxes_scores = tf.concat(bboxes_scores,axis=0) 
    '''
        这里的意思是，有可能他每个元素的shape不完全一样，比如[(3,4),(2,4),(1,4)]这样，
        他就是堆起来，比如[(2,4),(3,4),(1,4)]堆起来就是(6,2)
        所以最终的bboxes_coords包含了13*13,26*26,52*52的全部坐标和得分
        所以下面mask:(batch_size*2*(13*13+26*26+52*52),num_classes)
    '''
    mask = bboxes_scores >= score_threshold
    max_bboxes_tensor = tf.constant(max_bboxes,dtype='int32')

    final_coords,final_scores,final_classes = [],[],[]
    for c in range(num_classes):
        # 取出所有scores大于score_threshold的框坐标和分数
        cur_class_coords = tf.boolean_mask(bboxes_coords,mask[:,c])
        cur_class_scores = tf.boolean_mask(bboxes_scores[:,c],mask[:,c])
        
        nms_index = tf.image.non_max_suppression(cur_class_coords,
            cur_class_scores,max_bboxes_tensor,iou_threshold=iou_threshold)
        '''
        nms_index = tf.py_function(nms,inp=[cur_class_coords,cur_class_scores,
            max_bboxes_tensor,iou_threshold],Tout=tf.int32)
        # print(nms_index)   
        '''
        '''
            他返回的是，在coords里面分数大于threshold的coord对应的下标
        '''
        # (valid_num,4),(valid_num,),(valid_num,)
        selected_class_coords = tf.gather(cur_class_coords,nms_index)
        selected_class_scores = tf.gather(cur_class_scores,nms_index)
        selected_classes = tf.ones_like(selected_class_scores,dtype='int32') * c
        # 这个是对应的scores和coords的类别
        final_coords.append(selected_class_coords)
        final_scores.append(selected_class_scores)
        final_classes.append(selected_classes)

    final_coords = tf.concat(final_coords,axis=0)
    final_scores = tf.concat(final_scores,axis=0)
    final_classes = tf.concat(final_classes,axis=0)

    return final_coords,final_scores,final_classes
    # 也就是说，这里把nms之后的框在图片上的真实yxyx坐标，对应的得分，对应的类别，
    # 都返回出来了，即1330*1330大小下的bounding boxes的坐标，所以他后面才能直接画
    # 也证明其实anchors的宽高，挺万能的
