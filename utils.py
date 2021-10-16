import cv2 as cv
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import warnings

def rand(a:float=0.0,b:float=1.0):
    # np.random.rand()产生一个服从0,1均匀分布的值，# 这样写跟原来没区别
    return np.random.rand() * (b - a) + a

def merge_bboxes(bboxes, cutx, cuty):
    merge_bbox = []
    for i in range(len(bboxes)):
        for box in bboxes[i]:
            tmp_box = []
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            if i == 0:
                if y1 > cuty or x1 > cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2 - y1 < 5:
                        continue
                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    if x2 - x1 < 5:
                        continue

            if i == 1:
                if y2 < cuty or x1 > cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2 - y1 < 5:
                        continue
                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    if x2 - x1 < 5:
                        continue
            if i == 2:
                if y2 < cuty or x2 < cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2 - y1 < 5:
                        continue
                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2 - x1 < 5:
                        continue
            if i == 3:
                if y1 > cuty or x2 < cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2 - y1 < 5:
                        continue
                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2 - x1 < 5:
                        continue
            tmp_box.append(x1)
            tmp_box.append(y1)
            tmp_box.append(x2)
            tmp_box.append(y2)
            tmp_box.append(box[-1])
            merge_bbox.append(tmp_box)
    return merge_bbox

def get_random_data_with_Mosaic(annotation_line,input_shape,max_boxes=100,
    hue=.1,sat=1.5,val=1.5):
    '''random preprocessing for real-time data augmentation'''
    h, w = input_shape
    min_offset_x = 0.3
    min_offset_y = 0.3
    scale_low = 1 - min(min_offset_x, min_offset_y)
    scale_high = scale_low + 0.2

    image_datas = []
    box_datas = []
    index = 0

    place_x = [0,0,int(w * min_offset_x),int(w * min_offset_x)]
    place_y = [0,int(h * min_offset_y),int(h * min_offset_y),0]

    for line in annotation_line:
        # 每一行进行分割
        line_content = line.split()
        # 打开图片
        image = Image.open(line_content[0])
        image = image.convert("RGB")
        # 图片的大小
        iw,ih = image.size
        # 保存框的位置
        box = np.array([np.array(list(map(int,box.split(',')))) for box
                        in line_content[1:]])

        # 是否翻转图片
        flip = rand() < .5
        if flip and len(box) > 0:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            box[:,[0,2]] = iw - box[:,[2,0]]

        # 对输入进来的图片进行缩放
        new_ar = w / h
        scale = rand(scale_low,scale_high)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        # 进行色域变换
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        x = cv.cvtColor(np.array(image, np.float32) / 255, cv.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image = cv.cvtColor(x, cv.COLOR_HSV2RGB)  # numpy array, 0 to 1

        image = Image.fromarray((image * 255).astype(np.uint8))
        # 将图片进行放置，分别对应四张分割图片的位置
        dx = place_x[index]
        dy = place_y[index]
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image) / 255

        index = index + 1
        box_data = []
        # 对box进行重新处理
        if len(box) > 0:
            np.random.shuffle(box)
            box[:,[0,2]] = box[:,[0,2]] * nw / iw + dx
            box[:,[1,3]] = box[:,[1,3]] * nh / ih + dy
            box[:,0:2][box[:,0:2] < 0] = 0
            box[:,2][box[:,2] > w] = w
            box[:,3][box[:,3] > h] = h
            box_w = box[:,2] - box[:,0]
            box_h = box[:,3] - box[:,1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]
            box_data = np.zeros((len(box), 5))
            box_data[:len(box)] = box

        image_datas.append(image_data)
        box_datas.append(box_data)

    # 将图片分割，放在一起
    cutx = np.random.randint(int(w * min_offset_x),int(w * (1 - min_offset_x)))
    cuty = np.random.randint(int(h * min_offset_y),int(h * (1 - min_offset_y)))

    new_image = np.zeros([h, w, 3])
    new_image[:cuty,:cutx,:] = image_datas[0][:cuty,:cutx,:]
    new_image[cuty:,:cutx,:] = image_datas[1][cuty:,:cutx,:]
    new_image[cuty:,cutx:,:] = image_datas[2][cuty:,cutx:,:]
    new_image[:cuty,cutx:,:] = image_datas[3][:cuty,cutx:,:]

    # 对框进行进一步的处理
    new_boxes = merge_bboxes(box_datas,cutx,cuty)

    # 将box进行调整
    box_data = np.zeros((max_boxes,5))
    if len(new_boxes) > 0:
        if len(new_boxes) > max_boxes: new_boxes = new_boxes[:max_boxes]
        box_data[:len(new_boxes)] = new_boxes
    return new_image, box_data

def letterbox_image(image,input_shape):
    # 把image嵌入input_shape大小的灰度图里面
    img_w,img_h = image.size # image是个PIL对象，先宽后高
    input_w,input_h = input_shape
    scale = min(input_w / img_w,input_h / img_h)
    new_w,new_h = int(img_w * scale),int(img_h * scale)

    image = image.resize((new_w,new_h),Image.BICUBIC) # 这样是不会失真的
    new_image = Image.new('RGB',input_shape,(128,128,128))
    new_image.paste(image,((input_w - new_w) // 2,(input_h - new_h) // 2))
    return new_image

def get_random_data(annotation_line,input_shape,max_boxes=100,jitter=0.3,hue=0.1,
    sat=1.5,val=1.5,random=True):
    # 随机预处理，包含data augumentation
    # annotation_line:2007_train的一条信息,input_shape:(416,416)
    line = annotation_line.split()
    image = Image.open(line[0]) # 第一个是图像的信息
    img_w,img_h = image.size # 说明PIL的是w先h后
    input_h,input_w = input_shape # (416,416)
    bboxes = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
    '''
        本身在line.split()之后，就已经形成了这个样子['路径','坐标1','坐标2']的样子，
        box也就是'坐标1','坐标2'这种，而这一步，是将这张图片里面的所有框，全部抽取了出来
        并搞成了(xmin,ymin,xmax,ymax,class_id)的样子，split(',')出来的是字符串，
        所以map一下转为int
    '''
    if not random: # 就没有shuffle
        '''
            下面是将图片resize，填充灰度条，因为不是所有图片都是正方形而且是416*416，
            所以会缩得比416*416小一些或刚好宽或高贴合，没填满的部分就填灰色，灰色就是
            (128,128,128)，之后就是将图片resize并将图片的bboxes坐标放缩到缩小的图片里
        '''
        scale = min(input_w / img_w,input_h / img_h)
        new_w,new_h = int(img_w * scale),int(img_h * scale)
        delta_w,delta_h = (input_w - new_w) // 2,(input_h - new_h) // 2
        # 除以2就是放在中间，不除以2就是放在边缘
        image = image.resize((new_w,new_h),Image.BICUBIC)
        new_image = Image.new('RGB',(input_w,input_h),(128,128,128))
        new_image.paste(image,(delta_w,delta_h)) # 在适当的位置粘贴图片，画图好理解
        image_data = np.array(new_image,np.float32) / 255.

        # 下面是将那些框也缩小，因为bboxes是在图片上而不是灰度图上的
        bboxes_data = np.zeros((max_boxes,5)) # 5:(xmin,ymin,xmax,ymax,class_id)
        if len(bboxes) > 0:
            np.random.shuffle(bboxes)
            bboxes[:,[0,2]] = bboxes[:,[0,2]] * new_w / img_w + delta_w
            bboxes[:,[1,3]] = bboxes[:,[1,3]] * new_h / img_h + delta_h
            bboxes[:,0:2][bboxes[:,0:2] < 0] = 0 # 独有的数组掩码，一维变两维
            bboxes[:,2][bboxes[:,2] > input_w] = input_w
            bboxes[:,3][bboxes[:,3] > input_h] = input_h
            '''
                [0,2]是xmin,xmax，对x在横轴方向上放缩，加上偏移量delta_w,[1,3]同理
                而后面那三行是将超出(416,416)边界的图片强行缩回去
            '''
            bboxes_w = bboxes[:,2] - bboxes[:,0]
            bboxes_h = bboxes[:,3] - bboxes[:,1]
            bboxes = bboxes[np.logical_and(bboxes_w > 1,bboxes_h > 1)]
            '''
                原理是这样的，首先bboxes_w和bboxes_h维度要一样，然后对每个一个单体进行
                逻辑与比较，比如bboxes_w[0] > 1 & bboxes_h[0] < 1,因为他有两个w和h，
                所以还有bboxes_w[0] > 1 & bboxes_h[0] < 1，就会有两个结果，
                [False,False]然后就是与的语义，> 1的意思是所有bbox宽高都要大于1，
                不然就是废的
            '''
            if len(bboxes) > max_boxes: bboxes = bboxes[:max_boxes]
            bboxes_data[:len(bboxes)] = bboxes
        return image_data,bboxes_data 
        # image_data注意是归一化了的，bboxes_data是(416,416)下的

    # 对图像进行缩放和宽高的扭曲，jitter的意思是扰动
    random_ratio = input_w / input_h * rand(1 - jitter,1 + jitter) / \
                   rand(1 - jitter,1 + jitter)
    scale = rand(0.25,2)
    if random_ratio < 1:
        new_h = int(scale * input_h)
        new_w = int(new_h * random_ratio)
    else: # 其实个人估计这里只是一种随机放缩的方式，并且让h和w不要差太远
        new_w = int(scale * input_w)
        new_h = int(new_w / random_ratio)
    image = image.resize((new_w,new_h),Image.BICUBIC)

    # 将图像多余的部分加上灰条
    delta_w = int(rand(0,input_w - new_w))
    delta_h = int(rand(0,input_h - new_h))
    new_image = Image.new('RGB',(input_w,input_h),(128,128,128)) # 是原图的大小！！
    new_image.paste(image,(delta_w,delta_h))
    image = new_image
    '''
        首先，new_w和new_h是有可能比416大的，其次delta_w和delta_h是不能确定的，
        再者，因为是灰度图上paste，所以有可能是缩得很小也有可能是某些部分看不见了
    '''

    # 翻转图像
    flip = rand() < 0.5
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # 色域扭曲，转成HSV再干这个，Hue,Saturation,Value,色度，包和度，亮度
    hue = rand(-hue,hue)
    sat = rand(1,sat) if rand() < 0.5 else 1 / rand(1,sat) # 饱和度
    val = rand(1,val) if rand() < 0.5 else 1 / rand(1,val)
    image_hsv = cv.cvtColor(np.array(image,np.float32) / 255.,cv.COLOR_RGB2HSV)
    image_hsv[:,:,0] += hue * 360
    image_hsv[:,:,0][image_hsv[:,:,0] > 1] -= 1
    image_hsv[:,:,0][image_hsv[:,:,0] < 0] += 1
    '''
        虽然转换成hsv格式，但是依然是(w,h,c)这样的格式，只是c变成了(h,s,v)而不是(r,g,b)
        hue是0~360°的，saturation是0~1，value也是0~1(黑到白)
    '''
    image_hsv[:,:,1] *= sat
    image_hsv[:,:,2] *= val
    image_hsv[image_hsv[:,:,0] > 360,0] = 360 # hue不能超出360，sat和val不能超过1
    image_hsv[:,:,1:][image_hsv[:,:,1:] > 1] = 1
    image_hsv[image_hsv < 0] = 0
    image_data = cv.cvtColor(image_hsv,cv.COLOR_HSV2BGR) # 这个是归一化的

    # 下面是将那些框也缩小，因为bboxes是在图片上而不是灰度图上的
    bboxes_data = np.zeros((max_boxes,5))  # 5:(xmin,ymin,xmax,ymax,class_id)
    if len(bboxes) > 0:
        np.random.shuffle(bboxes)
        bboxes[:,[0,2]] = bboxes[:,[0,2]] * new_w / img_w + delta_w
        bboxes[:,[1,3]] = bboxes[:,[1,3]] * new_h / img_h + delta_h
        if flip: # 左右翻转像素
            bboxes[:,[0,2]] = input_w - bboxes[:,[2,0]]
        bboxes[:,0:2][bboxes[:,0:2] < 0] = 0  # 独有的数组掩码，一维变两维
        bboxes[:,2][bboxes[:,2] > input_w] = input_w
        bboxes[:,3][bboxes[:,3] > input_h] = input_h
        bboxes_w = bboxes[:,2] - bboxes[:,0]
        bboxes_h = bboxes[:,3] - bboxes[:,1]
        bboxes = bboxes[np.logical_and(bboxes_w > 1, bboxes_h > 1)]
        if len(bboxes) > max_boxes: bboxes = bboxes[:max_boxes]
        bboxes_data[:len(bboxes)] = bboxes
    return image_data, bboxes_data  # image_data注意是归一化了的

def calc_IOU(bboxes1,bboxes2,is1_boolean_mask=False,is2_boolean_mask=False,
    for_CIOU=False):
    '''
        boxes1:(batch_size,h,w,num_bboxes,4),4:(x_center,y_center,w,h)
        boxes2:(valid_num_bboxes,4),4:(x_center,y_center,w,h)
        IOU:(batch_size,h,w,num_bboxes)
        batch_size可有可无
        这个IOU是专门为boolean_mask后的bbox设计的，需要对boolean_mask后的bbox
        进行扩维，没有boolean_mask的也要扩维
        或者说通用，如果两个都是True暂时不考虑
        如果是有一个为True，返回的shape就是IOU:(13,13,3,valid_num_bboxes)
        其实如果这两个都是(valid_num_bboxes,4)的时候也行
    '''
    '''
        扩维后的结果是：
        (batch_size,h,w,3,4) -> (batch_size,h,w,3,1,4)
        (valid_num_bboxes,4) -> (1,valid_num_bboxes,4)
        batch_size可有可无
    '''
    if is1_boolean_mask and not is2_boolean_mask:
        bboxes1 = tf.expand_dims(bboxes1,axis=0) # 第一维扩维
        bboxes2 = tf.expand_dims(bboxes2,axis=-2) # 倒数第二维扩维
    elif is2_boolean_mask and not is1_boolean_mask:
        bboxes2 = tf.expand_dims(bboxes2, axis=0)  # 第一维扩维
        bboxes1 = tf.expand_dims(bboxes1, axis=-2)  # 倒数第二维扩维
    elif is1_boolean_mask and is2_boolean_mask:
        raise AttributeError('situation of both bboxes being boolean masked'
                             'not supported')
    bboxes1_xy = bboxes1[...,:2] # 这些其实简写了，前两个False的时候应该是
    bboxes1_wh = bboxes1[...,2:4] # bboxes1[:,:,:,:,:2],bboxes1[:,:,:,:,2:4]
    bboxes2_xy = bboxes2[...,:2]
    bboxes2_wh = bboxes2[...,2:4]
    bboxes1_areas = bboxes1_wh[...,0] * bboxes1_wh[...,1]
    bboxes1_xymin,bboxes1_xymax = bboxes1_xy - (bboxes1_wh * 0.5), \
                                    bboxes1_xy + (bboxes1_wh * 0.5)
    bboxes2_areas = bboxes2_wh[...,0] * bboxes2_wh[...,1]
    bboxes2_xymin,bboxes2_xymax = bboxes2_xy - (bboxes2_wh * 0.5), \
                                    bboxes2_xy + (bboxes2_wh * 0.5)
    lb = tf.maximum(bboxes1_xymin,bboxes2_xymin)
    ub = tf.minimum(bboxes1_xymax,bboxes2_xymax)
    intersection_wh = tf.maximum(ub - lb,0.0)
    intersection = intersection_wh[...,0] * intersection_wh[...,1]
    IOU = tf.truediv(intersection,bboxes1_areas + bboxes2_areas - intersection)
    if for_CIOU:
        return IOU,bboxes1_xy,bboxes1_wh,bboxes2_xy,bboxes2_wh,bboxes1_xymin,\
            bboxes1_xymax,bboxes2_xymin,bboxes2_xymax
    return IOU

@tf.function
def nms(cur_class_coords,cur_class_scores,max_bboxes_tensor,
    iou_threshold=0.5):
    '''
        args:
        cur_class_coords:当前所有confs大于score_threshold的点的坐标,(valid_num,4)
        cur_class_scores:当前所有confs大于score_threshold的点的置信度,(valid_num,)
        这两个必须一一对应
        max_bboxes_tensor:最多有几个bounding boxes，必须是tensor类型
        return:
        大于iou_threshold的coords在cur_class_coords的下标
    '''
    idx_scores = [(i,s) for i,s in enumerate(cur_class_scores)]
    nms_idx = []
    sorted_idx_scores = sorted(idx_scores,key=lambda x:x[1],reverse=True)
    while len(sorted_idx_scores) != 0:
        best = sorted_idx_scores.pop(0)
        nms_idx.append(best[0]) # 只返回index，就把index传进去就好了
        if len(sorted_idx_scores) == 0:break
        best_coord = tf.expand_dims(cur_class_coords[best[0]],axis=0) # (1,4)
        cur_indices = [i[0] for i in sorted_idx_scores] # 把那个剔除掉了
        other_coords = tf.gather(cur_class_coords,cur_indices) 
        # (valid_num_bboxes - 1,4)
        IOU = calc_IOU(best_coord,other_coords) # (valid_num_bboxes - 1,)
        cur_valid_num = len(sorted_idx_scores)
        sorted_idx_scores = [sorted_idx_scores[i] for i in range(
            cur_valid_num) if IOU[i] < nms_threshold] # 下一次要用到的list
    return tf.convert_to_tensor(nms_idx)

# https://blog.csdn.net/weixin_35848967/article/details/108493217
def cosine_decay_with_warmup(cur_step,
                            lr_base,
                            total_steps,
                            min_lr,
                            warmup_init_lr=0.0,
                            warmup_steps=0,
                            hold_base_rate_steps=0):
    '''
        也就是说，余弦退火本质上是要让学习率下降的，但是他里面加入了一个warmup阶段，在这个
        阶段学习率是从小慢慢上升的，原因是一开始如果学习率过大的话会造成模型动荡，可以使得
        开始训练的几个epoch或者一些step内学习率较小，在预热的小学习率下，模型可以慢慢
        趋于稳定，等模型相对稳定后在选择预先设置的学习率进行训练，使得模型收敛速度
        变得更快，模型效果更佳，而之后确实是随着Tcur的增加，lr慢慢减小
        cur_step:公式中的Tcur，记录当前执行到第几步，每个batch都会更新，
            本来是执行到第几个epoch，Ti是sample_count / batch_size = batch_count,
            Ti = batch_count * epoch
        lr_base:learning_rate_base，预设置的学习率，当warmup阶段学习率增加到这个值，
            就开始学习率下降
        total_steps:即Ti，总的训练的步数，等于epoch * sample_count / batch_size,
            (sample_count是样本总数，epoch是总的循环次数)
        min_lr:min_learning_rate，最低的学习率，公式里有
        warmup_lr:warmup_learning_rate，warmup阶段线性增长的初始值
        warmup_steps:warmup总的需要持续的步数，warmup_epoch * sample_count / batch_size
        hold_base_rate_steps:可选参数，即当warmup阶段结束后保持学习率不变，直到
            hold_base_rate_steps结束后才开始学习率下降
    '''
    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to warmup_steps.')
    new_lr = 0.5 * lr_base * (1 + np.cos(np.pi * (cur_step - warmup_steps - 
        hold_base_rate_steps) / float(total_steps - warmup_steps -
        hold_base_rate_steps))) # 这里实现了余弦退火公式，只不过简化设置了nmin=0
    # warmup_steps是在余弦退火之前的，所以Tcur和Ti都要减掉，hold_base_rate_steps虽说
    # 可选，但是有可能会hold on一下，所以还是要减，不过默认是0

    if hold_base_rate_steps > 0: # warmup后学习率在一定步数内保持不变
        new_lr = np.where(cur_step > warmup_steps + hold_base_rate_steps,
            new_lr,lr_base)
        # 跟tf.where一样，满足条件，输出前者，不满足输出后者，就此时lr可能已经开始退火了
        # 所以不应该退火，应该回到没有退货之前即learning_rate_base

    if warmup_steps > 0:
        if lr_base < warmup_init_lr:
            raise ValueError('lr_base must be larger or equal to warmup_init_lr.')
        # warmup阶段的线性增长实现
        slope = (lr_base - warmup_init_lr) / warmup_steps
        warmup_cur_lr = slope * cur_step + warmup_init_lr
        # 只有当global_step依然处于warm_up阶段才会使用线性增长的学习率warmup_cur_lr,
        # 否则使用余弦退火的学习率learning_rate
        new_lr = np.where(cur_step < warmup_steps,warmup_cur_lr,new_lr)

    new_lr = max(new_lr,min_lr)
    return new_lr

class WarmupCosineDecayScheduler(tf.keras.callbacks.Callback):
    # 学习率下降
    def __init__(self,
                learning_rate_base, # warmup的停止学习率，也是余弦退火的初始学习率
                total_steps, # 即Ti，总的训练的步数
                cur_step_init=0, # Tcur的初始化
                warmup_init_learning_rate=0.0, # warmup步骤的初始化lr
                warmup_steps=0, # warmup总共有多少步
                hold_base_rate_steps=0,
                min_learning_rate=0, # nmin
                interval_epoch=[0.05,0.15,0.30,0.50],
                verbose=0):
        # interval_epoch代表余弦退火之间的最低点
        super().__init__()
        self.lr_base = learning_rate_base
        self.warmup_init_lr = warmup_init_learning_rate
        self.verbose = verbose # 参数显示
        self.min_lr = min_learning_rate
        self.lr_record = [] # 记录每次更新后的学习率，方便图形化观察
        self.interval_epoch = interval_epoch # 间隔次数
        self.cur_step_for_interval = cur_step_init # 当前到了第几个step
        self.warmup_steps_for_interval = warmup_steps # warmup阶段的总step
        self.hold_steps_for_interval = hold_base_rate_steps # warmup后停一阵子的步数
        self.total_steps_for_interval = total_steps # 整个训练的总step，公式的Ti
        self.interval_index = 0

        # 计算出来的每两个最低点各自的间隔，就是差值
        self.interval_gap = [self.interval_epoch[0]] 
        for i in range(len(self.interval_epoch) - 1):
            self.interval_gap.append(self.interval_epoch[i + 1] - 
                self.interval_epoch[i])
        self.interval_gap.append(1 - self.interval_epoch[-1])

    # on_batch_begin()和on_batch_end()应该是会由模型自动调用的
    # 更新学习率
    def on_batch_begin(self,batch,logs=None):
        # 每到一次最低点就重新更新参数
        if self.cur_step_for_interval in [0] + [int(i * 
            self.total_steps_for_interval) for i in self.interval_epoch]:
            self.total_steps = self.total_steps_for_interval * \
                self.interval_gap[self.interval_index]
            self.warmup_steps = self.warmup_steps_for_interval * \
                self.interval_gap[self.interval_index]
            self.hold_base_rate_steps = self.hold_steps_for_interval * \
                self.interval_gap[self.interval_index]
            self.cur_step = 0
            self.interval_index += 1

        new_lr = cosine_decay_with_warmup(
                cur_step=self.cur_step,
                lr_base=self.lr_base,
                total_steps=self.total_steps,
                min_lr=self.min_lr,
                warmup_init_lr=self.warmup_init_lr,
                warmup_steps=self.warmup_steps,
                hold_base_rate_steps=self.hold_base_rate_steps
            )
        K.set_value(self.model.optimizer.lr,new_lr)
        if self.verbose > 0:
            print('\nBatch %05d: setting learning rate to %s.' % 
                (self.cur_step + 1,lr))

    # 更新global_step，并记录当前学习率
    def on_batch_end(self,batch,logs=None): 
        self.cur_step += 1 # 一个batch结束了，cur_step就加1
        self.cur_step_for_interval += 1 # 这个好像没啥用
        lr = K.get_value(self.model.optimizer.lr)
        self.lr_record.append(lr)

class ModelCheckpoint(keras.callbacks.Callback):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super().__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto','min','max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, fallback to auto mode.' 
                % (mode),RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self,epoch,logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1,**logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor),RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s' % (epoch + 1,self.monitor,
                                    self.best,current,filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath,overwrite=True)
                        else:
                            self.model.save(filepath,overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve' % 
                                (epoch + 1,self.monitor))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath,overwrite=True)
                else:
                    self.model.save(filepath,overwrite=True)