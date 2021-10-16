from tensorflow.keras.layers import Conv2D,BatchNormalization,Layer,ZeroPadding2D,\
    Add,Concatenate,MaxPool2D,UpSampling2D
from tensorflow.keras.models import Model
from tensorflow import constant_initializer,random_normal_initializer
from CSPDarkNet53 import CSPDarkNet53,compose,DBM,DBL

def DBL_for_five_times(filters):
    return compose(DBL(filters,1),DBL(filters * 2,3),
                   DBL(filters,1),DBL(filters * 2,3),
                   DBL(filters,1))

def yolov4_net(x,num_anchors,num_classes):
    # route1:[52,52,256],route2:[26,26,512],route3:[13,13,1024]
    route1,route2,route3 = CSPDarkNet53(x)
    # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 ->
    # 13,13,2048 -> 13,13,512 -> 13,13,1024 -> 13,13,512
    route3 = DBL(512,1)(route3) # 改了，原来写了DBM
    route3 = DBL(1024,3)(route3)
    route3 = DBL(512,1)(route3)
    # SPP就是下面四行，即不同尺度的最大池化后合并在一起
    maxpool1 = MaxPool2D(pool_size=13,strides=1,padding='same')(route3)
    maxpool2 = MaxPool2D(pool_size=9,strides=1,padding='same')(route3)
    maxpool3 = MaxPool2D(pool_size=5,strides=1,padding='same')(route3)
    route3 = Concatenate()([maxpool1,maxpool2,maxpool3,route3])
    # SPP concat后又来3次卷积
    route3 = DBL(512,1)(route3)
    route3 = DBL(1024,3)(route3)
    route3 = DBL(512,1)(route3)

    # PANet开始
    # 首先是最左边往上那条路，直到downsample停下，输出的是route1
    # 这里是route3,(13,13,512)->(13,13,256)->(26,26,256)
    route3_upsample = compose(DBL(256,1),UpSampling2D(2))(route3)
    # 这里是route2,(26,26,512)->(26,26,256),因为维度要匹配
    route2 = DBL(256,1)(route2)
    # (26,26,256)+(26,26,256)->(26,26,512)
    route2 = Concatenate()([route2,route3_upsample])
    # (26,26,512)->(26,26,256)->(26,26,512)->(26,26,256)->(26,26,512)->(26,26,256)
    route2 = DBL_for_five_times(256)(route2)
    # (26,26,256)->(26,26,128)->(52,52,128),这里是将route2继续上采样,也就是将宽高变2倍
    route2_upsample = compose(DBL(128,1),UpSampling2D(2))(route2)
    route1 = DBL(128,1)(route1)
    # (52,52,128)+(52,52,128)->(52,52,256)
    route1 = Concatenate()([route1,route2_upsample])
    # (52,52,256)->(52,52,128)->(52,52,128)->(52,52,256)->(52,52,128)
    route1 = DBL_for_five_times(128)(route1)
    # 至此，可以输出route1了
    route1_output = DBL(256,3)(route1) # (batch_size,52,52,3,85)
    route1_output = Conv2D(filters=num_anchors * (num_classes + 5),
           kernel_size=1,strides=1,padding='same',
           kernel_initializer=random_normal_initializer(stddev=0.01),
           bias_initializer=constant_initializer(0.))(route1_output)
    # 这里相当于Dense

    # 其次，这里是route1输出并联支路的downsample，输出的是route2
    # (52,52,128)->(26,26,256),其实downsample是在DBL的strides=2那里
    route1_downsample = ZeroPadding2D(((1,0),(1,0)))(route1)
    route1_downsample = DBL(256,3,strides=2)(route1_downsample)
    # (26,26,256)+(26,26,256)->(26,26,512)
    route2 = Concatenate()([route1_downsample,route2])
    # (26,26,512)->(26,26,256)->(26,26,512)->(26,26,256)->(26,26,512)->(26,26,256)
    route2 = DBL_for_five_times(256)(route2)
    # 至此，可以输出route2了
    route2_output = DBL(512,3)(route2)  # (batch_size,52,52,3,85)
    route2_output = Conv2D(filters=num_anchors * (num_classes + 5),
           kernel_size=1,strides=1,padding='same',
           kernel_initializer=random_normal_initializer(stddev=0.01),
           bias_initializer=constant_initializer(0.))(route2_output)
    # 最后输出的是最底层的路，route1
    # (26,26,256)->(13,13,512)
    route2_downsample = ZeroPadding2D(((1,0),(1,0)))(route2)
    route2_downsample = DBL(512,3,strides=2)(route2_downsample)
    # (13,13,512)+(13,13,512)->(13,13,1024)
    route3 = Concatenate()([route2_downsample,route3])
    # (13,13,1024)->(13,13,512)->(13,13,1024)->(13,13,512)->(13,13,1024)->(13,13,512)
    route3 = DBL_for_five_times(512)(route3)
    # 至此，可以输出route3了，全部都完了
    route3_output = DBL(1024,3)(route3)
    route3_output = Conv2D(filters=num_anchors * (num_classes + 5),
            kernel_size=1,strides=1,padding='same',
            kernel_initializer=random_normal_initializer(stddev=0.01),
            bias_initializer=constant_initializer(0.))(route3_output)
    return Model(x,[route3_output,route2_output,route1_output])
    # 这里的三个output其实就是对应yolo3的输出，output都是先3*3卷积之后再1*1卷积再输出，
    # 也顺便变成了num_anchors * (num_classes + 5)的输出格式
    # 这里的最终输出是[(13,13,3,25),(26,26,3,25),(52,52,3,25)]