import tensorflow as tf
from tensorflow.keras.layers import Conv2D,BatchNormalization,Layer,ZeroPadding2D,\
    Add,Concatenate,LeakyReLU
from tensorflow import constant_initializer,random_normal_initializer
from functools import reduce

class Mish(Layer): # 结构不同的第一点，Leaky relu换成了Mish
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
    def call(self,x):
        return x * tf.math.tanh(tf.math.softplus(x))
        # softplus就是ln(1+e^x)
    def get_config(self):
        config = super().get_config()
        return config
    def compute_output_shape(self,input_shape):
        return input_shape

def compose(*funcs):
    if funcs:
        # reduce(lambda f,g:g(f()),funcs),套娃,比如BatchNormalization(Conv2D(x))
        return reduce(lambda f,g:lambda *args,**kwargs:g(f(*args,**kwargs)),funcs)
    else:
        raise ValueError('composition of empty sequence is not supported')

def DBL(filters,kernel_size,strides=1,use_bias=False):
    padding = 'valid' if strides == 2 else 'same'
    return compose(Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,
                          padding=padding,use_bias=use_bias,
                          kernel_initializer=random_normal_initializer(stddev=0.01)),
                   BatchNormalization(),
                   LeakyReLU(alpha=0.1))

def DBM(filters,kernel_size,strides=1,use_bias=False):
    # DBM：DarkNetConv BN Mish l2正则化效果反而变差，不用
    padding = 'valid' if strides == 2 else 'same'
    return compose(Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,
                          padding=padding,use_bias=use_bias,
                          kernel_initializer=random_normal_initializer(stddev=0.01)),
                   BatchNormalization(),
                   Mish())

def residual_block(x,filters,num_blocks,half=True):
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DBM(filters,3,strides=2)(x)
    # 传统先用个Zeropadding和DBM，v3也有类似的实现
    skip = DBM(filters // 2 if half else filters,1)(x)
    '''
    CSPDarkNet的结构不同的第二点，有个很大的skip绕过所有的resnet层，最后再concat，
    跟DenseNet有点像，这个是CSPNet的作用
    '''
    x = DBM(filters // 2 if half else filters,1)(x) # 原作者就是这么设计的！
    for i in range(num_blocks):
        # //2是对的，residual block的第一层是比第二层少一半的输出通道数
        y = compose(DBM(filters // 2,1),
                    DBM(filters // 2 if half else filters,3))(x)
        x = Add()([x,y]) # 这个就是典型的residual残差设计了
    x = DBM(filters // 2 if half else filters,1)(x)
    # 他加入了好多1*1卷积来整合通道数
    route = Concatenate()([x,skip]) # 大残差边整合回来
    return DBM(filters,1)(route)

def CSPDarkNet53(x): # 加上SPP和PANet叫做FPN
    x = DBM(32,3)(x) # 正常
    x = residual_block(x,64,1,half=False)
    x = residual_block(x,128,2)
    x = residual_block(x,256,8)
    route1 = x
    x = residual_block(x,512,8)
    route2 = x
    x = residual_block(x,1024,4)
    return route1,route2,x
