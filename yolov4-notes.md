# 1.GIOU代替了IOU

原因是：

（1）IOU有可能为0，此时无法求梯度

（2）IOU对于两物体的形状不敏感，因为他只是面积的比值，有时候有的框可能很离谱但是因为重叠的面积一样，跟那些比较正常的框区分不出来

而GIOU就一定程度上解决了两个痛点，因为
$$
GIOU = IOU - (C-A∪B)/C\\
L_{GIOU}=1-GIOU
$$
IOU为0的时候，GIOU为负，他的值域是[-1,1]，很少出现为0的情况

*GIOU的缺点：一个框被另一个完全包含时，GIOU=IOU，无法区分，所以很有可能出现中心根本不接近的情况*

2.DIOU加上了对中心距离的惩罚

3.CIOU加上了对长宽比例的惩罚，所以相当于对xywh都利用上了，可以代替原来的MSE
$$
L_{CIOU}=1-IOU+d^2/c^2+αv\\
v=4/PI^2*(arctanh(w^{gt}/h^{gt})-arctanh(w/h))\\
α=v/(1-IOU + v)
$$

4.最好全部用tensor不要numpy

5.K.reshape蜜汁操作，K.shape也是

6.float32和float64和int32互不兼容

7.

```python
coords_scale = 2 - y_true[layer][:,:,:,:,2:3] * y_true[layer][:,:,:,:,3:4]
```

```python
coords_scale = 2 - y_true[layer][:,:,:,:,2] * y_true[layer][:,:,:,:,3]
```

这两种写法看似一样，但是第一种会保持最后一个维度，第二种不会！

8. shape和dtype，学会用keras的东西，不要.dtype

```python
input_shape = tf.cast(K.shape(y_true[0])[1:3] * 32,dtype=K.dtype(y_true[0]))
# 也就是h,w = 416,416
batch_size = tf.cast(K.shape(y_pred[0])[0],dtype=K.dtype(y_pred[0]))
```

9.reduce_max的参数叫keepdims不是keep_dims

10.如果ndarray的shape不一样，append不会多一维，比如(113,123,3)和(321,325,3)，合在一起就是(2,)，只有(416,416,3)和(416,416,3)合在一起才是(2,416,416,3)

11.tensor不能对10里面前面那种转换，后面可以

# 12.loss={'yolov4_loss':lambda y_true, y_pred:y_pred}

```python
model_loss = Lambda(loss,output_shape=(1,),name='yolov4_loss',
    arguments={'y_pred':y_pred,'anchors':anchors,'num_classes':\
    num_classes,'label_smoothing_kesi':label_smoothing_kesi})(loss_input)
model.compile(optimizer=Adam(),loss={'yolov4_loss':lambda y_true,
    y_pred:y_pred})
```
```python
	首先，我们知道的是yolov4_loss在model_loss定义的时候只是一个name,所以这里这种字典的形式估计也只是一个名字，就把这个Model对应的loss叫这个名字而已，因为发现没有'yolov4_loss'根本没区别，但是问题是，你不叫yolov4_loss就不行，会报这个错：
	Found unexpected keys that do not correspond to any Model output
代码里也说使用定制的yolov4_loss层，所以有可能是这样：如果有字典的出现，那么就会自动去匹配字典的key是不是跟输出层的name（如果有name的话）相同，因为他认为‘嗯，我就要叫你这个名字的输出层来计算loss’，不同就报上面那个错如果是相同的，那么就用输出层的输出作为y_pred来计算loss，所以这个也对应了作者说用特制化的Lambda层来计算的说法，如果没有字典，就直接lambda y_true, y_pred: y_pred的话应该就是默认用模型的output来计算loss，只不过少了‘强制匹配’的步骤，而loss本身是个函数！是个函数！匿名函数也是函数，而这个函数怎么运作的就是下面说的

	因为在上面的时候，他将loss变成了Lambda一个层，然后定义Model的输出层为这个Lambda loss层，也就是跟平常输出预测结果不一样，是直接输出loss，是直接输出loss的
	而因为这是keras的Model，所以看keras是怎么写损失函数的，每个损失函数都是输入y_true,y_pred然后return某种loss，比如：
def mean_squared_error(y_true, y_pred):
	return K.mean(K.square(y_pred - y_true), axis=-1)
那么在Model里面他也肯定是这样的，有y_true,y_pred和返回的loss，但是！
但是y_pred其实是模型的输出，正常来说是输出一个标签之类的，而这里是直接将模型的最后输出(即y_pred)设定为loss，也就是说我传进去的时候就是loss而不是y_pred了，所以返回y_pred就等于直接返回loss了！关于输入给keras定义的损失函数的y_true，个人理解就是y_train的(训练集label或者叫y_true)，而这里根本就不用管，就直接输出y_pred也就是loss了
	
	所以综合loss='yolov4_loss':lambda y_true,y_pred:y_pred来看，意思就是：我这里的loss要用这个函数来计算：
def keras_style_loss(y_true,model_loss):
	return model_loss # model_loss为那个Lambda层的输出
所以在这里他就会强制查找'yolov4_loss'这个name，原因是输出层就叫这个，那么找到了之后就使用model_loss这个Lambda层来计算loss，计算的原则就如上所说。

实验证明，lambda y_true, y_pred: y_pred和{'yolov4_loss':lambda y_true, y_pred: y_pred}根本没差
```

13.如果说不能用symbolic tensor，加这一句

```python
tf.config.experimental_run_functions_eagerly(True)
```

14.要很注意shape，anchors和intersection_wh都是三维的，扩维了

15.yield等迭代器的输出要看的话要这么看：

```python
from train import data_generator,get_anchors
annotation_path = '2007_train.txt'
anchors_path = './model_data/yolo_anchors.txt'
anchors = get_anchors(anchors_path)
with open(annotation_path) as f:
    lines = f.readlines()
num_train = len(lines)
batch_size = 2
input_shape = 416,416
num_classes = 20
mosaic = False
a = data_generator(lines[:num_train],batch_size,input_shape,
    anchors,num_classes,mosaic=mosaic,random=True)
for i in a: # 这样看
    print(i[0][0].shape)
    print(i[0][1].shape)
    # print(i[0][1])
    mask = np.unique(i[0][1])
    tmp = []
    for v in mask:
        tmp.append(np.sum(i[0][1] == v))
    ts = np.max(tmp)
    max_v = mask[np.argmax(tmp)]
    print(f'这个值：{max_v}出现的次数最多，为{ts}次')
```

# 16.tf很奇怪的坑，tf.Tensor不会直接乘以数值的大小而不是扩维

```python
grid_shape = tf.constant([13,13])
num_anchors = 3
print([K.arange(0,stop=grid_shape[0])])
print(tf.shape([K.arange(0,stop=grid_shape[0])] * grid_shape[1] * num_anchors))
print(tf.shape(tf.reshape([K.arange(0,stop=grid_shape[0])] * grid_shape[1] * num_anchors,
    (num_anchors,grid_shape[0],grid_shape[1]))))
    
'''
第二个的输出结果是这样的，tf.Tensor([ 1 13], shape=(2,), dtype=int32)，
如果grid_shape=(13,13),输出结果应该是tf.Tensor([ 39 13], shape=(2,), dtype=int32)，
说明是直接乘进去的而不是像list那样多几个元素
第三个直接报错：
Input to reshape is a tensor with 13 values, but the requested shape has 507 [Op:Reshape]
'''

grid_shape = tf.constant([13,13])
num_anchors = 3
print([K.arange(0,stop=grid_shape[0])])
print(tf.shape(tf.tile([K.arange(0,stop=grid_shape[0])],[grid_shape[1] * num_anchors,1])))
print(tf.shape(tf.reshape(tf.tile([K.arange(0,stop=grid_shape[0])],[grid_shape[1] * num_anchors,1]),
    (num_anchors,grid_shape[0],grid_shape[1]))))
'''
这样第二个输出就正常了，tf.Tensor([ 39 13], shape=(2,), dtype=int32)，
第三输出是tf.Tensor([ 3 13 13], shape=(2,), dtype=int32)，
'''

所以x_origin要这样写：
x_origin = tf.cast(tf.expand_dims(tf.transpose(tf.reshape(tf.tile(
    [K.arange(0,stop=grid_shape[0])],[grid_shape[1] * num_anchors,1]),
    (num_anchors,grid_shape[0],grid_shape[1])),(1,2,0)),axis=0),dtype=tf.float32)
```

# 17.关于keras.models.Model的fit的数据问题：

（1）我们之前用model.fit()的时候，数据的预处理是将x_train和y_train绑定在一起变成train_data的，当时用的是库自己的loss函数。所以如果自己要写损失函数，也需要把数据的格式变成train_data这样的格式化，这也解释了为什么作者写的loss里用args包含y_pred和y_true，而不是直接y_pred和y_true，

（2）至于损失函数怎么识别谁是y_pred和y_true，他是按顺序的，前面的是y_pred，后面的是y_true。

（3）再关于他是怎么识别x_train和y_train的，其实很简单，data_generator的前半部分是img_data后半部分是y_true，他只要看前半部分有没有符合input_shape:(416,416)的输入就可以，有就直接拿，后半部分的直接认为是y_true，然后输入模型获取输出作为y_pred传进loss里计算

（4）所以说，一开始将y_pred写在model_loss的argument那里，他最后会将model的output即y_pred和y_true一起打包输进loss，而这时接收这个打包的是y_true，y_pred根本没有东西（或者有东西，因为一开始定义了model.output给y_pred，但是更大可能是没有，一开始那个只是个壳），所以输入给loss的数据就不符合规范了，所以loss就输出0
