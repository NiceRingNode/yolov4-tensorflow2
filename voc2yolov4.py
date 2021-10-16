import os,random
random.seed(0)

'''
这个文件的作用，是将5011条数据全部写成txt文件，对应xml文档的命名，
下载下来的train.txt是只有2500条数据的
'''

xml_path = r'./VOCdevkit/VOC2007/Annotations'
save_path = r'./VOCdevkit/VOC2007/ImageSets/Main/'
#xml_path = r'./VOCdevkit/VOC2012/Annotations'
#save_path = r'./VOCdevkit/VOC2012/ImageSets/Main/'

train_val_ratio,train_ratio = 1,1

temp_xml = os.listdir(xml_path)
tot_xml = []
for xml in temp_xml:
    if xml.endswith('.xml'):
        tot_xml.append(xml)
num = len(tot_xml)
num_list = range(num)
tvr = int(num * train_val_ratio)
tr = int(tvr * train_ratio)
train_val_index = random.sample(num_list,tvr) # 返回指定数量，不会更改顺序
train_index = random.sample(train_val_index,tr)

print("train and val size", tvr)
print("train size", tr)

file_train_val = open(os.path.join(save_path,'my_trainval.txt'),'w')
file_test = open(os.path.join(save_path,'my_test.txt'),'w')
file_train = open(os.path.join(save_path,'my_train.txt'),'w')
file_val = open(os.path.join(save_path,'my_val.txt'),'w')

for i in num_list:
    name = tot_xml[i][:-4] + '\n'
    # tot_xml[i]是009926.xml这样子的，一直到倒数第四个就是名字
    if i in train_val_index:
        file_train_val.write(name)
        if i in train_index:
            file_train.write(name)
        else:
            file_val.write(name)
    else:
        file_test.write(name)

file_train_val.close()
file_train.close()
file_val.close()
file_test.close()