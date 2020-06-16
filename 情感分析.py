# 情感分析技术
# 1 训练或载入一个词向量生成模型
# encoding: utf-8
import numpy as np
word_list = np.load('.npy')
print('载入文本列表')
word_list=word_list.tolist()
word_list=[word.decode('UTF-8') for word in word_list]
word_vector=np.load('.npy')
print('载入文本向量')
# 检查
home_index = word_list.index('Home')
word_vector(home_index)

# 2 创建一个用于训练集的ID矩阵
# 首先确定文件长度分布
import os
from os.path import isfile, join
pos_files = ['pos/' + f for f in os.listdir('pos/') if isfile(join('pos/', f))]
neg_files = ['neg/' + f for f in os.listdir('neg/') if isfile(join('neg/', f))]

l = []
for pf in pos_files:
    with open(pf, 'r', encoding='UTF-8')as f:
        temp_sentence = f.readline()
        counter = len(temp_sentence)
        l.append(counter)
print('正面评价完毕')

for nf in neg_files:
    with open(nf, 'r', encoding='UTF-8') as f:
        temp_sentence = f.readline()
        counter = len(temp_sentence)
        l.append(counter)
print('负面评价完毕')

num_files = len(l)
num_words = sum(l)
# 可视化（省略），根据文本长度的分布，确定max_seq_len
# 接下来将文本生成索引矩阵
import re
def cleansentence(string):
    special_char = re.compile('[^A-Za-z0-9]+')
    string = string.lower().replace('<br />'," ")
    return re.sub(special_char," ",string.lower())
file_count= 0
max_seq_num = 300
ids =np.zeros((num_files,max_seq_num),dtype='int32')

for pf in pos_files:
    with open(pf, "r", encoding='UTF-8')as f:
        indexCounter = 0
        line = f.readline()
        cleanLine = cleansentence(line)
        split = cleanLine.split()
        for word in split:
            try:
                ids[file_count][indexCounter] = word_list.index(word)
            except ValueError:
                ids[file_count][indexCounter] = 399999
            indexCounter += 1
            if indexCounter == max_seq_num:
                break
        file_count += 1

for np in neg_files:
    with open(np, 'r', encoding='UTF-8') as f:
        line = f.readline()
        line = line.cleansentence()
        split = line.split()
        for word in split:
            try:
                ids[file_count][indexCounter] = word_list.index(word)
            except ValueError:
                ids[file_count][indexCounter] = 39999
            indexCounter += 1
            if indexCounter == max_seq_num
                break
        file_count += 1

np.save('idsMatrix',ids)

# 3 创建LSTM计算单元
# 辅助函数 返回一批训练/测试集合
from random import randint
batch_size = 24
def get_train_batch():
    labels=[]
    arr=np.zeros([batch_size],[max_seq_num])
    for i in range(batch_size):
        if(i%2 == 0):
            num = randint(1,11499)
            labels.append([1,0])
        else:
            num = randint(13499,24999)
            labels.append([0,1])
        arr[i] =ids[num-1:num]
    return arr, labels

def get_test_batch():
    labels=[]
    arr = np.zeros([batch_size,max_seq_num])
    for i in batch_size:
        if (i%2==0):
            num=randint(11499,12499)
            labels.append=([1,0])
        else:
            num=randint(12500,12499)
            labels.append=([0,1])
        arr[i]=ids[num-1:num]
    return arr, labels












# 4 训练
# 5 测试


