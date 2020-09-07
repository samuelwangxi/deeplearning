import pandas as pd
import tensorflow as tf
from tensorflow import keras
import math
from word2vec.recommend.InterFace import *
import numpy as np
import re
from word2vec.w2v import *
# digit_inputs.csv 是7分类
from tensorflow.keras import Sequential, layers, optimizers, losses

raw_data = pd.read_csv('data/digit_inputs_5.csv')
# print(raw_data)
# print(raw_data.describe())
# print(raw_data.isna().sum())
dataset = raw_data.dropna()
# print(dataset.isna().sum())
# print(dataset.describe())

train_set = dataset.sample(frac=0.8, random_state=0)
test_set = dataset.drop(train_set.index)

train_labels = train_set.pop('industry')
test_labels = test_set.pop('industry')
print(train_labels)


# 把数据合成104维作为输入数据
def get_traindata(train_set):
    all = np.c_[train_set['etapes'], train_set['workage'], train_set['skills_num'], \
                train_set['interests_num']]
    interet_vec = train_set['interest_vec']
    skill_vec = train_set['skills_vec']
    interet_vec = process_str2vec(interet_vec)
    skill_vec = process_str2vec(skill_vec)
    # specilities = process_str2vec(train_set['specilities'])
    all = np.c_[all, interet_vec, skill_vec]
    return all


# 把string转换成nparray
def process_str2vec(li_str):
    new = []
    for e in li_str:
        new_e = re.sub(r'\n|\[|\]|\'', '', e).split(' ')
        new_e = [float(i) for i in new_e if i]
        new.append(new_e)
    return np.array(new, dtype=float)


def norm(x):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    return (x - mean) / std


train_set = norm(get_traindata(train_set))
test_set = norm(get_traindata(test_set))
print(train_set)

train_db = tf.data.Dataset.from_tensor_slices((train_set, process_str2vec(train_labels)))
train_db = train_db.shuffle(100).batch(20)
test_db = tf.data.Dataset.from_tensor_slices((test_set, process_str2vec(test_labels.values)))
test_db = test_db.shuffle(100).batch(600)


class WideDeepModel(keras.models.Model):
    def __init__(self):
        super(WideDeepModel, self).__init__()
        # 定义模型层次
        # self.hidden0_layer = keras.layers.Dense(256, activation='relu')
        self.hidden1_layer = keras.layers.Dense(128, activation='relu')
        self.hidden2_layer = keras.layers.Dense(64, activation='relu')
        self.hidden3_layer = keras.layers.Dense(32, activation='relu')
        self.output_layer = keras.layers.Dense(5,activation='sigmoid')
        self.dropout_layer1 = keras.layers.Dropout(rate=0.2)
        self.dropout_layer2 = keras.layers.Dropout(rate=0.5)

    def call(self, input):
        # 完成模型的正向计算
        # hidden0 = self.hidden0_layer(input)
        # hidden0 = self.dropout_layer2(hidden0)
        hidden1 = self.hidden1_layer(input)
        hidden1 = self.dropout_layer2(hidden1)
        hidden2 = self.hidden2_layer(hidden1)
        hidden2 = self.dropout_layer1(hidden2)
        hidden3 = self.hidden3_layer(hidden2)
        hidden3 = self.dropout_layer1(hidden3)
        concat = keras.layers.concatenate([input,hidden3]) #tf.add(input[:,4:54],input[:,54:104])/2
        output = self.output_layer(concat)
        return output

model = WideDeepModel()
model.build(input_shape=(20,104))
model.summary()
model.compile(optimizer=optimizers.Adam(lr=0.001),
                loss='MSE',
                metrics=['categorical_accuracy'])

# history = model.fit(train_db,
#                     epochs=50,
#                     validation_data=test_db,
#                     validation_freq=2)
# out = model(train_set)
# print(out)
# model.save_weights('weight/weight.ckpt')
model.load_weights('weight/weight.ckpt')
[loss,accuracy] = model.evaluate(test_db)
print(loss,accuracy)


def result(out):
    n = 5
    out_ord = {}#归一化结果
    total = sum(out[0])
    for i in range(5):
        out_ord[i] = math.floor(10*(out[0][i]/total))
        # out_ord[i] = 5*(out[0][i]/total)
    out_ord = dict(sorted(out_ord.items(),key= lambda x:x[1],reverse=True))
    return out_ord



ex = {'etapes': 1,
      'workage': 12,
      'skills_num': 26,
      'skills': ['Web Development', 'Program Development', 'Resource Management',
                                                               'Veterans', 'Military', 'Web Content', 'Career Counseling',
                                                               'Career Development', 'Creative Writing', 'Resume Writing',
                                                               'Grant Administration', 'Project Management', 'Instructional Videos',
                                                               'Coaching', 'Educational Leadership', 'Social Media',
                                                               'Social Media Marketing', 'Corporate Social Responsibility',
                                                               'Social Media Consulting', 'Linked Data', 'linkedin consulting',
                                                               'Non-profits', 'Not for Profit', 'Marketing for Small Business',
                                                               'Social Networking', 'Business Networking'],
      'interests_num': 4, 'interests': ['Lately', ' in addition to my work to develop new networking and business opportunities for clients and businesses',
                                        " I've been getting back in touch with my writing",
                                        ' something I love.']}
def yanshi(ex):
    interest = get_listVector(ex['interests'])
    skill = get_listVector(ex['skills'])
    out = np.r_[ex['etapes'],ex['workage'],ex['skills_num'],ex['interests_num'],interest,skill]
    return np.reshape(out,(1,104))

out = model(yanshi(ex))
print(out)
out_ord = result(out)
print(out_ord)
interface(out_ord)