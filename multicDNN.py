

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.layers import Conv2D,MaxPooling2D
from tensorflow.keras import backend as K
from sklearn.metrics import roc_curve, auc
K.clear_session()
data=np.loadtxt(r"E:\2025年工作材料\杜国梁老师合作文件\new\train_data.txt")
indices=np.random.permutation(len(data))
train_size=int(len(data)*0.8)
X=data[:,0:10]
y=data[:,10]
data_train=X[indices[:train_size]]
label_train=y[indices[:train_size]]
data_test=X[indices[train_size:]]
label_test=y[indices[train_size:]]

# data_train=data[0:8267,0:10]
# label_train=tf.constant(data[0:8267,10])

c0_train=tf.constant(data_train[:,0])    
r1_train=tf.constant(data_train[:,1])
c2_train=tf.constant(data_train[:,2])
r3_train=tf.constant(data_train[:,3])
# r4_train=tf.constant(data_train[:,4])
r5_train=tf.constant(data_train[:,4])
r6_train=tf.constant(data_train[:,5])
r7_train=tf.constant(data_train[:,6])
# r8_train=tf.constant(data_train[:,8])
c9_train=tf.constant(data_train[:,7])
r10_train=tf.constant(data_train[:,8])
c11_train=tf.constant(data_train[:,9])


c0=Input(shape=(1))
r1=Input(shape=(1))
c2=Input(shape=(1))
r3=Input(shape=(1))
# r4=Input(shape=(1))
r5=Input(shape=(1))
r6=Input(shape=(1))
r7=Input(shape=(1))
# r8=Input(shape=(1))
c9=Input(shape=(1))
r10=Input(shape=(1))
c11=Input(shape=(1))


c0=Dense(8,activation='sigmoid')(c0)
c0=Dense(16,activation='sigmoid')(c0)
c0=Dense(32,activation='sigmoid')(c0)
c0=Dense(16,activation='sigmoid')(c0)
c0=Dense(1,activation='sigmoid')(c0)
output0=tf.keras.layers.BatchNormalization()(c0)
model0=Model(inputs=c0,outputs=output0)

r1=Dense(8,activation='sigmoid')(r1)
r1=Dense(16,activation='sigmoid')(r1)
r1=Dense(32,activation='sigmoid')(r1)
r1=Dense(16,activation='sigmoid')(r1)
r1=Dense(1,activation='sigmoid')(r1)
output1=tf.keras.layers.BatchNormalization()(r1)
model1=Model(inputs=r1,outputs=output1)

c2=Dense(8,activation='sigmoid')(c2)
c2=Dense(16,activation='sigmoid')(c2)
c2=Dense(32,activation='sigmoid')(c2)
c2=Dense(16,activation='sigmoid')(c2)
c2=Dense(1,activation='sigmoid')(c2)
output2=tf.keras.layers.BatchNormalization()(c2)
model2=Model(inputs=c2,outputs=output2)

r3=Dense(8,activation='sigmoid')(r3)
r3=Dense(16,activation='sigmoid')(r3)
r3=Dense(32,activation='sigmoid')(r3)
r3=Dense(16,activation='sigmoid')(r3)
r3=Dense(1,activation='sigmoid')(r3)
output3=tf.keras.layers.BatchNormalization()(r3)
model3=Model(inputs=r3,outputs=output3)

# r4=Dense(8,activation='sigmoid')(r4)
# r4=Dense(16,activation='sigmoid')(r4)
# r4=Dense(32,activation='sigmoid')(r4)
# r4=Dense(16,activation='sigmoid')(r4)
# r4=Dense(1,activation='sigmoid')(r4)
# output4=tf.keras.layers.BatchNormalization()(r4)
# model4=Model(inputs=r4,outputs=output4)

r5=Dense(8,activation='sigmoid')(r5)
r5=Dense(16,activation='sigmoid')(r5)
r5=Dense(32,activation='sigmoid')(r5)
r5=Dense(16,activation='sigmoid')(r5)
r5=Dense(1,activation='sigmoid')(r5)
output5=tf.keras.layers.BatchNormalization()(r5)
model5=Model(inputs=r5,outputs=output5)

r6=Dense(8,activation='sigmoid')(r6)
r6=Dense(16,activation='sigmoid')(r6)
r6=Dense(32,activation='sigmoid')(r6)
r6=Dense(16,activation='sigmoid')(r6)
r6=Dense(1,activation='sigmoid')(r6)
output6=tf.keras.layers.BatchNormalization()(r6)
model6=Model(inputs=r6,outputs=output6)

r7=Dense(8,activation='sigmoid')(r7)
r7=Dense(16,activation='sigmoid')(r7)
r7=Dense(32,activation='sigmoid')(r7)
r7=Dense(16,activation='sigmoid')(r7)
r7=Dense(1,activation='sigmoid')(r7)
output7=tf.keras.layers.BatchNormalization()(r7)
model7=Model(inputs=r7,outputs=output7)

# r8=Dense(8,activation='sigmoid')(r8)
# r8=Dense(16,activation='sigmoid')(r8)
# r8=Dense(32,activation='sigmoid')(r8)
# r8=Dense(16,activation='sigmoid')(r8)
# r8=Dense(1,activation='sigmoid')(r8)
# output8=tf.keras.layers.BatchNormalization()(r8)
# model8=Model(inputs=r8,outputs=output8)

c9=Dense(8,activation='sigmoid')(c9)
c9=Dense(16,activation='sigmoid')(c9)
c9=Dense(32,activation='sigmoid')(c9)
c9=Dense(16,activation='sigmoid')(c9)
c9=Dense(1,activation='sigmoid')(c9)
output9=tf.keras.layers.BatchNormalization()(c9)
model9=Model(inputs=c9,outputs=output9)

r10=Dense(8,activation='sigmoid')(r10)
r10=Dense(16,activation='sigmoid')(r10)
r10=Dense(32,activation='sigmoid')(r10)
r10=Dense(16,activation='sigmoid')(r10)
r10=Dense(1,activation='sigmoid')(r10)
output10=tf.keras.layers.BatchNormalization()(r10)
model10=Model(inputs=r10,outputs=output10)

c11=Dense(8,activation='sigmoid')(c11)
c11=Dense(16,activation='sigmoid')(c11)
c11=Dense(32,activation='sigmoid')(c11)
c11=Dense(16,activation='sigmoid')(c11)
c11=Dense(1,activation='sigmoid')(c11)
output11=tf.keras.layers.BatchNormalization()(c11)
model11=Model(inputs=c11,outputs=output11)

combined1=K.concatenate([model0.output,model2.output,model9.output,model11.output])

combined1=Dense(8,activation='sigmoid')(combined1)
combined1=Dense(16,activation='sigmoid')(combined1)
combined1=Dense(32,activation='sigmoid')(combined1)
combined1=Dense(16,activation='sigmoid')(combined1)
combined1=Dense(8,activation='sigmoid')(combined1)



combined2=K.concatenate([model1.output,model3.output,model5.output,model6.output,model7.output,model10.output])
combined2=Dense(8,activation='sigmoid')(combined2)
combined2=Dense(16,activation='sigmoid')(combined2)
combined2=Dense(32,activation='sigmoid')(combined2)
combined2=Dense(16,activation='sigmoid')(combined2)
combined2=Dense(8,activation='sigmoid')(combined2)

combined=K.concatenate([combined1,combined2])
combined=Dense(64,activation='sigmoid')(combined)
combined=Dense(32,activation='sigmoid')(combined)
combined=Dense(16,activation='sigmoid')(combined)
combined=Dense(8,activation='sigmoid')(combined)
combined=Dense(1,activation='sigmoid')(combined)


model=Model(inputs=[model0.input,model1.input,model2.input,model3.input,model5.input,model6.input,model7.input,model9.input,model10.input,model11.input],
            outputs=[combined])

model.summary()


model.compile(
    optimizer=tf.keras.optimizers.Adam(lr = 0.001,decay=0.00001),
    loss = 'binary_crossentropy',
    metrics = ['mse']
)


history=model.fit([c0_train,r1_train,c2_train,r3_train,r5_train,r6_train,r7_train,c9_train,r10_train,c11_train],
                  [label_train],epochs=500,batch_size=128)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist['epoch']=hist['epoch']+1

def plot_history(hist):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.xlabel('Epoch')
    plt.plot(hist['epoch'], hist['loss'],
            label='loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.xlabel('Epoch')
    plt.plot(hist['epoch'], hist['mse'],
            label = 'mse',color = 'red')
#     plt.ylim([0,30])
    plt.legend()

plot_history(hist)
"""模型评价"""

c0_test=tf.constant(data_test[:,0]) 
r1_test=tf.constant(data_test[:,1])
c2_test=tf.constant(data_test[:,2]) 
r3_test=tf.constant(data_test[:,3]) 
# r4_test=tf.constant(data[8267:11810,4]) 
r5_test=tf.constant(data_test[:,4]) 
r6_test=tf.constant(data_test[:,5]) 
r7_test=tf.constant(data_test[:,6]) 
# r8_test=tf.constant(data[8267:11810,8]) 
c9_test=tf.constant(data_test[:,7]) 
r10_test=tf.constant(data_test[:,8]) 
c11_test=tf.constant(data_test[:,9]) 


y_pred1 = model.predict([c0_test,r1_test,c2_test,r3_test,
                         r5_test,r6_test,r7_test,c9_test,r10_test,c11_test],batch_size=1)
y_pred1=y_pred1[:,0]
# model.save("./model.multichanel_train")
# def roc(y_true, y_score, pos_label):
#     """
#     y_true：真实标签
#     y_score：模型预测分数
#     pos_label：正样本标签，如“1”
#     """
#     # 统计正样本和负样本的个数
#     num_positive_examples = (y_true == pos_label).sum()
#     num_negtive_examples = len(y_true) - num_positive_examples

#     tp, fp = 0, 0
#     tpr, fpr, thresholds = [], [], []
#     score = max(y_score) + 1
    
#     # 根据排序后的预测分数分别计算fpr和tpr
#     for i in np.flip(np.argsort(y_score)):
#         # 处理样本预测分数相同的情况
#         if y_score[i] != score:
#             fpr.append(fp / num_negtive_examples)
#             tpr.append(tp / num_positive_examples)
#             thresholds.append(score)
#             score = y_score[i]
            
#         if y_true[i] == pos_label:
#             tp += 1
#         else:
#             fp += 1

#     fpr.append(fp / num_negtive_examples)
#     tpr.append(tp / num_positive_examples)
#     thresholds.append(score)

#     return fpr, tpr, thresholds
# y_pred=np.reshape(y_pred1, (3543,))
# fpr, tpr, thresholds = roc(label_test, y_pred, pos_label=1)

# fpr1=np.array(fpr)
# tpr1=np.array(tpr)
# thresholds1=np.array(thresholds)
# plt.plot(fpr1, tpr1)
# plt.axis("square")
# plt.xlabel("False positive rate")
# plt.ylabel("True positive rate")
# plt.title("ROC curve")
# plt.show()


def calculate_correlation_rmse(array1, array2):
    """
    计算两个数组的相关系数和均方根误差(RMSE)
    
    参数:
    array1, array2 -- 输入数组（列表或NumPy数组）
    
    返回:
    (correlation, rmse) -- 相关系数和均方根误差
    """
    # 转换为NumPy数组
    a1 = np.array(array1)
    a2 = np.array(array2)
    
    # 计算相关系数
    correlation = np.corrcoef(a1, a2)[0, 1]
    
    # 计算均方根误差
    mse = np.mean((a1 - a2) ** 2)
    rmse = np.sqrt(mse)
    absolute_errors = np.abs(a1 - a2)
    mae = np.mean(absolute_errors)
    
    return correlation, mae, rmse

# 示例数据


# 计算并打印结果
corr, mae, rmse = calculate_correlation_rmse(label_test, y_pred1)


fpr, tpr, thresholds = roc_curve(label_test, y_pred1)

# 计算AUC（ROC曲线下面积）
roc_auc = auc(fpr, tpr)
print(f"R2: {corr:.4f}")
print(f"rmse: {rmse:.4f}")
print(f"mae: {mae:.4f}")
print(f"auc: {roc_auc:.4f}")

# data_pre=np.loadtxt(r"E:\2025年工作材料\杜国梁老师合作文件\new\field_data.txt")
# parameter0_pre=tf.constant(data_pre[:,0]) 
# parameter1_pre=tf.constant(data_pre[:,1])
# parameter2_pre=tf.constant(data_pre[:,2]) 
# parameter3_pre=tf.constant(data_pre[:,3]) 
# # parameter4_pre=tf.constant(data_pre[:,4]) 
# parameter5_pre=tf.constant(data_pre[:,4]) 
# parameter6_pre=tf.constant(data_pre[:,5]) 
# parameter7_pre=tf.constant(data_pre[:,6]) 
# # parameter8_pre=tf.constant(data_pre[:,8]) 
# parameter9_pre=tf.constant(data_pre[:,7])
# parameter10_pre=tf.constant(data_pre[:,8]) 
# parameter11_pre=tf.constant(data_pre[:,9])

# y_pred2 = model.predict([parameter0_pre,parameter1_pre,parameter2_pre,parameter3_pre,
#                          parameter5_pre,parameter6_pre,parameter7_pre,parameter9_pre,parameter10_pre,parameter11_pre,],batch_size=1)
