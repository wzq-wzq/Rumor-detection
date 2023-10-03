import json
import pandas as pd
import numpy as np


from bert4keras.backend import keras
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.optimizers import Adam
from sklearn import metrics

from sklearn.metrics import classification_report
from sklearn.utils import shuffle

from mybilstm import build_bert_model
from mydata import load_data

import matplotlib.pyplot as plt

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('accuracy'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_accuracy'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('accuracy'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_accuracy'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        #创建一个图
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train accuracy')#plt.plot(x,y)，这个将数据画成曲线
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val accuracy')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)#设置网格形式
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')#给x，y轴加注释
        plt.legend(loc="upper right")#设置图例显示位置
        plt.savefig('./plt/'+loss_type+'.jpg')
        plt.show()
        plt.pause(10)
        plt.close()


#参数
maxlen=190
batch_size=16
adam_num=1e-5
epoch_size=20

class_num=6
dropout_num_1=0.2
dropout_num_2=0.3

config_path='D:/Users/74148/wzq/BERT/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path='D:/Users/74148/wzq/BERT/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path='D:/Users/74148/wzq/BERT/chinese_L-12_H-768_A-12/vocab.txt'

tokenizer=Tokenizer(dict_path)
errortext=[]

class data_generator(DataGenerator):
    """
    数据生成
    """

    def __iter__(self, random=False):
        batch_token_ids,batch_segments_ids,batch_labels=[],[],[]
        for is_end,(text,label) in self.sample(random):
            try:
                token_ids,segment_ids=tokenizer.encode(text,maxlen=maxlen)
                batch_token_ids.append(token_ids)
                batch_segments_ids.append(segment_ids)
                batch_labels.append([label])
                if len(batch_token_ids)==self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segments_ids=sequence_padding(batch_segments_ids)
                    batch_labels=sequence_padding(batch_labels)
                    yield [batch_token_ids,batch_segments_ids],batch_labels
                    batch_token_ids,batch_segments_ids,batch_labels=[],[],[]
            except Exception as e:
                print(str(e))
                print(str(text)+":"+str(label))
                errortext.append(text)



if __name__=="__main__":
    keras.backend.clear_session()

    train_data=load_data('train.csv')
    val_data=load_data('val.csv')
    test_data=load_data('test.csv')

    train_generator=data_generator(train_data,batch_size)
    val_generator=data_generator(val_data,batch_size)
    test_generator=data_generator(test_data,batch_size)

    model=build_bert_model(config_path,checkpoint_path,class_num,dropout_num_1=dropout_num_1,dropout_num_2=dropout_num_2)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(adam_num),
        metrics=['accuracy'],
    )

    earlystop = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        verbose=2,
        mode='max'
    )

    best_model_filepath='best_model.weights'

    checkpoint=keras.callbacks.ModelCheckpoint(
        best_model_filepath,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )

    history=LossHistory()

    board=keras.callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=0,
        write_graph=False,
        write_images=False,
        update_freq=1000,
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None
    )


    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epoch_size,
        validation_data=val_generator.forfit(),
        validation_steps=len(val_generator),
        shuffle=True,
        callbacks=[checkpoint,history,board,earlystop]
        #callbacks=[checkpoint,history]
    )
    #model.save_weights('best_model.weights')

    model.load_weights('best_model.weights')
    errortext=[]
    test_pred=[]
    test_true=[]
    test_true_temp=[]

    for x,y in test_generator:
        p=model.predict(x).argmax(axis=1)
        test_pred.extend(p)
    
    test_true=test_data[:,1].tolist()

    print("len of errortext:"+str(len(errortext)))
    needtodelete=[]
    for i in range(len(test_data)):
        try:
            #print(str(test_data[i][0]))
            if test_data[i][0] in errortext:
                needtodelete.append(i)
        except:
            needtodelete.append(i)
    print("len of needtodelete:"+str(len(needtodelete)))
    for i in range(len(test_data)):
        if i in needtodelete:
            continue
        else:
            test_true_temp.append(test_true[i])
    test_true=test_true_temp
    print(len(test_true))

    print(set(test_true))
    print(set(test_pred))

    target_names=[line.strip() for line in open('label','r',encoding='utf-8')]
    print(classification_report(test_true,test_pred,target_names=target_names))

    history.loss_plot('epoch')
    history.loss_plot('batch')
