from gc import callbacks
import json
import pandas as pd
import numpy as np

#import tensorflow as tf
from bert4keras.backend import keras
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.optimizers import Adam
from sklearn import metrics

from sklearn.metrics import classification_report
from sklearn.utils import shuffle

from mybilstm import build_bert_model
from mydata import load_data

#参数
maxlen=190
#batch_size=8
#adam_num=5e-6
#epoch_size=20
batch_size=16
#class_num=2
class_num=6
dropout_num_1=0.2
dropout_num_2=0.3

config_path='D:/Users/74148/wzq/BERT/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path='D:/Users/74148/wzq/BERT/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path='D:/Users/74148/wzq/BERT/chinese_L-12_H-768_A-12/vocab.txt'
load_path='D:/Users/74148/wzq/code/python/project1/'



tokenizer=Tokenizer(dict_path)
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
    train_data=load_data('train.csv')
    test_data=load_data('test.csv')

    train_generator=data_generator(train_data,batch_size)
    test_generator=data_generator(test_data,batch_size)

    model=build_bert_model(config_path,checkpoint_path,class_num,dropout_num_1=dropout_num_1,dropout_num_2=dropout_num_2)
    best_model_filepath='best_model.weights'
    
    '''model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(5e-6),
        metrics=['accuracy'],
    )

    earlystop = keras.callbacks.EarlyStopping(
        monitor='val_acc',
        patience=2,
        verbose=2,
        mode='max'
    )

    

    checkpoint=keras.callbacks.ModelCheckpoint(
        best_model_filepath,
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        mode='max'
    )

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=10,
        validation_data=test_generator.forfit(),
        validation_steps=len(test_generator),
        shuffle=True,
        callbacks=[earlystop,checkpoint]
    )'''

    model.load_weights('best_model.weights')
    test_pred=[]
    test_true=[]

    errortext=[]
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
    #target_names=[line.strip() for line in open(load_path+'oldlabel','r',encoding='utf-8')]
    target_names=[line.strip() for line in open('label','r',encoding='utf-8')]
    print(classification_report(test_true,test_pred,target_names=target_names,digits=4))