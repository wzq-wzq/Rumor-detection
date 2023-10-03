from ast import Lambda
import json
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np

def gen_training_data(raw_data_path):
    label_list=[line.strip() for line in open('label','r',encoding='utf-8')]
    print(label_list)
    id2label = {idx:label for idx,label in enumerate(label_list)}
    print(id2label)
    temp_data=[]
    data=[]
    train_data=[]
    val_data=[]
    test_data=[]
    train_num=0.8
    val_num=0.9
    temp_len=0
    label_set=set()
    for i in range(len(label_list)):
        train_num=0.8
        val_num=0.9
        temp_len=0
        temp_data=[]
        with open(raw_data_path+str(i)+".txt", 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            text = line.replace('\n','')
            label_set.add(id2label[i])
            temp_data.append([text,id2label[i],i])
            temp_len=temp_len+1
        
        print(id2label[i]+"["+str(i)+"]:"+str(temp_len))

        train_num=int(train_num*temp_len)
        val_num=int(val_num*temp_len)
        #print("["+str(i)+"]t_num,v_num:"+str(train_num)+","+str(val_num))
        data.extend(temp_data)
        random.shuffle(temp_data)
        train_data.extend(temp_data[:train_num])
        val_data.extend(temp_data[train_num:val_num])
        test_data.extend(temp_data[val_num:])
        print(id2label[i]+"["+str(i)+"]:"+str(len(temp_data[:train_num]))+","+str(len(temp_data[train_num:val_num]))+","+str(len(temp_data[val_num:])))
        print("当前总长度["+str(i)+"]:"+str(len(train_data))+","+str(len(val_data))+","+str(len(test_data)))

    
    print(label_set)

    data=pd.DataFrame(data,columns=['text','label_class','label'])
    print(data['label_class'].value_counts())

    data['text_len']=data['text'].map(lambda x: len(x))
    print(data['text_len'].describe())

    plt.hist(data['text_len'],bins=30,rwidth=0.9,density=True,)
    plt.show()

    del data['text_len']

    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    train=pd.DataFrame(train_data,columns=['text','label_class','label'])
    val=pd.DataFrame(val_data,columns=['text','label_class','label'])
    test=pd.DataFrame(test_data,columns=['text','label_class','label'])
    train=train.sample(frac=1.0)
    val=val.sample(frac=1.0)
    test=test.sample(frac=1.0)

    '''data=data.sample(frac=1.0)
    train_num=int(0.8*len(data))
    val_num=int(0.9*len(data))
    train,val,test=data[:train_num],data[train_num:val_num],data[val_num:]'''
    train.to_csv("train.csv",index=False)
    val.to_csv("val.csv",index=False)
    test.to_csv("test.csv",index=False)

def load_data(filename):
    df=pd.read_csv(filename,header=0)
    return df[['text','label']].values

if __name__ == '__main__':
    gen_training_data('alltext/new-data/')