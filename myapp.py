'''
def mydo(self):
        #参数
        maxlen=190
        #batch_size=32
        class_num=6

        config_path='D:/Users/74148/wzq/BERT/chinese_L-12_H-768_A-12/bert_config.json'
        checkpoint_path='D:/Users/74148/wzq/BERT/chinese_L-12_H-768_A-12/bert_model.ckpt'
        dict_path='D:/Users/74148/wzq/BERT/chinese_L-12_H-768_A-12/vocab.txt'

        tokenizer=Tokenizer(dict_path)

        model=build_bert_model(config_path,checkpoint_path,class_num)
        #best_model_filepath='best_model.weights'
    
        model.load_weights('best_model.weights')
        label_list=[line.strip() for line in open('label','r',encoding='utf-8')]
        id2label = {idx:label for idx,label in enumerate(label_list)}
        t,s=tokenizer.encode(self.textEdit.text(),maxlen=maxlen)
        mylabelid=model.predict([[t],[s]]).argmax(axis=1)[0]
        mylabel=id2label[mylabelid]
        self.label.setText(mylabel)
'''

import sys 	
from PyQt5.QtWidgets import QApplication , QMainWindow
from Ui_app import *

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



class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, m,t,parent=None):    
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self,model=m,tokenizer=t)
            
if __name__=="__main__":
        #参数
    maxlen=190
    #batch_size=32
    class_num=6
    config_path='D:/Users/74148/wzq/BERT/chinese_L-12_H-768_A-12/bert_config.json'
    checkpoint_path='D:/Users/74148/wzq/BERT/chinese_L-12_H-768_A-12/bert_model.ckpt'
    dict_path='D:/Users/74148/wzq/BERT/chinese_L-12_H-768_A-12/vocab.txt'

    tokenizer=Tokenizer(dict_path)

    model=build_bert_model(config_path,checkpoint_path,class_num)
    #best_model_filepath='best_model.weights'
    model.load_weights('best_model.weights')  
    app = QApplication(sys.argv)  
    myWin = MyMainWindow(model,tokenizer)  
    myWin.show()  
    sys.exit(app.exec_())