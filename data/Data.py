# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 14:25:56 2020

@author: jrmfi
"""
import chardet
import pandas as pd
import numpy as np

class Data():
    def __init__(self,num_days,size=int):
        self.num_days = num_days
        self.size = size
    def import_data(self):
        with open('data/M4.csv', 'rb') as f:
            result = chardet.detect(f.read())  # or readline if the file is large
    
        base = pd.read_csv('data/M4.csv', encoding=result['encoding'])     
        
        base1 = pd.DataFrame(data=base[-self.num_days:-1].values,columns=base.columns)
        base1 = base1.drop(['open', 'high', 'low', 'close','VOL','OBV','Acumulacao','Force','band1', 'band2', 'band3'], axis=1)
        entrada_rnn,entrada_trader,media,std = self.training_assess(base,self.num_days)
        entrada_rnn = self.batch_size(entrada_rnn, self.size)
        return entrada_rnn,entrada_trader,base1,media,std
        
    def duration(self,base):
        index = 0
        for i in base.values:
            base1 = i[0].split(':')
            base.at[index, 'Hora'] = float(base1[0])*100 + float(base1[1])
            index += 1
        return base
    def training_assess(self,base,num_days = 565,colunas = ['Hora','dif', 'retacao +',
                                                            'retracao -', 'RSI', 'M22M44', 
                                                            'M22M66', 'M66M44', 'ADX', 'ATR',
                                                            'Momentum', 'Force']):
        colunas1 = ['Hora', 'open', 'high', 'low', 'close'] 
        # colunas2 = ['Hora', 'dif'] 
        entrada_RNN = pd.DataFrame(data=base[-num_days:-330].values,columns=base.columns)      
        entrada_trade = pd.DataFrame(data=base[-num_days:-330].values,columns=base.columns)
        entrada_RNN = entrada_RNN.drop(['Data', 'open', 'high', 'low', 'close','VOL','OBV','Acumulacao','Force','band1', 'band2', 'band3'], axis=1)
        # entrada_RNN = entrada_RNN[colunas2]
        entrada_trade = entrada_trade[colunas1]
        entrada_RNN = self.duration(entrada_RNN)
        train_mean = entrada_RNN.mean(axis=0)
        train_std = entrada_RNN.std(axis=0)
        entrada_RNN = (entrada_RNN - train_mean) / train_std
        
        return entrada_RNN,entrada_trade,train_mean,train_std
    
    def batch_size(self,data,size):
        entrada = []
        saida = []
        for i in range(len(data)):
            entrada = []
            for c in range(size):
                g = i-c
                if g < 0:
                    g=0
                entrada.append(data.values[g].tolist())
            saida.append(entrada)
        saida = np.array(saida)
            
        return saida