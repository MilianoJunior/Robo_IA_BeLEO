# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 06:58:35 2020

@author: jrmfi
"""

import math
import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from data.Data import Data
from agent.AI_Trader import AI_Trader
from enviroment.Env_trader import Env_trader
import os

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# TF_XLA_FLAGS=--tf_xla_auto_jit=1
# TF_XLA_FLAGS=--tf_xla_auto_jit=2 
#Hiperparametros
num_days = 549
window_size = 1
action_space = 3
episodes = 10
stop = -300
gain = 300
batch_size = 256

data = Data(num_days,window_size)

input_rnn,input_trader,base,media,std= data.import_data()
print(media)
print(std)
env_trader = Env_trader()
agent = AI_Trader(state_size=window_size,state_sizex =input_rnn.shape[2], action_space=action_space)

agent.model.summary()

# def discount_rewards(box):
#     discounted = []
#     aux = []
    
#     for i in range(len(box)-1):
#         if isinstance(box[i][9], list):
#             discounted = np.array(box[i][9])
#             for step in range(len(box[i][9])-2, -1, -1):
#                 # if len(box[i][9]) > len(box):
#                 #     box3 = []
#                 #     return box3,trader.memory
#                 discounted[step] += discounted[step + 1] * 0.95
#                 trader.memory.append((box[i-step][0],box[i-step][1],round(discounted[step],3),box[i-step][2],box[i-step][3]))
#                 aux.append([box[i-step][0],round(discounted[step],3)])
#                 # state,action,reward,next_state,done
#     return aux,trader.memory

media = []

for episode in range(1,episodes + 1):
    for t in range(1,len(input_trader)-1):
        t1 = time.time()
        action = agent.trade(input_rnn[t])
        # action2 = np.argmax(action[0])
        # print(t,' acao: ',action,' escolha: ',action2)
        # action = random.randrange(0,3)
    #     print(action)
        buy,shell,trading,state,comprado,vendido,reward = env_trader.agente(input_trader.values[t-1],action,stop,gain,0)
        # state,action,reward,next_state,done
        if t >= (len(input_trader)-2):
            done = True
        else:
            done = False
        # print('------------------------------')
        # print('action: ',action)
        # print('comprado: ',comprado)
        # print('vendido: ',vendido)
        # print('reward: ',reward)
        # print('epsilon: ',agent.epsilon)
        # print('done: ',done)
        # print('t: ',t,' tamanho: ',len(input_trader))
        # print(t,'-',[input_rnn[t],action2,reward,input_rnn[t+1],done])
        agent.memory.append([input_rnn[t],action,reward,input_rnn[t],done])
        if t%540 == 0 and t > 10:
            t2 = time.time() - t1
            print('-----------------------------')
            print('tempo gasto amostra: ',t2)
            print('------------------------------')
            t1 = time.time()
            agent.batch_train(len(agent.memory)) 
            t2 = time.time() - t1
            print('------------------------------')
            print('tempo gasto trainamento: ',t2)
            print('------------------------------')
            media.append(sum(trading.ganhofinal))
            print(episode,'- ganho: ',sum(trading.ganhofinal),' media: ',sum(media)/len(media))
            print('numero de operações: ',len(trading.ganhofinal), ' epsilon: ',agent.epsilon)
            env_trader.reset()  
# agent.model.save('modelo_01')
    # agent.batch_train()
    # print('Ganho final: ',sum(trading.ganhofinal),' quantidade: ',len(trading.ganhofinal))
    # # print("Episode: {}/{}".format(episode,episodes))
    # state = box1[0]
    # total_profit = 0
    # trader.inventory = [] #armazena todas as negociações
    # box2 = []
    # aux1 = 0
    # cont = 0
    # for t in range(data_sample):
    #     action,prev,teste = trader.Trade(state)
    #     # if teste == 'rede neural':
    #     #     print('rede neural: ',action)
    #     #     print('rede neural: ',prev)
    #     next_state = box1[t]
    #     compras,vendas,trades,ficha,cc,vv,pos = neg.agente(dados3.values[t],action,stop,gain,0)
    #     if t == data_sample-1:
    #         done = True
    #     else:
    #         done = False
    #     if pos == 4 or pos == 2:
    #         cont += 1
    #         ab = 1 if  int(trades.duracao[len(trades.ganhofinal)-1]) <=0 else int(trades.duracao[len(trades.ganhofinal)-1])
    #         reward = [0 for i in range(ab)]
    #         reward.append(trades.ganhofinal[len(trades.ganhofinal)-1])
    #         box2.append([state,action,next_state,done,ficha,cc,vv,pos,trades.ganhofinal[len(trades.ganhofinal)-1],reward])
    #         if cont >batch_size:
    #             memory = discount_rewards(box2)
    #             print('memori: ',cont,len(box2),len(trader.memory))
    #             aux = True
    #     else:
    #         aux = False
    #         box2.append([state,action,next_state,done,ficha,cc,vv,pos,0,0])
    #     state = next_state
    #     if cont > batch_size and aux :
    #         # print(trader.memory)
    #         cont = 0
    #         t1 = time.time()
    #         epsilon = trader.batch_train(len(trader.memory) )
    #         t2 = time.time() - t1
    #         print('tempo gasto: ',t2)
    #         neg.reset()
    #         del box2
    #         box2 = []
    #         lucro.append(sum(trades.ganhofinal))
    #         print('   ')
    #         print(episode,'##########################')
    #         print('Total profit: ',sum(trades.ganhofinal), ' Media de ganho: ',statistics.mean(lucro),'n trades: ',len(trades.ganhofinal))
    #         print('Epsilon: ',epsilon,batch_size,t)
    #         print('##########################')
    #         memoria = len(trader.memory)
        
        
        
        
        
        
        
        
        
        