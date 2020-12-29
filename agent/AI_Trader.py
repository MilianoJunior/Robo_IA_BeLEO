from collections import deque
import tensorflow as tf
import random
import numpy as np
import os

class AI_Trader():
    def __init__(self,state_size,state_sizex,action_space,model_name='AITrader'):
        self.state_size = state_size
        self.action_space = action_space
        self.state_sizex = state_sizex
        self.memory = deque(maxlen = 2000)
        self.model_name = model_name
        
        self.gamma = 0.95
        self.epsilon = 0.2
        self.epsilon_final = 0.01
        self.epsilon_decay = 0.995
        self.model = self.model()   
    def model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(units=32,activation='relu',input_shape=(self.state_size,self.state_sizex)))
        model.add(tf.keras.layers.Dense(units=64,activation='relu'))
        model.add(tf.keras.layers.Dense(units=128,activation='relu'))
        model.add(tf.keras.layers.Dense(units=3,activation='softsign'))
                  
        model.compile(optimizer='adam',
              loss='poisson',
              metrics=['accuracy'])
        return model
    def trade(self,state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_space)
        actions = self.model.predict(state)
        return np.argmax(actions[0])
    
    def batch_train(self,batch_size):
        batch = []
        for i in range(len(self.memory)):
            batch.append(self.memory[i])
        self.memory.clear()
        y = 0
        for state,action,reward,next_state,done in batch:
            reward1 = reward
            a1 = tf.constant([reward], dtype = tf.float32)
            reward = tf.keras.activations.softsign(a1)
            y += 1
            if not done:
                prev = np.amax(self.model.predict(next_state)[0])
                reward = reward.numpy()[0] + self.gamma * prev
                a = tf.constant([reward], dtype = tf.float32)
            else:
                a = tf.constant([reward1], dtype = tf.float32)
            b = tf.keras.activations.softsign(a)
            target = self.model.predict(state)
            # print(y,'-',b.numpy(),action)
            target[0][action] = b.numpy()[0]
            # print('contador: ',y)
            # print('alvo: ',target)
            # print('action: ',action)
            # print('state: ',state)
            # print('reward : ',reward)
            # print('reward: ',reward1)
            # print('prev: ',prev)
            # print('gama: ',self.gamma)
            # print('teste: ',b.numpy()[0])
            # print('done: ',done)
            # print('-----------------------')
            self.model.fit(state,target,epochs=1,verbose=0)
            
        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay
