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
        self.epsilon = 0.5
        self.epsilon_final = 0.01
        self.epsilon_decay = 0.995
        self.model = self.model()   
    def model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(units=32,activation='relu',input_shape=(self.state_size,self.state_sizex)))
        model.add(tf.keras.layers.Dense(units=64,activation='relu'))
        model.add(tf.keras.layers.Dense(units=128,activation='relu'))
        model.add(tf.keras.layers.Dense(units=3,activation='linear'))
                  
        model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])
        return model
    def trade(self,state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_space)
        actions = self.model.predict(state)
        return np.argmax(actions[0])
    
    def batch_train(self,batch_size):
        try:
            # Specify an invalid GPU device
          with tf.device('/job:localhost/replica:0/task:0/device:GPU:0'):
            batch = []
            for i in range(len(self.memory)):
                batch.append(self.memory[i])
            self.memory.clear()
            for state,action,reward,next_state,done in batch:
                reward1 = reward
                if not done:
                    reward = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
                target = self.model.predict(state)
                target[0][action] = reward
                # print('alvo: ',target)
                # print('action: ',action)
                # print('state: ',state)
                # print('reward : ',reward)
                # print('reward: ',reward1)
                # print('done: ',done)
                # print('-----------------------')
                self.model.fit(state,target,epochs=1,verbose=0)
                
            if self.epsilon > self.epsilon_final:
                self.epsilon *= self.epsilon_decay
        except RuntimeError as e:
            print(e)