from collections import deque
import tensorflow as tf
import random
import numpy as np
import kerastuner as kt
from kerastuner import HyperParameters
import IPython

class ClearTrainingOutput(tf.keras.callbacks.Callback):
  def on_train_end(*args, **kwargs):
    IPython.display.clear_output(wait = True)
class AI_Trader():
    def __init__(self,state_size,state_sizex,action_space,tuner=False):
        self.state_size = state_size
        self.action_space = action_space
        self.state_sizex = state_sizex
        self.memory = deque(maxlen = 20000)
        
        self.tuner = tuner 
        self.gamma = 0.95
        self.epsilon = 0.2
        self.epsilon_final = 0.01
        self.epsilon_decay = 0.95
        self.model = self.model() 
    def model_builder(self,hp):
        model = tf.keras.models.Sequential()
        hp_units = hp.Int('units', min_value = 32, max_value = 512, step = 32)
        hp_units = hp.Int('units', min_value = 32, max_value = 512, step = 32)
        hp_units = hp.Int('units', min_value = 32, max_value = 512, step = 32)
        hp_units = hp.Int('units', min_value = 32, max_value = 512, step = 32)
        hp_units = hp.Int('units', min_value = 32, max_value = 512, step = 32)
        hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4]) 
        model.add(tf.keras.layers.Dense(units=32,activation='relu',input_shape=(self.state_size,self.state_sizex)))
        model.add(tf.keras.layers.Dense(units=hp_units,activation='relu'))
        model.add(tf.keras.layers.Dense(units=64,activation='relu'))
        model.add(tf.keras.layers.Dense(units=128,activation='relu'))
        model.add(tf.keras.layers.Dense(units=3,activation='softsign'))       
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = hp_learning_rate),
              loss='mse',
              metrics=['accuracy'])
        return model
    def model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(units=32,activation='relu',input_shape=(self.state_size,self.state_sizex)))
        model.add(tf.keras.layers.Dense(units=160 ,activation='relu'))
        model.add(tf.keras.layers.Dense(units=128,activation='relu'))
        model.add(tf.keras.layers.Dense(units=3,activation='softsign'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate =0.001),
                  loss='mse',
                  metrics=['accuracy'])
        return model
    def trade(self,state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_space)
        actions = self.model.predict(state)
        return np.argmax(actions[0])
    
    def batch_train(self,batch_size):
        batch = []
        state_batch =[]
        target_batch = []
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
            if self.tuner:
                state_batch.append(state.tolist())
                target_batch.append(target.tolist())
            else:
                self.model.fit(state,target,epochs=1,verbose=0)
        if self.tuner:
            state_batch = np.array(state_batch)
            target_batch = np.array(target_batch)
            self.fit_tuner(state_batch,target_batch)
            
        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay
    def fit_tuner(self,state,target):
        print('--------------------------------')
        print(' ')
        print('Executatndo Tuner')
        print(' ')
        print('--------------------------------')
        tuner = kt.Hyperband(self.model_builder,
                     objective ='loss', 
                     # hyperparameters=self.hp,
                     max_epochs = 10,
                     factor = 3,
                     directory = 'my_dir',
                     project_name = 'intro_to_kt')
        tuner.search(state, target, epochs = 10, callbacks = [ClearTrainingOutput()])
        best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
        print('--------------------------------')
        print(' ')
        # print(best_hps)
        print(f"""
            The hyperparameter search is complete. The optimal number of units in the first densely-connected
            layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
            is {best_hps.get('learning_rate')}.
            """)
        print(' ')
        print('--------------------------------')
        return 0
