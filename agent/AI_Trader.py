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
        self.memory = deque(maxlen = 50000)
        self.filtro_memory = deque(maxlen = 50000)
        print('   ')
        print('entradas: ',self.state_size,self.state_sizex)
        print('   ')
        self.tuner = tuner 
        self.gamma = 0.95
        self.epsilon = 0.2
        self.epsilon_final = 0.01
        self.epsilon_decay = 0.95
        self.model = self.model() 
        self.model_filter = self.model_filter() 
    def model_builder(self,hp):
        model = tf.keras.models.Sequential()
        hp_units1 = hp.Int('units1', min_value = 16, max_value = 512, step = 32)
        hp_units2 = hp.Int('units2', min_value = 32, max_value = 512, step = 32)
        hp_units3 = hp.Int('units3', min_value = 32, max_value = 512, step = 32)
        hp_units4 = hp.Int('units4', min_value = 32, max_value = 512, step = 32)
        hp_units5 = hp.Int('units5', min_value = 32, max_value = 512, step = 32)
        # hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4]) 
        hp_function1 = hp.Choice('function1', values = ['relu','selu', 'elu',
                                                      'softmax','sigmoid','linear',
                                                      'softplus','softsign','tanh','gelu']) 
        hp_function2 = hp.Choice('function2', values = ['relu','selu', 'elu',
                                                      'softmax','sigmoid','linear',
                                                      'softplus','softsign','tanh','gelu'])
        hp_function3 = hp.Choice('function3', values = ['relu','selu', 'elu',
                                                      'softmax','sigmoid','linear',
                                                      'softplus','softsign','tanh','gelu'])
        hp_function4 = hp.Choice('function4', values = ['relu','selu', 'elu',
                                                      'softmax','sigmoid','linear',
                                                      'softplus','softsign','tanh','gelu'])
        hp_function5 = hp.Choice('function5', values = ['relu','selu', 'elu',
                                                      'softmax','sigmoid','linear',
                                                      'softplus','softsign','tanh','gelu'])
        hp_function6 = hp.Choice('function6', values = ['relu','selu', 'elu',
                                                      'softmax','sigmoid','linear',
                                                      'softplus','softsign','tanh','gelu'])
        hp_kernel = hp.Choice('kernel', values = ['glorot_uniform','glorot_normal','lecun_normal','lecun_uniform','he_normal','he_uniform']) 
        hp_optimizer =hp.Choice('optimizer', values = [ 'Adam','RMSprop','SGD','Nadam','Adamax','Adagrad','Adadelta']) 
        hp_loss =hp.Choice('loss', values = ['mae',
                                            'mse',
                                            'mape',
                                            'log_cosh',
                                            'huber_loss',
                                            'poisson'])
                                                                            
                
        model.add(tf.keras.layers.Dense(units=hp_units1,activation=hp_function1,kernel_initializer=hp_kernel,input_shape=(self.state_size,self.state_sizex)))
        model.add(tf.keras.layers.Dense(units=hp_units2,activation=hp_function2))
        model.add(tf.keras.layers.Dense(units=hp_units3,activation=hp_function3))
        model.add(tf.keras.layers.Dense(units=hp_units4,activation=hp_function4))
        model.add(tf.keras.layers.Dense(units=hp_units5,activation=hp_function5))
        model.add(tf.keras.layers.Dense(units=3,activation=hp_function6))       
        model.compile(optimizer=hp_optimizer,
              loss= hp_loss ,
              metrics=['accuracy'])
        return model
    def model_filter(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(units=32,activation='relu',kernel_initializer='glorot_normal',input_shape=(self.state_size,self.state_sizex)))
        model.add(tf.keras.layers.Dense(units=128,activation='relu'))
        model.add(tf.keras.layers.Dense(units=1,activation='linear'))
        model.compile(optimizer='Adam',
                  loss='mse',
                  metrics=['accuracy'])
        return model
    def model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(units=32,activation='relu',kernel_initializer='glorot_normal',input_shape=(self.state_size,self.state_sizex)))
        model.add(tf.keras.layers.Dense(units=128 ,activation='relu'))
        model.add(tf.keras.layers.Dense(units=256,activation='relu'))
        model.add(tf.keras.layers.Dense(units=256,activation='relu'))
        model.add(tf.keras.layers.Dense(units=128,activation='relu'))
        model.add(tf.keras.layers.Dense(units=3,activation='softsign'))
        model.compile(optimizer='Adam',
                  loss='mse',
                  metrics=['accuracy'])
        return model
    def trade(self,state):
        if random.random() <= self.epsilon:
            # filters = self.model_filter.predict(state) 
            return random.randrange(self.action_space) #,filters
        actions = self.model.predict(state)
        # filters = self.model_filter.predict(state) 
        return np.argmax(actions[0]) #,filters
    
    def batch_train(self,batch_size):
        
        
        #---------------------------------------------
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
            target[0][action] = b.numpy()[0]
            if self.tuner:
                state_batch.append(state.tolist())
                target_batch.append(target.tolist())
            else:
                self.model.fit(state,target,epochs=5,verbose=0)
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
                     max_epochs = 100,
                     factor = 3,
                     directory = 'my_dir',
                     project_name = 'intro_to_kt')
        tuner.search(state, target, epochs = 100, callbacks = [ClearTrainingOutput()])
        best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
        print('--------------------------------')
        print(' ')
        # print(best_hps)
        print(f"""
            The hyperparameter search is complete. The optimal number of units in the first densely-connected
            layer is {best_hps.get('units1')},{best_hps.get('units2')},{best_hps.get('units3')},{best_hps.get('units4')},
            {best_hps.get('units5')},and the activation function is
            {best_hps.get('function1')},{best_hps.get('function2')},{best_hps.get('function3')},{best_hps.get('function4')},
            {best_hps.get('function5')},{best_hps.get('function6')},
            and optimizer is {best_hps.get('optimizer')}.and loss is {best_hps.get('loss')}, and kernel is {best_hps.get('kernel')}
            """)
        print(' ')
        print('--------------------------------')
        return 0
