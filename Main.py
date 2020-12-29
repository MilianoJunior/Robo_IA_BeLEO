import time
import tensorflow as tf
from data.Data import Data
from agent.AI_Trader import AI_Trader
from enviroment.Env_trader import Env_trader

#Hiperparametros
num_days = 550
window_size = 1
action_space = 3
episodes = 10
stop = -300
gain = 300
batch_size = 256
with tf.device('/GPU:0'):
    data = Data(num_days,window_size)
    
    input_rnn,input_trader,base,media,std= data.import_data()
    print(media)
    print(std)
    env_trader = Env_trader()
    agent = AI_Trader(state_size=window_size,state_sizex =input_rnn.shape[2], action_space=action_space)
    
    agent.model.summary()
    
    def discount_rewards(reward,contador,done,a,f,action,agent):
        reward = reward - contador
        desconto = []
        for step in range(contador):
            desconto.append(reward)
            reward = reward * 0.95
        for step in range(contador-1,-1,-1):
            done = False
            if step == 0:
                done = True
            agent.memory.append([a[(contador-1)-step],action[(contador-1)-step],desconto[step],f[(contador-1)-step],done])

    
    media = []
    estado = []
    estado_futuro = []
    action_memoria = []
    ficha = False
    mode = 0
    aux = 0
    for episode in range(1,episodes + 1):
        env_trader.reset() 
        aux = 0
        for t in range(1,len(input_trader)-1):
            t1 = time.time()
            action = agent.trade(input_rnn[t])
            buy,shell,trading,state,comprado,vendido,reward = env_trader.agente(input_trader.values[t-1],action,stop,gain,0,mode)
            if t >= (len(input_trader)-2):
                done = True
            else:
                done = False
            if comprado == True or vendido == True:
                aux += 1
                if mode != 2:
                    agent.memory.append([input_rnn[t-1],action,reward,input_rnn[t],done])
                else:
                    estado.append(input_rnn[t-1])
                    estado_futuro.append(input_rnn[t])
                    action_memoria.append(action)
                ficha = True
            if ficha == True and comprado == False and vendido == False:
                done = True
                if mode != 2:
                    agent.memory.append([input_rnn[t-1],action,reward,input_rnn[t],done])
                else:
                    aux += 1
                    estado.append(input_rnn[t-1])
                    estado_futuro.append(input_rnn[t])
                    action_memoria.append(action)
                    discount_rewards(reward,aux,done,estado,estado_futuro,action_memoria,agent)
                    aux = 0
                    estado = []
                    estado_futuro = []
                    action_memoria = []
                ficha = False
            # print('------------------------------')
            # print('action: ',action)
            # print('comprado: ',comprado)
            # print('vendido: ',vendido)
            # print('reward: ',reward)
            # print('epsilon: ',agent.epsilon)
            # print('done: ',done)
            # print('Em operação: ',ficha)
            # print('t: ',t,' tamanho: ',len(input_trader))
            # agent.memory.append([input_rnn[t],action,reward,input_rnn[t],done])
            if t%540 == 0 and t > 10:
                t2 = time.time() - t1
                print('-----------------------------')
                print('tempo gasto amostra: ',t2, 'tamanho da memoria: ',len(agent.memory))
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
                 
    # agent.model.save('modelo_01')

        
        
        
        
        
        
        
        
        