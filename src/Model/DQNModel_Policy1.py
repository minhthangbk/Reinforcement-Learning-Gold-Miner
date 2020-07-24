# -*- coding: utf-8 -*-
from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
import tensorflow as tf
from random import random, randrange, choice

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
# THANG
ACTION_GO_LEFT = 0
ACTION_GO_RIGHT = 1
ACTION_GO_UP = 2
ACTION_GO_DOWN = 3
ACTION_FREE = 4
ACTION_CRAFT = 5
MINIMUM_ENERGY = 10
# Deep Q Network off-policy
class DQN_Policy1:

    def __init__(
            self,
            input_dim, #The number of inputs for the DQN network
            action_space, #The number of actions for the DQN network
            gamma = 0.99, #The discount factor
            epsilon = 1, #Epsilon - the exploration factor
            epsilon_min = 0.01, #The minimum epsilon 
            epsilon_decay = 0.0005,#The decay epislon for each update_epsilon time
            learning_rate = 0.00025, #The learning rate for the DQN network
            tau = 0.125, #The factor for updating the DQN target network from the DQN network
            model = None, #The DQN model
            target_model = None, #The DQN target model 
            sess=None
            
    ):
      self.input_dim = input_dim
      self.action_space = action_space
      self.gamma = gamma
      self.epsilon = epsilon
      self.epsilon_min = epsilon_min
      self.epsilon_decay = epsilon_decay
      self.learning_rate = learning_rate
      self.tau = tau
            
      #Creating networks
      self.model        = self.create_model() #Creating the DQN model
      self.target_model = self.create_model() #Creating the DQN target model
      
      #Tensorflow GPU optimization
      config = tf.compat.v1.ConfigProto()
      config.gpu_options.allow_growth = True
      self.sess = tf.compat.v1.Session(config=config)
      K.set_session(sess)
      self.sess.run( tf.compat.v1.global_variables_initializer()) 
      
    def create_model(self):
      #Creating the network
      #Two hidden layers (300,300), their activation is ReLu
      #One output layer with action_space of nodes, activation is linear.
      model = Sequential()
      model.add(Dense(512, input_dim=self.input_dim, kernel_initializer='he_uniform'))
      model.add(Activation('relu'))
      model.add(Dense(256, kernel_initializer='he_uniform'))
      model.add(Activation('relu'))
      model.add(Dense(self.action_space, kernel_initializer='he_uniform'))
      model.add(Activation('linear'))
      model.summary()
      adam = optimizers.Adam(lr=self.learning_rate)
      sgd = optimizers.SGD(lr=self.learning_rate, decay=1e-6, momentum=0.95)
      model.compile(optimizer = adam,
              loss='mse')
      return model
  
    
    # def act(self,state):
    #   #Get the index of the maximum Q values
    #   a_max = np.argmax(self.model.predict(state.reshape(1,len(state))))
    #   if (random() < self.epsilon):
    #     a_chosen = randrange(self.action_space)
    #   else:
    #     a_chosen = a_max
    #   return a_chosen

    # THANG
    def act_v1(self,state,selectedMovement,remained_energy):
      #Get the index of the maximum Q values
      a_max = np.argmax(self.model.predict(state.reshape(1,len(state))))
      if (random.random() < self.epsilon):  # then explore the map
          if selectedMovement == 0: #Top left corner
              if remained_energy >= MINIMUM_ENERGY:
                  a_chosen =  random.choice([ACTION_GO_RIGHT,ACTION_GO_DOWN,ACTION_CRAFT])  #not move left (0) and up (2)
              else:
                  a_chosen = ACTION_FREE
          elif selectedMovement == 1:  #Top right corner
              if remained_energy >= MINIMUM_ENERGY:
                  a_chosen = random.choice([ACTION_GO_LEFT,ACTION_GO_DOWN,ACTION_CRAFT])  #not move right (0) and up (2)
              else:
                  a_chosen = ACTION_FREE
          elif selectedMovement == 2:  #Bottom right corner
              if remained_energy >= MINIMUM_ENERGY:
                  a_chosen = random.choice([ACTION_GO_LEFT,ACTION_GO_UP,ACTION_CRAFT])  #not move right and down
              else:
                  a_chosen = ACTION_FREE
          elif selectedMovement == 3:  #Bottom left corner
              if remained_energy >= MINIMUM_ENERGY:
                  a_chosen = random.choice([ACTION_GO_RIGHT,ACTION_GO_UP,ACTION_FREE,ACTION_CRAFT]) #not move left and down
              else:
                  a_chosen = ACTION_FREE
          elif selectedMovement == 4:  #Normal
              if remained_energy >= MINIMUM_ENERGY:
                  a_chosen = random.choice([ACTION_GO_LEFT,ACTION_GO_RIGHT,ACTION_GO_UP,ACTION_GO_DOWN,ACTION_CRAFT])
              else:
                  a_chosen = ACTION_FREE
        # # ACTIONS = {0: 'move left', 1: 'move right', 2: 'move up', 3: 'move down', 4: 'stand', 5: 'mining'}
        # # THANG
        # if self.state.mapInfo.gold_amount(self.info.posx, self.info.posy) > 0:

      else:
        a_chosen = a_max
      return a_chosen
    
    
    def replay(self,samples,batch_size):
      inputs = np.zeros((batch_size, self.input_dim))
      targets = np.zeros((batch_size, self.action_space))
      
      for i in range(batch_size):
        state = samples[0][i,:]
        action = samples[1][i]
        reward = samples[2][i]
        new_state = samples[3][i,:]
        done= samples[4][i]
        
        inputs[i,:] = state
        targets[i,:] = self.model.predict(state.reshape(1,len(state)))
        target_old[i,:]  = np.array(targets)
        target_next[i,:]  = self.target_model.predict(new_state.reshape(1,len(new_state)))
        if done:
          targets[i,action] = reward # if terminated, only equals reward
        else:
          Q_future = np.max(target_next)
          targets[i,action] = reward + Q_future * self.gamma
      indices = np.arange(self.batch_size, dtype=np.int32)
      absolute_errors = np.abs(target_old[indices, np.array(action)]-targets[indices, np.array(action)])
      #Training
      loss = self.model.train_on_batch(inputs, targets)
      return absolute_errors
    
    def target_train(self): 
      weights = self.model.get_weights()
      target_weights = self.target_model.get_weights()
      for i in range(0, len(target_weights)):
        target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
      
      self.target_model.set_weights(target_weights) 
    
    
    def update_epsilon(self, decay_step):
      # self.epsilon =  self.epsilon*self.epsilon_decay
      # self.epsilon =  max(self.epsilon_min, self.epsilon)
      self.epsilon = self.epsilon_min + (self.epsilon - self.epsilon_min) * np.exp(-self.epsilon_decay * decay_step)
    
    
    def save_model(self,path, model_name):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(path + model_name + ".json", "w") as json_file:
            json_file.write(model_json)
            # serialize weights to HDF5
            self.model.save_weights(path + model_name + ".h5")
            print("Saved model to disk")
 

