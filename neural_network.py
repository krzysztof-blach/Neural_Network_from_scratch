# -*- coding: utf-8 -*-
"""
Created on Fri May 15 09:07:54 2020

@author: kblac
"""

import numpy as np

class Neural_Network:
    
    def __init__(self, x, y, layer_dims):
        
        self.x = x
        self.y = y
        self.layer_dims = layer_dims
        self.parameters = None
        
    def sigmoid(self, z):
        """
        Function to calculate sigmoid function on (multidimensional) vector z
        Returns: result of sigmoid function as well as the input vector z
        """
        return 1 /( 1 + np.exp(-z)), z
    
    def relu(self, z):
        """
        Function to calculate the ReLU function on (multidimensional) vector z
        Returns: result of relu function as well as the input vector z
        """
        return np.maximum(0, z), z
        
        
    def initialize_weights(self):
        """
        Function to initialize weights of the network for each layer
        Weights W are rondomly initialized
        Coeeficients b are initialized as zeros     
        """
        parameters = {}
        L = len(self.layer_dims)
        for i in range(1, L):
            parameters["W" + str(i)] = np.random.randn((self.layer_dims[i],
                                                        self.layer_dims[i - 1])) * 0.01
            parameters["b" + str(i)] = np.zeros((self.layer_dims[i], 1))
        self.parameters = parameters
        return self
            
    def forward_linear_step(self, a_prev, w, b):
        """
        Function to compute the linear part of single forward propagation step, namely multiplication of
        weights and result from previous layer
        """
        z = np.dot(w, a_prev) + b
        cache = a_prev, w, b
        return z, cache
    
    
    def forward_linear_activation(self, a_prev, w, b, activation_func):
        """
        Function to calculate linear and activation parts of a single forward step
        """
        z, linear_cache = self.forward_linear_step(a_prev, w, b)
        if activation_func == "relu":
            a, activation_cache = self.relu(z)
        elif activation_func == "sigmoid":
            a, activation_cache = self.sigmoid(z)
        cache = (linear_cache, activation_cache)
        return a, cache
    
    def full_forward_model(self):
        
        self.initialize_weights()
        caches = []
        a = self.x
        for i in range(len(self.layer_dims) - 1):
            a, cache = self.forward_linear_activation(a, 
                                                 self.parameters["W" + str(i)],
                                                 self.parameters["b" + str(i)],
                                                 "relu")
            caches.append(cache)
        a, cache = self.forward_linear_activation(a, 
                                                 self.parameters["W" + str(i)],
                                                 self.parameters["b" + str(i)],
                                                 "sigmoid")
        caches.append(cache)
        return a, caches
    
    def loss_function(self):
        """
        Function to calculate the loss function
        """
        pass
    
    def backpropagation(self):
        pass
    
    def model(self):
        pass
        