#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 15:12:22 2021

@author: eric

para cada ATR, miramos cual es el seleccionado
computamos señal flow desde el modelo correspondiente
calculamos RMSE

"""

import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from numpy.random import randn
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt


# generate points in latent space as input for the g_model
def generate_latent_points(latent_dim, n_samples):
    """Generates 'n_samples' points in the latent space for the g_model

    Args:
        latent_dim (int): dimension of the latent space
        n_samples (int): number of samples to be generated
    """
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim,1)
    return x_input


ATR_colection = [3452, 3460, 3518, 3580, 3581, 3642, 3909, 3910, 3912, 3913, 3917,
       3918, 3919, 3921, 3926, #5467,
       5474, 5476, 5477, 5480, 5481, 5482, 
       5499, 5506, 5530, 5537, 5538, 5539, 5542, 5543, 5544, 5546, 5547, 
       5549, 5555, 5556, 5558, 5562, 5563, 5565, 5568, 5570, 5572, 5576, 
       5578, 5579, 5581, 5583, 5588, 5592, 5600, 5607, 5611, 5616, 5620, 6980] #7085#]

#for saving results
d ={}

results = pd.read_csv('error_metrics.csv', index_col=0)

#devuelve el ID del modelo
selected_ATR = results.loc['selected_ATR_embedding']

for selec in range(len(ATR_colection)):
    # selec = 0
    
    #load model
    model = tf.keras.models.load_model('model_'+str(int(selected_ATR[selec]))+'.h5')
    
    #we need a scaler
    df_train = pd.read_csv('C-colombia_BATCH/'+str(int(selected_ATR[selec]))+'_B.csv', index_col=0)
    # data normalization
    scaler = MinMaxScaler()
    scaler.fit(df_train.values   )
    
    df_test = pd.read_csv('C-colombia_BATCH/'+str(ATR_colection[selec])+'_B.csv', index_col=0)
    X_test = df_test.values
    #cargamos los modos, que son las condiciones de la cGAN
    n_classes=14
    df_modos = pd.read_csv('modos_colombia_'+str(n_classes)+'.csv')
    
    latent_dim = 96*2
    
    #numero de veces calculamos el RMSE
    n_iterations=10
    
    #guardamos la media de los 'n_iterations' errores
    error_ATR = list()
    
    print('TESTING ATR '+str(ATR_colection[selec])+'...')
    for i in range(len(X_test)):

        print('Test for day '+str(i))
        error = list()
        for j in range(n_iterations):
            # generate points in latent space
            latent_points = generate_latent_points(latent_dim, 1)
            ts = model.predict([latent_points, df_modos.iloc[i].values])
            ts = np.squeeze(scaler.inverse_transform(np.squeeze(ts,axis=2)))
            error.append(sqrt(mean_squared_error(X_test[i],ts)))
        
        error_ATR.append(np.mean(error))
        
    
    
    d[ATR_colection[selec]]= np.array(error_ATR)


final = pd.DataFrame(d)
final.to_csv('results_cGAN.csv')

















