#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 10:39:22 2021

@author: eric
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from numpy.random import randn
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['axes.titlesize'] = 20
sns.set_style("whitegrid")


ATR_colection = [3452, 3460, 3518, 3580, 3581, 3642, 3909, 3910, 3912, 3913, 3917,
       3918, 3919, 3921, 3926, #5467,
       5474, 5476, 5477, 5480, 5481, 5482, 
       5499, 5506, 5530, 5537, 5538, 5539, 5542, 5543, 5544, 5546, 5547, 
       5549, 5555, 5556, 5558, 5562, 5563, 5565, 5568, 5570, 5572, 5576, 
       5578, 5579, 5581, 5583, 5588, 5592, 5600, 5607, 5611, 5616, 5620, 6980] #7085#]

methods = ['Replica', 'Cluster','cGAN']

df_cluster = pd.read_csv('results_clusters.csv').T
#remove first row
df_cluster = df_cluster.iloc[1:]

df_register = pd.read_csv('results_register.csv').T
#remove first row
df_register = df_register.iloc[1:]

df_gan = pd.read_csv('results_cGAN.csv').T
#remove first row
df_gan = df_gan.iloc[1:]

#comparamos las medias para cada ATR
tipo=list()
for i in range(len(df_register)):

    comp = [np.mean(df_register.loc[str(ATR_colection[i])]),
            np.mean(df_cluster.loc[str(ATR_colection[i])]),
            np.mean(df_gan.loc[str(ATR_colection[i])])]
    #guardamos el valor
    tipo.append(methods[np.argmin(comp)])
    
#generamos cada una de las 3 series del DF a plotear
road, hue,error = list(), list(), list()
for i in range(len(df_register)):
    for j in range(730):
        
        # road.append(ATR_colection[i]) # descomentar para ATR IDs en el eje X
        road.append(i+1)
        hue.append(tipo[i])
        #errores
        if tipo[i]== 'Replica':
            error.append(df_register.iloc[i][j])
        elif tipo[i]== 'Cluster':
            error.append(df_cluster.iloc[i][j])
        elif tipo[i]== 'cGAN':
            error.append(df_gan.iloc[i][j])
        
df_draw = pd.DataFrame({'Target road segment':road,'RMSE':error, 'Generation approach':hue})
        
# flier = outlier
flierprops = dict(marker='x', markerfacecolor='None', markersize=4,  markeredgecolor='grey')


ax = sns.boxplot(data=df_draw, x='Target road segment', y='RMSE', hue='Generation approach',flierprops=flierprops, dodge=False,width=0.6)
ax.set_ylim([0,1100])

# plt.xticks(rotation = 90)


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    