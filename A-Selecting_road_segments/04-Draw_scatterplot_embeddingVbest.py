#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 14:49:45 2021

@author: eric

calcula y almacena en CSV nRMSE, distancias euclideas y posiciones relativas respecto del desempeño nRMSE

"""


import pandas as pd
# import networkx as nx
# import osmnx as ox
from math import radians, sin, sqrt, cos, atan2
import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import json
import matplotlib.ticker as ticker



plt.rcParams['font.family'] = 'serif'
# plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 1
plt.rcParams['axes.labelsize'] = 1
# plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 1
plt.rcParams['ytick.labelsize'] = 1
plt.rcParams['legend.fontsize'] = 1
plt.rcParams['figure.titlesize'] = 1
sns.set(style="whitegrid")

df_metrics= pd.read_csv('error_metrics.csv', index_col=0).T

# 3 columnas: distancia, nRMSE, lowest/embedding
a=pd.concat([df_metrics['dist_embedding'],df_metrics['dist_lowest']],ignore_index=True)
b=pd.concat([df_metrics['nRMSE_embedding'],df_metrics['nRMSE_lowest']],ignore_index=True)
c=list()
for i in range(len(a)):
    if i <len(a)/2:
        c.append('Embedding')
    else:
        c.append('Best')
c= pd.Series(c)

#juntamos las series
df_draw=pd.DataFrame({'dist':a,'nrmse':b})
df_draw['Selection criteria']=c


# df_draw = df_draw.sort_values(by=['type'], ascending=True)
# g = sns.FacetGrid(df_metrics)
# a =sns.scatterplot(data=df_metrics, x='nRMSE_embedding', y='euclidean_dist', s=100, zorder=10)
ax =sns.scatterplot(data=df_draw, x='nrmse', y='dist', zorder=10 ,hue='Selection criteria', s=100,  palette='viridis_r', legend=True, style='Selection criteria',markers=['o', 'P'])

# ax.axhline(0,color='k',zorder=1, linewidth=3)
# ax.axvline(0,color='k',zorder=1, linewidth=3)
# plt.legend(title='Euclidean \n distance',loc='upper right')
ax.set(ylabel='Euclidean distance', xlabel='Normalized RMSE')
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.4))
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())





























































