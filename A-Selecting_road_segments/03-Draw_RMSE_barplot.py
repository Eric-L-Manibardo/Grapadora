#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 14:49:45 2021

@author: eric


"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



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


# =============================================================================
# Start of the script
# =============================================================================

df_metrics = pd.read_csv('error_metrics.csv', index_col=0).T

df_mean_flow = pd.read_csv('flow_mean_weekdays_single_value.csv', index_col=0)


'''
necesito hacer una columna tipo HUE, para que me agrupe.
eje y: RMSE
eje x: ATR ID
hue: categorical{embedding,coordenadas}
'''
# creamos un nuevo dataframe
a=pd.concat([df_metrics['RMSE_embedding'],df_metrics['RMSE_coordenadas']])

b=['Embedding','Naive']
c,d=list(),list()
for j in range(2):
    for i in range(55):
        c.append(b[j])
        d.append(i+1)
        
c=pd.Series(np.array(c))
d=pd.Series(np.array(d))
f=pd.DataFrame({'RMSE':a.values,'Selection method':c.values,'Target road segment':d.values})

ax = sns.barplot(data=f,x='Target road segment',y='RMSE',hue='Selection method',edgecolor="black")



#dibujamos RMSE lowest y el mean flow de los weekdays de cada espira
a = df_metrics['RMSE_lowest'].values
b = df_mean_flow.values
for i in range(55):
    plt.hlines(xmin=i-0.4, xmax=i+0.4  ,y=a[i],color='black',linewidth=2)
    # plt.hlines(xmin=i-0.4, xmax=i+0.4  ,y=b[i],color='red',linewidth=2)
    plt.scatter(x=i,y=b[i], marker='P',zorder=10, color='black',s=100 )


ax.set_xlim([-1,55])
ax.set_ylim([0,1200])






















































