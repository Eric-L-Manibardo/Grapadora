#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 09:47:47 2021

@author: eric

dibujar una comparativa en concreto
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.despine(left=True)
sns.set(style="whitegrid")

plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['axes.titlesize'] = 20

def compute_dfdays(selected_array):
    d_flow={}
    d_time={}
    k=0
    for i in range(selected_array.shape[0]):       
        for j in range(selected_array.shape[1]):        
            d_flow[k]=selected_array[i][j]
            d_time[k]=i
            k=k+1
    return pd.concat([pd.Series(d_time,name='time'),pd.Series(d_flow,name='flow')],axis=1)
        
def compute_split(df_flow):
    '''
    separamos findes de weekdays
    '''
    #2018-01-01 fue lunes
    selected_days, count, selected_weekends = list(), 0, list()
    for j in range(len(df_flow)):
        # de lunes a viernes
        if count<5:
            selected_days.append(df_flow.iloc[j].values)
        else:
            selected_weekends.append(df_flow.iloc[j].values)
        count = count+1
        if count == 7:
            count=0
    #computamos la mediana por intervalos
    selected_days = np.array(selected_days).T
    # selected_median = np.median(selected_days, axis=1)
    selected_weekends = np.array(selected_weekends).T
    return selected_days,selected_weekends


ATR = [3452, 3460, 3518, 3580, 3581, 3642, 3909, 3910, 3912, 3913, 3917,
       3918, 3919, 3921, 3926, #5467,
       5474, 5476, 5477, 5480, 5481, 5482, 
       5499, 5506, 5530, 5537, 5538, 5539, 5542, 5543, 5544, 5546, 5547, 
       5549, 5555, 5556, 5558, 5562, 5563, 5565, 5568, 5570, 5572, 5576, 
       5578, 5579, 5581, 5583, 5588, 5592, 5600, 5607, 5611, 5616, 5620, 6980] #7085#]

medianas_dataset = list()




# =============================================================================
# plot zone
# =============================================================================
# fig, axs = plt.subplots(3, 1, sharex=True, sharey=False)
# (ax1, ax2, ax3) = axs

#AX1 target
i=14   #selecciona espira
df_flow = pd.read_csv('C-colombia_BATCH/'+str(ATR[i])+'_B.csv',index_col=0)
selected_days, _ = compute_split(df_flow)

# dibujamos la +-std de cada uno de las muestras disponibles para cada timestamp 
sns.lineplot(data=compute_dfdays(selected_days),x='time',y='flow',estimator=np.median,ci='sd', legend='auto')    

# embedding
i=6  #selecciona espira
df_flow = pd.read_csv('C-colombia_BATCH/'+str(ATR[i])+'_B.csv',index_col=0)
selected_days, _ = compute_split(df_flow)

# dibujamos la +-std de cada uno de las muestras disponibles para cada timestamp 
sns.lineplot(data=compute_dfdays(selected_days),x='time',y='flow',estimator=np.median,ci='sd', legend='auto')   

# embedding
i=10  #selecciona espira
df_flow = pd.read_csv('C-colombia_BATCH/'+str(ATR[i])+'_B.csv',index_col=0)
selected_days, _ = compute_split(df_flow)

# dibujamos la +-std de cada uno de las muestras disponibles para cada timestamp 
ax1= sns.lineplot(data=compute_dfdays(selected_days),x='time',y='flow',estimator=np.median,ci='sd', legend='auto')   



# ESTETICA
eje_x = pd.date_range(start='2018/01/01',periods=96,freq='15MIN' )
hour_minutes = eje_x.strftime('%H:%M')
#frecuencia de las x labels
frec= 8
plt.xticks(range(len(hour_minutes))[::frec], hour_minutes[::frec], size='small', rotation=45, horizontalalignment='center')
# ax1.legend()

plt.legend(['Target \#15', 'Embedding \#7', 'Best \#11'], loc='upper left')
plt.xlabel('Time of day')
ax1.set_ylabel('Flow')
plt.title('Traffic profile of road segment \#15')

plt.xlim([0,95])
plt.ylim([0,1500])
ax1.set_ylim(bottom=0)

# ax1.title.set_text('ATR 2')


# fig.subplots_adjust(hspace=0.132, wspace=0.200, top=0.988, bottom=0.109, left=0.156,right=0.993)

















































