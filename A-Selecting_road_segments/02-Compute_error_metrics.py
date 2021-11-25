#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 14:49:45 2021

@author: eric

calcula y almacena en CSV nRMSE, distancias euclideas y posiciones relativas respecto del desempeño nRMSE

Compute and store in CSV file for road feature embedding method and geographical distance:
    - Euclidean distances of their road feature embeddings
    - RMSE
    - nRMSE
    - ranking positions (not used in paper)

"""


import pandas as pd
from math import radians, sin, sqrt, cos, atan2
import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


def distanciasLatLon(lat1, lon1, lat2, lon2):
    '''
    
    Returns distancia entre dos puntos del globo
    -------
    distance : TYPE
        DESCRIPTION.

    '''
    R = 6373.0 # radio de la Tierra 
   
    lat1 = radians(float (lat1))
    lon1 = radians(float (lon1))
    lat2 = radians(float (lat2))
    lon2 = radians(float (lon2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c   
    return distance

def encontrarPuntoMasCercano(lat, lon, listaLat, listaLon):
    '''
    Parameters
    ----------
    lat : of the point of interest
    lon : of the point of interest
    listaLat : array float
    geographical points to be checked
    listaLon : array float
    geographical points to be checked

    Returns: index of chosen geograficalpoint 
    -------
    None.

    '''
    elegido = -100
    distanciaminima = 99999999999
#    print "lat,lon oringal:"+str(lat)+","+str(lon)
    for k in range(len(listaLat)):
        distancia = distanciasLatLon(lat, lon, listaLat[k], listaLon[k])
        if distancia < distanciaminima:
            elegido = k
            distanciaminima = distancia
    return elegido


def distancia_euclidea(X):
    # se puede asignar unos pesos a cada feature
    dist_pareja, dist_pareja2, index_pareja, index_pareja2 = list(), list(), list(), list()
    for i in range(len(X)):
        dist =list()
        for j in range(len(X)):
            dist.append(distance.euclidean(X[i], X[j]))
        
        dist_sort=np.sort(dist)
        #la primera posicion es el mismo vector en si
        dist_pareja.append(dist[np.where(np.array(dist)==dist_sort[1])[0][0]])
        dist_pareja2.append(dist[np.where(np.array(dist)==dist_sort[2])[0][0]])
        index_pareja.append(np.where(np.array(dist)==dist_sort[1])[0][0])
        index_pareja2.append(np.where(np.array(dist)==dist_sort[2])[0][0])
        
    dist_pareja = np.array(dist_pareja)
    dist_pareja2 = np.array(dist_pareja2)
    index_pareja = np.array(index_pareja)
    index_pareja2 = np.array(index_pareja2)
    return dist_pareja, dist_pareja2, index_pareja, index_pareja2

def distancia_euclidea_v2(X):
    # se puede asignar unos pesos a cada feature
    dist_pareja, index_pareja = list(), list()
    for i in range(len(X)):
        dist =list()
        for j in range(len(X)):
            dist.append(distance.euclidean(X[i], X[j]))
        
        dist_sort=np.sort(dist)
        #la primera posicion es el mismo vector en si
        dist_pareja.append(dist[np.where(np.array(dist)==dist_sort[1])[0][0]])
        
        #en caso de que el embedding sea idéntico entre dos ATRs asignemos el otro, y no el mismo
        if np.where(np.array(dist)==dist_sort[1])[0][0]==i:
            index_pareja.append(np.where(np.array(dist)==dist_sort[1])[0][1])
        else:
            index_pareja.append(np.where(np.array(dist)==dist_sort[1])[0][0])

        
    dist_pareja = np.array(dist_pareja)
    
    index_pareja = np.array(index_pareja)
   
    return dist_pareja, index_pareja

def distancia_euclidea_lowest(X,pos_lowest):
    # se puede asignar unos pesos a cada feature
    dist_pareja, index_pareja = list(), list()
    for i in range(len(X)):
        dist =list()
        for j in range(len(X)):
            dist.append(distance.euclidean(X[i], X[j]))
        
        dist_sort=np.sort(dist)
        #la primera posicion es el mismo vector en si
        dist_pareja.append(dist[np.where(np.array(dist)==dist_sort[1])[0][0]])
        
        #en caso de que el embedding sea idéntico entre dos ATRs asignemos el otro, y no el mismo
        if np.where(np.array(dist)==dist_sort[1])[0][0]==i:
            index_pareja.append(np.where(np.array(dist)==dist_sort[1])[0][1])
        else:
            index_pareja.append(np.where(np.array(dist)==dist_sort[1])[0][0])

        
    dist_pareja = np.array(dist_pareja)
    
    index_pareja = np.array(index_pareja)
   
    return dist_pareja, index_pareja

def distancia_rmse(flow):
    # se puede asignar unos pesos a cada feature
    dist_pareja, dist_pareja2, index_pareja, index_pareja2 = list(), list(), list(), list()
    for i in range(len(flow)):
        dist =list()
        for j in range(len(flow)):
            dist.append(sqrt(mean_squared_error(flow[i],flow[j])))
        
        dist_sort=np.sort(dist)
        #la primera posicion es el mismo vector en si
        dist_pareja.append(dist[np.where(np.array(dist)==dist_sort[1])[0][0]])
        dist_pareja2.append(dist[np.where(np.array(dist)==dist_sort[2])[0][0]])
        index_pareja.append(np.where(np.array(dist)==dist_sort[1])[0][0])
        index_pareja2.append(np.where(np.array(dist)==dist_sort[2])[0][0])
        
    dist_pareja = np.array(dist_pareja)
    dist_pareja2 = np.array(dist_pareja2)
    index_pareja = np.array(index_pareja)
    index_pareja2 = np.array(index_pareja2)
    return dist_pareja, dist_pareja2, index_pareja, index_pareja2

def rmse_ATR(flow, pos):
    '''
    Devuelve el RMSE sobre el resto de ATRs
    '''
    # pos = index ATR target
    metricas = list()
    
    for i in range(len(flow)):
        metricas.append(sqrt(mean_squared_error(flow[pos], flow[i])))
            
    return np.array(metricas)

def Nrmse_ATR(flow, pos):
    '''
    Devuelve el normalized RMSE sobre el resto de ATRs
    normalizado sobre el rango dinámico de la señal original.
    Tiene sentido penalizar más el mismo RMSE para señales de 
    menor rango dinámico
    '''
    # pos = index ATR target
    metricas = list()
    
    for i in range(len(flow)):
        denominador = max(flow[pos]) - min(flow[pos])
        metricas.append(sqrt(mean_squared_error(flow[pos], flow[i])) / denominador)
            
    return np.array(metricas)

def Nrmse_ATR_bymean(flow, pos,mean):
    '''
    Devuelve el normalized RMSE sobre el resto de ATRs
    normalizado sobre el mean flow de la señal original.
    Tiene sentido penalizar más el mismo RMSE para señales de 
    menor rango dinámico
    '''
    # pos = index ATR target
    metricas = list()
    
    for i in range(len(flow)):
        
        metricas.append(sqrt(mean_squared_error(flow[pos], flow[i])) / mean)
            
    return np.array(metricas)

# =============================================================================
# Start of the script
# =============================================================================


df_embeddings= pd.read_csv('road_feature_embedding_grafos_chamartin.csv', index_col=0)

# mean flow value of each location, considering only weekdays
df_mean_single = pd.read_csv('flow_mean_weekdays_single_value.csv',index_col=0)

#traffic profile of each location, considering only weekdays
df_flow = pd.read_csv('flow_medianas_weekdays.csv', index_col=0)
flow = df_flow.values

# ATR's ID colection
ATR_colection = [3452, 3460, 3518, 3580, 3581, 3642, 3909, 3910, 3912, 3913, 3917,
       3918, 3919, 3921, 3926, #5467,
       5474, 5476, 5477, 5480, 5481, 5482, 
       5499, 5506, 5530, 5537, 5538, 5539, 5542, 5543, 5544, 5546, 5547, 
       5549, 5555, 5556, 5558, 5562, 5563, 5565, 5568, 5570, 5572, 5576, 
       5578, 5579, 5581, 5583, 5588, 5592, 5600, 5607, 5611, 5616, 5620, 6980] #7085#]

#Normalization of road features
scaler = MinMaxScaler()
X_s = scaler.fit_transform(df_embeddings.T)

num_features =X_s.shape[1] 

# Return most similar road feature embedding
dist_pareja, pos_pareja = distancia_euclidea_v2(X_s)

results = {}
results['dist_embedding'] = dist_pareja

# =============================================================================
# Find nearest sensorized location by geographical distance 
# =============================================================================
df_coo_espiras = pd.read_csv('coordenadas_sensores.csv', index_col=0)

vecinos_por_coordenadas = list()
for i in range(len(flow)):
    lat_array = df_coo_espiras['lat'].values
    lat_array = np.delete(lat_array, i) #filtramos sus propias coordenadas
    lon_array = df_coo_espiras['lon'].values
    lon_array = np.delete(lon_array, i) #filtramos sus propias coordenadas
    name_array = df_coo_espiras.index.values
    name_array = np.delete(name_array,i)#filtramos su propio ID
    
    '''
    esta posicion no me sirve como definitiva, ya que es de una escala de 54 en la que falta la posicion i
    (la número 55). Es por ello que luego busco el nombre correspondiente (ID) y con este ID la posición seleccionada
    '''
    pos=encontrarPuntoMasCercano(df_coo_espiras.iloc[i][0],df_coo_espiras.iloc[i][1],lat_array, lon_array )
    
    vecinos_por_coordenadas.append(name_array[pos])
    
#convertimos a posiciones verdaderas
name_array = df_coo_espiras.index.values
pos_vecinos_coor=list()
for i in range(len(vecinos_por_coordenadas)):
    pos_vecinos_coor.append(np.where(name_array==vecinos_por_coordenadas[i])[0][0])
  

# =============================================================================
# Compute the RMSE of each selected location regarding the traffic profile
# =============================================================================
ee, ec, rank_e, rank_c,ee2,ec2, lowest_rmse, id_lowest, dist_lowest, lowest_nrmse = list(), list(), list(), list(),list(), list(),list(),list(),list(),list()
for j in range(len(flow)):
# =============================================================================
#   RMSE
# =============================================================================
    #sacamos el error respecto del flow del ATR[j]
    metricas= rmse_ATR(flow, j)    
    # guardamos el error del ATR closest de acuerdo a embeddings
    error_embedding = metricas[pos_pareja[j]]
    ee2.append(error_embedding)
    # guardamos el error del ATR closest de acuerdo a coordenadas
    error_coordenadas = metricas[pos_vecinos_coor[j]]
    ec2.append(error_coordenadas)
    
    metricas_sin = metricas.copy()
    metricas_sin.sort()
    metricas_sin= metricas_sin[1:]
    lowest_rmse.append(metricas_sin[0])
    
# =============================================================================
#     nRMSE (normalized by mean flow)
# =============================================================================
    #sacamos el error respecto del flow del ATR[j]
    metricas= Nrmse_ATR_bymean(flow, j, df_mean_single.iloc[j].values[0])    
    # guardamos el error del ATR closest de acuerdo a embeddings
    error_embedding = metricas[pos_pareja[j]]
    ee.append(error_embedding)
    # guardamos el error del ATR closest de acuerdo a coordenadas
    error_coordenadas = metricas[pos_vecinos_coor[j]]
    ec.append(error_coordenadas)
    
    

        
    metricas_sin = metricas.copy()
    metricas_sin.sort()
    metricas_sin= metricas_sin[1:]

    '''
    la posición se calcula desde 1 a 54, siendo 55 la posición 0 en caso de usar n/55
    '''
    rank_e.append(np.where(metricas_sin==error_embedding)[0][0]+1)
    rank_c.append(np.where(metricas_sin==error_coordenadas)[0][0]+1)
    lowest_nrmse.append(metricas_sin[0])
    
    '''
    ID del ATR que da el menor RMSE
    distancia euclidea lowest
    '''
    id_lowest.append(ATR_colection[np.where(metricas==min(metricas_sin))[0][0]])
    dist_lowest.append(distance.euclidean(X_s[j], X_s[np.where(metricas==min(metricas_sin))[0][0]]))


    
# =============================================================================
# guardamos a file el ID de los ATRs escogidos como similares por el sistema
# =============================================================================
results['dist_lowest'] = np.array(dist_lowest)
results['nRMSE_embedding'] = np.array(ee)
results['nRMSE_coordenadas'] = np.array(ec)
results['nRMSE_lowest'] = np.array(lowest_nrmse)
results['RMSE_embedding'] = np.array(ee2)
results['RMSE_coordenadas'] = np.array(ec2)
results['RMSE_lowest'] = np.array(lowest_rmse)
results['rank_embedding'] = np.array(rank_e)
results['rank_coordenadas'] = np.array(rank_c)




# =============================================================================
# comparativas de rankings
# =============================================================================
comp=np.array(rank_e)-np.array(rank_c)
results['RMSE_comparativa']=comp

a=list()
b,c,d =0,0,0
for i in range(len(comp)):
    if comp[i] < 0:
        #win
        a.append(1)
        b=b+1
    elif comp[i] > 0:
        #loses
        a.append(-1)
        c=c+1
    else:
        #draws
        a.append(0)
        d=d+1
print('Win: '+str(b)+' / Loses: '+str(c)+' / Draws: '+str(d))
        



a,b = list(), list()
for i in range(len(pos_pareja)):
    a.append(ATR_colection[i])
    b.append(ATR_colection[pos_pareja[i]])

results['target_ATR']=a
results['selected_ATR_embedding']=b
results['ID_lowest'] = np.array(id_lowest)
df = pd.DataFrame.from_dict(results, orient='index')

df.to_csv('error_metrics.csv')
































































