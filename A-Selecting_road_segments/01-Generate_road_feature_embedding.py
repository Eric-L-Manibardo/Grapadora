#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 12:41:58 2021

There is a timer when using the REST service so the server do not overload,
so be patient when executing the script if several queries are launched


This script downloads graph representations from OpenstreetMap.com
Computes the features for road feature embedding representation
Save these features in a CSV file

@author: eric
"""
import pandas as pd
import networkx as nx
import osmnx as ox
from math import radians, sin, sqrt, cos, atan2
import numpy as np

# =============================================================================
# FUNCTIONS
# =============================================================================

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




def spbc_neighbours(egoList, neighbourNodes, norm):
        results = list()
        for i in range(len(egoList)):
            spbc = nx.betweenness_centrality(egoList[i], normalized=norm, weight='travel_time')
            results.append(spbc[neighbourNodes[i]])
            
        return np.median(results), max(results), results.index(max(results))
    
def spbc_closest_motorway_primary( G, node_aux, via):
        select=-1
        if via =='motorway':select=0
        elif via == 'primary':select=1
    
        
        # encuentro los edge tipo motorway/primary en todo el grafo G
        edges_aux = list(G.edges(data=True))
        edge_highway_pos = list()
        for i in range(len(edges_aux)):
            via = edges_aux[i][2]['highway']
            
            if select==0:
                if via == 'motorway' or via == 'motorway_link':
                    edge_highway_pos.append(i)
            elif select==1:
                if via == 'primary' or via == 'primary_link':
                    edge_highway_pos.append(i)
        
        # en caso de que no existan vias del tipo especificado cercanas
        if len(edge_highway_pos) == 0:
            return -1,-1,-1,-1
        
        
        # compute the shortest path to nodes u and v of the selected edges
        tt, hop, destinos = list(), list(), list()
        edge_highway_pos = np.array(edge_highway_pos)
        for i in range(len(edge_highway_pos)):
            u = edges_aux[edge_highway_pos[i]][0]
            v = edges_aux[edge_highway_pos[i]][1]
            utt, vtt = 99999999, 99999999
            utt_hop, vtt_hop = 999, 999
            try:
                utt = nx.shortest_path_length(G, source= node_aux, target=u, weight='travel_time')
                utt_hop = nx.shortest_path_length(G, source= node_aux, target=u)
            except:
                # print('Node '+str(u)+' not reacheable from node_aux')
                pass
            try:
                vtt = nx.shortest_path_length(G, source= node_aux, target=v, weight='travel_time')
                vtt_hop = nx.shortest_path_length(G, source= node_aux, target=v)
            except:
                # print('Node '+str(v)+' not reacheable from node_aux')
                pass
            if utt != 99999999 and vtt != 99999999: 
                # store minimum tt and the respective number of hops
                a = [utt,vtt]
                b = [utt_hop, vtt_hop]
                tt.append(min(a))
                hop.append(b[a.index(min(a))])
                destinos.append( edges_aux[edge_highway_pos[i]][a.index(min(a))])
                
        if len(tt)==0:
            return -1,-1,-1,-1
        else:
            #filter shortest paths with a diference less than a minute
            tt, hop = np.array(tt), np.array(hop)    
            # nº saltos al edge autpista mas cercana y su travel time
            edge_destino = edges_aux[edge_highway_pos[np.where(tt==min(tt))[0][0]]]
            hop_destino  = hop[np.where(tt==min(tt))[0][0]]
            tt_destino   = tt [np.where(tt==min(tt))[0][0]]
            node_destino = destinos[np.where(tt==min(tt))[0][0]]
            
        
            return  edge_destino, hop_destino, tt_destino, node_destino

def tipo_via(centralEdge):
    via = centralEdge['highway']
    if via == 'motorway' or via == 'motorway_link':
        v = 2
    elif via == 'primary' or via == 'primary_link':
        v = 1.5
    elif via == 'secondary' or via == 'secondary_link':
        v = 1
    elif via == 'tertiary' or via == 'tertiary_link':
        v = 0.5 
    elif via == 'residential' or via == 'living_street':
        v = 0 
    else:
        print('UNKNOWN ROAD TYPE!!!!1 *************************')
        
    return v

 


# =============================================================================
# START!!!
# =============================================================================

# ATR's ID colection
ATR = [3452, 3460, 3518, 3580, 3581, 3642, 3909, 3910, 3912, 3913, 3917,
       3918, 3919, 3921, 3926, #5467,
       5474, 5476, 5477, 5480, 5481, 5482, 
       5499, 5506, 5530, 5537, 5538, 5539, 5542, 5543, 5544, 5546, 5547, 
       5549, 5555, 5556, 5558, 5562, 5563, 5565, 5568, 5570, 5572, 5576, 
       5578, 5579, 5581, 5583, 5588, 5592, 5600, 5607, 5611, 5616, 5620, 6980] #7085#]


df_coo_espiras = pd.read_csv('coordenadas_sensores.csv', index_col=0)
df_coo_nodos_u = pd.read_csv('coordenadas_nodos_U.csv', index_col=0)
df_coo_nodos_v = pd.read_csv('coordenadas_nodos_V.csv', index_col=0)

for selec in range(len(ATR)):

    
    ox.config(use_cache=True, log_console=True)
    
    #coordenadas de cada ATR, nodo_U y nodo_V
    latlon_espira = df_coo_espiras.loc[ATR[selec]]
    latlon_nodo_u = df_coo_nodos_u.loc[ATR[selec]]
    latlon_nodo_v = df_coo_nodos_v.loc[ATR[selec]]
    
    # download street network data from lat-long point of the loop in a radius of 2km
    #  dist = retain only those nodes within this many meters of the center of the graph
    # 'drive' get only drivable streets, vehycles
    G = ox.graph_from_point((latlon_espira[0], latlon_espira[1]), dist=2000, network_type="drive", simplify=True, clean_periphery=True)
     
    
    # impute edge (driving) speeds and calculate edge travel times in seconds
    # precision modified to round to decenas
    G = ox.add_edge_speeds(G,precision=-1)
    G = ox.add_edge_travel_times(G)
    
    # in order to keep all features and lat,lon
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
    
   

    G = ox.utils_graph.get_digraph(G)
    #Search for central edge in graph, ya que te no te devuelve ningun ID
    arrayLat = gdf_nodes['y'].values
    arrayLon = gdf_nodes['x'].values
    posNode_u = encontrarPuntoMasCercano(latlon_nodo_u[0], latlon_nodo_u[1], arrayLat , arrayLon)
    posNode_v = encontrarPuntoMasCercano(latlon_nodo_v[0], latlon_nodo_v[1], arrayLat , arrayLon)
    
    print('posicion node_u: ('+str(arrayLat[posNode_u]) +', '+str(arrayLon[posNode_u])+')'  )
    print('posicion node_v: ('+str(arrayLat[posNode_v]) +', '+str(arrayLon[posNode_v])+')'  )
    
   
    node_u = gdf_nodes.iloc[posNode_u].name
    node_v = gdf_nodes.iloc[posNode_v].name
    node_aux = 999
    
    #get loop's edge information and remove it from graph, para crear un nodo central justo en el centro de la target road donde se encuentra el ATR
    if G.get_edge_data(node_u,node_v):
        print('dirección (u , v)')
        centralEdge = G.get_edge_data(node_u,node_v)
        G.remove_edge(node_u, node_v)
        G.add_edge(node_u, node_aux, oneway= centralEdge['oneway'], highway=centralEdge['highway']
                   , length=centralEdge['length']/2, 
                   speed_kph=centralEdge['speed_kph'], travel_time=centralEdge['travel_time']/2)
        G.add_edge(node_aux, node_v, oneway= centralEdge['oneway'], highway=centralEdge['highway']
                   , length=centralEdge['length']/2, 
                   speed_kph=centralEdge['speed_kph'], travel_time=centralEdge['travel_time']/2)
        #para saber por donde se sale de esa via
        node_out=node_v
    # en caso de que el enlace sea (v,u)
    else:
        print('dirección (v , u)')
        centralEdge = G.get_edge_data(node_v,node_u)
        G.remove_edge(node_v, node_u)
        G.add_edge(node_v, node_aux, oneway= centralEdge['oneway'], highway=centralEdge['highway']
                   , length=centralEdge['length']/2, 
                   speed_kph=centralEdge['speed_kph'], travel_time=centralEdge['travel_time']/2)
        G.add_edge(node_aux, node_u, oneway= centralEdge['oneway'], highway=centralEdge['highway']
                   , length=centralEdge['length']/2, 
                   speed_kph=centralEdge['speed_kph'], travel_time=centralEdge['travel_time']/2)
        #para saber por donde se sale de esa via
        node_out=node_u
    
    
    # =============================================================================
    # ego networks 
    # =============================================================================
    #debemos recoger en undirected, nodos entrantes y salientes
    egoBi_h5 = nx.ego_graph(G, node_aux, radius=5, undirected=True) 
    
    
    #collect node ID from neihgbours
    listNodes = list(egoBi_h5.nodes(data=True))
    neighbourNodes = list()
    for i in range(len(listNodes)):
        if listNodes[i][0] != node_aux:
            neighbourNodes.append(listNodes[i][0])
    
    #obtain ego_graph from each neighbour
    egoNeighboursBi = list()
    for i in range(len(neighbourNodes)):

        egoNeighboursBi.append(nx.ego_graph(G, neighbourNodes[i], radius=5 ))
    
    
    
    
    features = {}

    
    # =============================================================================
    # CENTRALITY features
    # =============================================================================
    
    '''
    Si un grafo existen X combinaciones de nodos origen-destino entre todos sus nodos, el spbc devuelve
    el numero de rutas que son SPs y pasan por el nodo target (de acuerdo al peso de las aristas, que
    por defecto es igual a 1). Si no se normaliza entre el numero de caminos, el spbc devuelto es la 
    suma de los pesos de todos los caminos que pasen por target.
    Por lo tanto, aquellos nodos target con pesos de aristas elevados suman mucho por cada nuevo SP.
    Sigo sin saber si es buena idea usar los valores ABSolutos.
    
    '''
    
    # feature 1
    #shortest path betweenness centrality of aux node
    spbc_abs = nx.betweenness_centrality(egoBi_h5, normalized=False, weight='travel_time')
    features['spbc_aux_abs']=spbc_abs[node_aux]
    
    
    # feature 2 y 3
    features['spbc_neigbours_median_abs'],  features['spbc_neighbours_max_abs'], max_id_abs  = spbc_neighbours(egoNeighboursBi,  neighbourNodes, False)

 
    # =============================================================================
    # centralidad respecto motorway/primary mas cercano
    # =============================================================================
    '''
    Tengo que encontrar la mas cercana por travel time o en numero de hops.
    Primero busco 2 con travel time y luego me quedo el de menor numero de hops 
    si la diferencia es menor a un minuto
    '''
    # features 4 y 5
    edge_motorway, hop_motorway, tt_motorway, node_motorway = spbc_closest_motorway_primary(G, node_out,'motorway')    
    features['tt_2motorway']  = tt_motorway    
    edge_primary, hop_primary, tt_primary, node_primary = spbc_closest_motorway_primary( G, node_out,'primary')    
    features['tt_2primary']  = tt_primary
    

                
    #feature 6
    ''' 
    Road type
    '''
    features['via'] = tipo_via(centralEdge)
    
    #feature 7
    '''
    number of lanes 
    '''
    try:
        #en caso de que sea string, lo quiero como int
        features['n_lanes'] = int(centralEdge['lanes'])
    except:
        #en caso de que no exista el campo
        features['n_lanes'] = 0
        
    features['ATR_ID']=ATR[selec]

# =============================================================================
# store features to dataframe
# =============================================================================
    if selec==0:
        df_features = pd.DataFrame.from_dict(features, orient='index')
    else:
        df_features[selec] = pd.Series(features)
        
    
df_features.to_csv('road_feature_embedding_grafos_chamartin.csv')






















