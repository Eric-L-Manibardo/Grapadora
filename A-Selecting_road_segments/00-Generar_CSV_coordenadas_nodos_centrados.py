#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 16:32:18 2021

@author: eric
"""
import xmltodict
import pandas as pd

# with open('coordenadas_colombia_centrado.xml') as fd:
with open('coordenadas_colombia_nodos_V.xml') as fd:
    my_xml = xmltodict.parse(fd.read())


lat, lon, name= list(), list(), list()
for i in range(len(my_xml['kml']['Document']['Placemark'])):
    lat.append(float(my_xml['kml']['Document']['Placemark'][i]['ExtendedData']['Data'][0]['value']))
    lon.append(float(my_xml['kml']['Document']['Placemark'][i]['ExtendedData']['Data'][1]['value']))
    name.append(int(float(my_xml['kml']['Document']['Placemark'][i]['name'])))

df_latlon = pd.DataFrame({'id':name,
                          'lat':lat,
                          'lon':lon})

df_latlon.to_csv('coordenadas_colombia_nodos_V.csv', index=False)