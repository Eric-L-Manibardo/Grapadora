#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 12:01:23 2021

@author: eric

añade una etiqueta numérica de acuerdo al dia de la semana y si es festivo o no

"""

import json
import pandas as pd
import numpy as np

# Opening JSON file
f = open('madrid_holidays_2016_2019.json',)
  
# returns JSON object as a dictionary
holidays = json.load(f)

df = pd.read_csv('C-colombia_BATCH/3452_B.csv', index_col=0)

dates= df.index

'''las fechas no tienen el mismo formato que el JSON.
Estan del reves, los / deben se - y hay que quitar los 0s de las decenas en dia y mes'''

festivo_mask = np.zeros(len(dates))
for i in range(len(dates)):
    #slicing
    cortes = dates[i].split('/')
    #days
    try:
        a=cortes[2].replace('0','')
    except:
        a=cortes[2]
    #months
    b1,b2 = cortes[1]
    if b1=='0':
        b=b2
    else:
        b=cortes[1]
    
    # string a encontrar en holidays
    fecha = a +str('-')+ b +str('-')+ cortes[0]
    
    #sacamos un array boolean con las fiestas y el index del dia
    if fecha in holidays['academia_holiday'] or fecha in holidays['local_holiday'] or fecha in holidays['national_holiday'] :
        # print('fiesta')
        festivo_mask[i] = 1

'''Ahora que tenemos marcados los festivos, los combinamos con el dia de la semana.
 El 01/01/2018 fue lunes
 cada dia de la semana es su numero cardinal +7 si es festivo. e.g. 8 para lunes festivo'''

clase = list()
count = 0
for i in range(len(dates)):
    clase.append(count + (7*festivo_mask[i]))
    count += 1
    if count == 7:
        count=0
    
clases = pd.DataFrame(clase, columns=["Modo"])
clases.to_csv('modos_colombia_14.csv', index=False)

'''vamos a agrupar lunes-jueves en una misma etiqueta'''

clase2 = list()
for i in range(len(dates)):
    if clase[i] < 4:
        #lunes-jueves laboral
        clase2.append(0)
    elif clase[i] == 4:
        clase2.append(1)
    elif clase[i] == 5:
        clase2.append(2)
    elif clase[i] == 6:
        clase2.append(3)  
    elif 6 < clase[i] < 11:
        #lunes-jueves festivo
        clase2.append(4)  
    elif clase[i] == 11:
        clase2.append(5)
    elif clase[i] == 12:
        clase2.append(6)
    elif clase[i] == 13:
        clase2.append(7) 

clases = pd.DataFrame(clase2, columns=["Modo"])
clases.to_csv('modos_colombia_8.csv', index=False)   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    