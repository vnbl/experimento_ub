#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 10:30:49 2020

@author: fernanda
"""

import pandas as pd #Para manipular datasets
import numpy as np #Funciones matematicas de manera vectorial
import random as rand
import sklearn.metrics as metrics # Para utilizar algoritmos de machine learning. Metrics: utiliza m'etricas de desempeno, funciones de score, etc. de forma a cuantificar la calidad de predicciones
import seaborn as sns # Visualizacion de datos

# matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.patches as patches
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.3)

#Pruebas

s1_info = [364288, 364289, 364290, 364291, 364292, 364293, 364294, 364298, 364299, 364301]
s1_info_tag = ['P1', 'DDB', 'IO', 'ALGE', 'CAL', 'MD', 'FIS', 'ALGO', 'P2', 'ED']
s2_info = [364297, 364300, 364303, 364305, 364302, 364296, 364295, 364306, 364304, 364307]
s2_info_tag = ['ELEC', 'AA', 'DS', 'EC', 'ICC', 'EMP', 'PIE', 'PAE', 'PIS', 'SO1']
s3_info = [364314, 364308, 364322, 364315, 364309, 364311, 364323, 364328, 364310, 364312]
s3_info_tag = ['IA', 'SO2', 'TNUI', 'VA', 'XAR', 'BD', 'FHIC', 'GiVD', 'LIL', 'SWD']


#concatenacion de datos
info_ids = s1_info + s2_info + s3_info
info_tags = s1_info_tag + s2_info_tag + s3_info_tag



# Matematica
s1_mates = [360142, 360140, 360136, 360138, 360134, 360135, 360139, 360143, 360137, 360141]
s1_mates_tag = ['ADIP', 'ELPR', 'IACD', 'LIRM', 'MAVE', 'ALLI', 'ARIT', 'FISI', 'IACI', 'PRCI']
s2_mates = [360144, 360148, 360151, 360150, 360146, 360145, 360152, 360161, 360153, 360155]
s2_mates_tag = ['CDDV', 'ESAL', 'GELI', 'GRAF', 'MNU1', 'CIDV', 'GEPR', 'HIMA', 'MMSD', 'TOPO']
s3_mates = [360158, 360149, 360156, 360147, 360162, 360159, 360154, 360163, 360160, 360157]
s3_mates_tag = ['ANMA', 'EQAL', 'GDCS', 'MNU2', 'PROB', 'ANCO', 'EQDI', 'ESTA', 'MODE', 'TGGS']

mates_ids = s1_mates + s2_mates + s3_mates
mates_tags = s1_mates_tag + s2_mates_tag + s3_mates_tag

# Derecho
s1_dret      = [362441, 362442, 362444, 362451, 362446, 362443, 362452, 362449, 362450, 362447]
s1_dret_tag  = ['TTC', 'CP', 'FDD', 'DRO', 'PIC', 'EC', 'SDL', 'FDPTD', 'HD', 'DCP']
s2_dret      = [362448, 362453, 362454, 362456, 362459, 362461, 362469, 362458]
s2_dret_tag  = ['OTE', 'PD', 'DOC', 'DIC', 'DFT', 'FDA', 'DPC', 'IDCE']
s3_dret      = [362507, 362460, 362462, 362466, 362465, 362470, 362467, 362463]
s3_dret_tag  = ['DR', 'PST', 'CAA', 'DEM', 'DTS', 'DPP', 'DS', 'BPU']

dret_ids     = s1_dret + s2_dret + s3_dret
dret_tags    = s1_dret_tag + s2_dret_tag + s3_dret_tag



# Educacion
s1_edu       = [361020, 361032, 361039, 361041, 361044, 361046, 361047, 361049, 361094]
s1_edu_tag   = ['PIP', 'PED', 'PDAA', 'AT', 'SOC', 'LCAT', 'LESP', 'DIDA', 'LENG']
s2_edu       = [361029, 361036, 361051, 361069, 361072, 361087, 361091, 361099, 361704]
s2_edu_tag   = ['INCL', 'SEOE', 'DIDC', 'DIDM', 'CINA', 'PLST', 'EDFI', 'PRAC', 'DGEO']

edu_ids      = s1_edu + s2_edu
edu_tags    = s1_edu_tag + s2_edu_tag

##### Importamos los datos

raw_grades_mates = pd.read_csv("data/grades_mates_2010-2016.csv", index_col=0) #index_col esta asociado a la primera columna, que es de
raw_grades_info  = pd.read_csv("data/grades_info_2011-2017.csv", index_col=0) # el id_alumno.
raw_grades_edu   = pd.read_csv("data/grades_edu_2009-2014.csv", index_col=0)
raw_grades_dret  = pd.read_csv("data/grades_dret_2009-2015.csv", index_col=0)

print("Notas")
print(raw_grades_mates)
print("Fin Notas")

##### Filtramos los datos

####### hacer ya una base de datos limpia y usar esa.

def filter_dataset(grades, t1, t2, t3, th1=8, th2=7, th3=0, gt=11, fill="row"):
    ''' Pivots raw datasets and cleans / fills missing data, returns tuple of filtered, all and filled'''
    _grades = grades.copy()
    _grades_all = _grades.copy() #si usamos el método tradicional para asignar, se modifica el dato original, por eso usamos copy()

    # Separamos por anho, aplicamos threshold, unimos
    _grades_first = _grades[t1]
    _grades_first = _grades_first.dropna(thresh=th1) #si hay al menos 8, no echa (este numero parece ser arbitrario)

    _grades_second = _grades[t2]
    _grades_second = _grades_second.dropna(thresh=th2) # si hay al menos 7, no echa

    _grades_third = _grades[t3]
    _grades_third = _grades_third.dropna(thresh=th3) # no echa, si hay al menos 0 (que siempre hay)

# Los threshholds difieren aca segun lo que dice en el paper.


    # # Join back as "inner" Concatenacion de datos de 1er anho, 2ndo y 3ro FILTRADO
    ## join(how = inner) > interseccion entre datos de alumnos de primer, segundo y tercer semestre (solo une los que tienen datos en los 3 semestres)
    _grades = _grades_first.join(_grades_second, how="inner").join(_grades_third, how="inner")

    print("all samples      ", _grades_all.count().sum())
    print("cleaned samples  ", _grades.count().sum())
    print("total students   ", _grades_all.shape[0])
    print("sampled students ", _grades.shape[0])


#### ESTO NO NOS GUSTA.
    if fill != 'row':
        # Fill with column mean
        _grades_fill = _grades.fillna(_grades.mean().round(1)) #que onda este redondeo tekk

    else:
        # Fill with row mean
        # Axis 1= fila, axis 0 = columna 
        ######## ESTA FUNCION NO ENTENDEMOS BIEN, ESTUDIARLA MAS 
        _row_mean = pd.DataFrame({col: _grades.mean(axis=1).round(1) for col in _grades.columns}) # for abreviado?
        _grades_fill = _grades.fillna(_row_mean)

    return _grades, _grades_all, _grades_fill
# _grades = todas las notas habiendo eliminado a los alumnos que no cumplen la condicion del problema
# _grades_all = todas las notas incluyendo a los alumnos que no cumplen la condicion del problema
# _grades_fill = datos filtrados, pero con las notas faltantes estimadas con ese round raro.

# Imprimimos los resultados


def combinarArchivos(df1, df2):
    
    dfi = df1.merge(df2, how = "inner", on = df1.index) #inteserccion
    
    sid_col = pd.DataFrame()
    
    for s
    
    
    
    
    
    
    
    

