import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as st
from datetime import datetime,date
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pyod.models.knn import KNN
from joblib import dump, load
from os import path 
import timeit
import logging
import shutil



#timpo de ejecucion
start = timeit.default_timer()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s',datefmt='%d-%m-%Y')
handler_log = logging.FileHandler("logging.log")
handler_log.setLevel(logging.DEBUG)
handler_log.setFormatter(formatter)
logger.addHandler(handler_log)


logger.info("----Inicio de Estimacion Min Max----")




#Path
name_maestro_establecimientos = "Input_fijo/maestro_establecimientos.xlsx"
name_transit_time  = "Input_fijo/2019-2022 DMS1__Transit Time.xlsx"
name_establecimientos = "Input/establecimientos.xlsx"
name_config = "Input/config.xlsx"

#Parametros del modelo a definir por configuracion
try:
    logger.info("Carga de Parametros")
    df_config = pd.read_excel(name_config)
    vel_min = int(df_config.loc[0,"vel_min"])
    vel_max = int(df_config.loc[0,"vel_max"])
    vel = int(df_config.loc[0,"vel_media"])
    alpha = float(df_config.loc[0,"confianza"]/100)
    hs_manejo = int(df_config.loc[0,"hs_manejo"])
    hs_descanso = int(df_config.loc[0,"hs_descanso"])
    segmentacion =  bool(1 if str(df_config.loc[0,"segmentacion"]).strip().lower() == "si"
                         else 0)
    logger.info(f"KNN Segmentacion es: {segmentacion}")
except Exception as ex:
    logger.warning("Error carga de parametros")
    quit()

# # Carga de datos
logger.info("Carga de Datos")
try:
    df_consolidado = pd.read_excel(name_transit_time,
                                sheet_name="Consolidado",
                                header=0,
                                dtype = {"tiempo_real":datetime},
                                names = ['id_dms',
                                            'año',
                                            'zone_x',
                                            'subZone',
                                            'batch',
                                            'truck_num',
                                            'truck Type',
                                            'delivery_num',
                                            'start_dt',
                                            'end_dt',
                                            'tiempo_min_zone',
                                            'tiempo_max_zone',
                                            'temp_min',
                                            'temp_max',
                                            'obs_temp',
                                            'tiempo_real',
                                            'obs_tiempo',
                                            'obs_peso',
                                            'kg',
                                            'establecimiento',
                                            'tiempo',
                                            'conteo_batch',
                                            'conteo_subzone',
                                            'logueo_manual',
                                        ]                                                                   
                                )

    df_distancias = pd.read_excel(name_establecimientos)
    duplicated = df_distancias.loc[df_distancias[["establecimiento","latitud","longitud"]].duplicated()].index
    df_duplicado = df_distancias.loc[df_distancias[["zone","establecimiento","latitud","longitud"]].duplicated()]
    df_distancias.drop(duplicated,inplace=True)
    df_distancias = df_distancias.reset_index(drop=True)
    establecimiento_upper = df_distancias.establecimiento.to_list()
    df_consolidado.establecimiento = df_consolidado.establecimiento.str.strip().str.lower()
    df_distancias.establecimiento = df_distancias.establecimiento.str.strip().str.lower()
    df_maestro_establecimientos = pd.read_excel(name_maestro_establecimientos, dtype={"latitud":"float","longitud":"float"})
except Exception as ex:
    logger.warning("Error al cargar archivos",exc_info=True)
    quit()
    
#Maestro de establecimientos

df_maestro_establecimientos.establecimiento = df_maestro_establecimientos.establecimiento.str.strip().str.lower()
df_maestro_establecimientos.establecimiento.drop_duplicates(inplace=True)

aux = df_distancias.loc[df_distancias.establecimiento.isin(df_maestro_establecimientos.establecimiento)==False,
                       ["establecimiento",
                       "latitud",
                       "longitud"]].copy()
if aux.shape[0]>0:
    df_maestro_establecimientos = pd.concat ([df_maestro_establecimientos, aux])
    df_maestro_establecimientos =  df_maestro_establecimientos.reset_index(drop=True)
df_maestro_establecimientos["Duplicado"] = 0

aux = df_distancias [["establecimiento","latitud","longitud"]]
aux = df_maestro_establecimientos.merge(aux,"left",on="establecimiento")
aux.dropna(subset=["latitud_y","longitud_y"], inplace=True)
aux ["Duplicado"] = aux.apply(lambda row: 1 if ((round(row.latitud_x,9) != round(row.latitud_y,9))|
                                        (round(row.longitud_x,9) != round(row.longitud_y,9))) else 0,axis=1)

aux = aux [["establecimiento","latitud_y","longitud_y","Duplicado"]]
aux.rename(columns={"latitud_y":"latitud","longitud_y":"longitud"},inplace=True)
aux = aux.loc[aux["Duplicado"] == 1]
if aux.shape[0]>0:
    df_maestro_establecimientos = pd.concat ([df_maestro_establecimientos, aux]).reset_index(drop=True)

df_maestro_establecimientos.loc[df_maestro_establecimientos.Duplicado==1,"Duplicado"] = "Diferencia geolocalizacion"


#Carga de modelo de Segmentacion
try:
    path_libs = "libs/"

    df_distancias ["duracion_cdescanso"] = df_distancias.apply(lambda x: 
                                                           ((x.distancia/vel)*60)/((hs_manejo))*hs_descanso
                                                           if ((x.distancia/vel)*60) > ((hs_manejo)*60)
                                                           else 0,
                                                           axis=1)
    df_distancias ["duracion_min_cdescanso"] = df_distancias["duracion_cdescanso"] + df_distancias["duracion_min"]

    df_cluster = df_distancias[[
                    'distancia_value_mts',
                    'duracion_min_cdescanso']].copy()
    kmeans = load(path.join(path_libs,"cluster_establecimientos.joblibs"))

    group = kmeans.predict(df_cluster)
    df_distancias ["group"] = group
    #Segementacion de los establecimientos mediante el modelo de clustering
    df_distancias["zone_group"] = ""
    zones = df_distancias.zone.value_counts().index.to_list()
    for zone in zones:
        groups = df_distancias.loc[df_distancias.zone == zone,"group"].value_counts().index.to_list()
        groups.sort()
        df_distancias.loc[df_distancias.zone==zone,"zone_group"] = df_distancias.loc[df_distancias.zone==zone].apply(lambda x: zone + str(groups.index(x.group)+1) if len(groups) > 1 else x.zone, axis=1) 

    resumen_segmentacion = df_distancias.pivot_table(values=["distancia","duracion_min_cdescanso"], index="group", aggfunc=["min","max"])
    resumen_segmentacion.columns = resumen_segmentacion.columns.to_flat_index()
    resumen_segmentacion.columns = ["Distancia Minima","Duracion Minima","Distancia Maxima","Duracion Maxima"]
    resumen_segmentacion = resumen_segmentacion.sort_values(by="Distancia Minima").reset_index(drop=True)
    resumen_segmentacion.reset_index(inplace=True)
    resumen_segmentacion.rename(columns={"index":"Grupo"},inplace = True)
    resumen_segmentacion["Grupo"] +=1
    resumen_segmentacion = resumen_segmentacion [["Grupo","Distancia Minima","Distancia Maxima","Duracion Minima","Duracion Maxima"]]   
except Exception as ex:
    logger.warning("Error al cargar modelo de segmentacion",exc_info=True)
    quit()
#Union de los establecimientos a los datos historicos
df_consolidado = df_consolidado.merge(df_distancias, "left", left_on="establecimiento", right_on="establecimiento")
#Mantiene la zona de segmentacion de cluster si es True caso contrario mantiene las zonas del cliente
if (segmentacion==True):
    df_consolidado.subZone = df_consolidado.zone_group
    df_distancias.zone =  df_distancias.zone_group
else:
    df_consolidado.subZone = df_consolidado.zone #valores segun el input

#Preprocesado de Variables

df_consolidado.tiempo_real.dropna(axis=0,inplace=True)
df_consolidado.drop(df_consolidado.loc[df_consolidado.tiempo_real.apply(lambda x: type(x) is float)].index, axis=0,inplace=True)

#tiempo real en horas
df_consolidado["tiempo_real"] = df_consolidado.tiempo_real.apply(lambda x: ( ((x.day*24)*60) + (x.hour*60) +(x.minute)  )                                 if isinstance(x,datetime)  else ((x.hour*60) + (x.minute)))
df_consolidado = df_consolidado[["subZone","establecimiento","distancia","tiempo_real"]]
#Eliminar outliers
logger.info("Eliminacion de outliers")
#velocidades para eliminar  outliers
min, max = 40,110
#Calculo de la Duracion media, minima y maxima

df_consolidado["duracion_cdescanso"] = df_consolidado.apply(lambda x: ((x.distancia/vel)*60),
                                                            axis=1)
df_consolidado["duracion_min_cdescanso"] = df_consolidado.apply(lambda x: ((x.distancia/max)*60),
                                                            axis=1)
df_consolidado["duracion_max_cdescanso"] = df_consolidado.apply(lambda x: ((x.distancia/min)*60),
                                                            axis=1)

#Suma de horas de descanso si la duracion del viaje es mayor que la establecida para horas de manejo
df_consolidado["duracion_cdescanso"] += df_consolidado.apply(lambda x:  ((x.duracion_cdescanso)/hs_manejo)*hs_descanso
                    if (x.duracion_cdescanso > (hs_manejo*60)) else 0, axis=1)

df_consolidado["duracion_min_cdescanso"] += df_consolidado.apply(lambda x:  ((x.duracion_min_cdescanso)/hs_manejo)*hs_descanso
                    if (x.duracion_cdescanso > (hs_manejo*60)) else 0 ,  axis=1)

df_consolidado["duracion_max_cdescanso"] += df_consolidado.apply(lambda x:  ((x.duracion_max_cdescanso)/hs_manejo)*hs_descanso
                    if (x.duracion_cdescanso > (hs_manejo*60)) else 0, axis=1)

#Agrego una holgura de media hora para los valores que tienen una ditancia menor a la velocidad min 
#esto es porque al dividir por un numero mas grande se hace pequeña la duracion min
#y la duracion maxima tambien es se hace pequeña
df_consolidado["duracion_max_cdescanso"] += df_consolidado.apply(lambda x: 30 if x.distancia <= vel_min else 0 ,  axis=1)
#Separo los outliers encontrados
df_outliers = df_consolidado.loc[(df_consolidado.tiempo_real > df_consolidado.duracion_max_cdescanso ) ]
df_outliers = pd.concat( [df_outliers,df_consolidado.loc[(df_consolidado.tiempo_real) < (df_consolidado.duracion_min_cdescanso)]])    
df_consolidado.drop(index=df_outliers.index, inplace=True)
df_consolidado.dropna(inplace=True)

df_modelo = df_consolidado [["distancia","tiempo_real"]].copy()
#Eliminando outliers mediante el analisis de todos los datos KNN
local_outlier = KNN(contamination=0.03)
local_outlier = local_outlier.fit(df_modelo)
y_pred_outlier = local_outlier.predict(df_modelo)
df_consolidado["out"] = y_pred_outlier
drop = df_consolidado.loc[df_consolidado.out == 1].index
df_consolidado.drop(drop,inplace = True)
df_consolidado.drop("out",axis=1,inplace=True)

# Intervalos de Confianza
# Tools, funciones de utilidad

def ic_to_horas (minutos):
    """ 
    Descripcion
    --------------------------------
    Formatea los minutos 
    calculados para una zona en str
    de HH:mm ry rendondea los
    resultados
    
    Parametros
    --------------------------------
    minutos (float) = cantidad de
    minutos establecidos para una 
    ventana de transit time.
    """
    horas = int(divmod(minutos,60)[0])
    minutos = int(divmod(minutos,60)[1])
    horas_s = f"{horas}"
    minutos_s = f"{minutos}"
    if (minutos <= 14):
        minutos_s = f"00"
    if ( 15 <= minutos < 45):
        minutos_s = f"30"
    if ( 45 <= minutos < 60):
        minutos_s = f"00"
        horas += 1
        horas_s = f"{horas}"      
    if(horas < 10):
        horas_s = f"0{horas}"
    tiempo = f"{horas_s}:{minutos_s}"
    return tiempo


# In[19]:


def min_to_horas (minutos):
    """
    Descripcion
    --------------------------------
    Formatea los minutos 
    calculados en un str tipo
    HH:mm 
    
    Parametros
    --------------------------------
    minutos (float) = cantidad de
    minutos
    """
    horas = int(divmod(minutos,60)[0])
    minutos = int(divmod(minutos,60)[1])
    horas_s = f"{horas}"
    minutos_s = f"{minutos}"
    if(horas < 10):
        horas_s = f"0{horas}"
    if (minutos < 10):
        minutos_s = f"0{minutos}"
    tiempo = f"{horas_s}:{minutos_s}"
    return tiempo


# In[20]:


def hs_descansos (minutos):
    """
    Descripcion
    --------------------------------
    Formatea los minutos 
    calculados en un str tipo
    HH:mm 
    
    Parametros
    --------------------------------
    minutos (float) = cantidad de
    minutos
    """
    horas = int(divmod(minutos,60)[0])
    minutos = int(divmod(minutos,60)[1])
    horas_s = f"{horas}"
    minutos_s = f"{minutos}"
    if (minutos == 0):
        minutos_s = "00"
    if ( 0 < minutos <= 30):
        minutos_s = "30"
    if ( 30 < minutos <= 59):
        minutos_s = "00"
        horas += 1
    if(horas < 10):
        horas_s = f"0{horas}"
        
    tiempo = f"{horas_s}:{minutos_s}"
    return tiempo


# Descansos, los descansos establecidos para los choferes corresponde actualmente a
# una hora de descanso cada 3 hs de viaje
# se calcula si la distancia dividida la velocidad media de 60 km supera las horas de manejo
# si supera las horas de manejo se divide la distancia por la velocidad de manejo permitida (3 hs)
# luego se multiplica por 60 para obtener el resultado en minutos y se divide por el tiempo de descanso en hs
# obteniendo los minutos de descanso para le viaje.
logger.info("Calculo Descansos")
df_distancias ["duracion_cdescanso"] = df_distancias.apply(lambda x: 
                                                           ((x.distancia/vel)*60)/((hs_manejo))*hs_descanso
                                                           if ((x.distancia/vel)*60) > ((hs_manejo)*60)
                                                           else 0,
                                                           axis=1)
#Horas de descanso a Horas
# pasamos los descansos expresados en minutos a horas
df_distancias ["duracion_cdescanso"] = df_distancias.duracion_cdescanso.apply(lambda x: hs_descansos (x) if x > 0 else "00:00" )

#Intervalos de Confianza Estadistico x Subzona
#las velocidades medias y maximas estadisitcas se calculan apartir de la distribucion de probabilidad de la muestra
# se agrupan los tiempos de todas los establecimientos que conforman la zona
# se genera su distribucion de probabilidad
# y se calcula el min y max segun un intervalo de confianza en el cual agrupa los datos alrededor de la media
logger.info("Calculos de intervalos de confianza")
# Calculo de Intervalos de Confianza por Subzona
try:
    subzone =  df_consolidado.subZone.value_counts().index
    for zone in subzone:
        datos = df_consolidado.loc[df_consolidado.subZone == zone,"tiempo_real"]
        min, max = st.norm.interval (alpha , loc = np.mean(datos), scale = datos.std()) 
        df_consolidado.loc[df_consolidado.subZone == zone,"ic_min"] = min
        df_consolidado.loc[df_consolidado.subZone == zone,"ic_max"] = max
        if min < 14:
            min =  df_consolidado.loc[df_consolidado.subZone == zone,"tiempo_real"].quantile(0.2) 
            df_consolidado.loc[df_consolidado.subZone == zone,"ic_min"] = min


    df_consolidado[["ic_min","ic_max"]] = df_consolidado[["ic_min","ic_max"]].round()
    df_consolidado[["ic_min","ic_max"]] = df_consolidado[["ic_min","ic_max"]].astype(int)
    df_consolidado[["ic_min","ic_max"]]
    df_consolidado ["ic_min_hs"] = df_consolidado.ic_min.apply(lambda x:  ic_to_horas(x) )
    df_consolidado ["ic_max_hs"] = df_consolidado.ic_max.apply(lambda x:   ic_to_horas(x) )
    df_consolidado [["subZone","ic_min_hs", "ic_max_hs"]]
    df_subzonas = df_consolidado.pivot_table(values=["ic_min","ic_max"], index="subZone",aggfunc="mean").reset_index()
    df_subzonas ["ic_min_hs"] = df_subzonas.ic_min.apply(lambda x:  ic_to_horas(x))
    df_subzonas ["ic_max_hs"] = df_subzonas.ic_max.apply(lambda x:  ic_to_horas(x))
    confianza = df_consolidado.pivot_table(values="establecimiento",index="subZone" ,aggfunc="count" )
    confianza.columns = confianza.columns.to_flat_index()
    confianza = confianza.reset_index()
    confianza.columns = ["subZone" , "conteo"]
    confianza["Confianza"] = confianza.apply (lambda x: "Si" if x.conteo > 30 else "No" ,axis = 1)
    df_subzonas = df_subzonas.merge(confianza , "left", on="subZone")
    aux = df_distancias.pivot_table(values="distancia",index="zone" )
    aux.columns = aux.columns.to_flat_index()
    aux.columns = ["distancia"]
    aux = aux.reset_index()
    aux.drop("distancia",axis=1,inplace=True)
    df_subzonas = aux.merge(df_subzonas,"left",left_on="zone",right_on="subZone")
    df_subzonas ["subZone"] = df_subzonas.zone
    df_subzonas.Confianza.fillna("Nuevo", inplace=True)
    df_subzonas.fillna("-",inplace=True)
except Exception as ex:
    logger.warning("Error al calcular min max zonas estadistico", exc_info=True)
    quit()
#Intervalos de Confianza Estadistico x Establecimiento
# Calculo de Intervalos de Confianza por Establecimiento
try:
    establecimientos = df_distancias.establecimiento.value_counts().index
    for zone in establecimientos:
        datos = df_consolidado.loc[df_consolidado.establecimiento == zone,"tiempo_real"]
        min, max = st.norm.interval (alpha , loc = np.mean(datos), scale = datos.std()) 
        df_consolidado.loc[df_consolidado.establecimiento == zone,"ic_min_establecimiento"] = min
        df_consolidado.loc[df_consolidado.establecimiento == zone,"ic_max_establecimiento"] = max 
    df_consolidado ["ic_esta_min_hs"] = df_consolidado.ic_min_establecimiento.apply(lambda x:  ic_to_horas(x) )
    df_consolidado ["ic_esta_max_hs"] = df_consolidado.ic_max_establecimiento.apply(lambda x:   ic_to_horas(x) )
    df_establecimiento = df_consolidado.pivot_table(values=["ic_min_establecimiento","ic_max_establecimiento","tiempo_real"], 
                                                    index="establecimiento",
                                                    aggfunc = {
                                                        "ic_min_establecimiento" : "mean",
                                                        "ic_max_establecimiento" : "mean",
                                                        "tiempo_real" : "count"
                                                    } ).reset_index()
    df_establecimiento ["ic_min_hs"] = df_establecimiento.ic_min_establecimiento.apply(lambda x:  min_to_horas(x))
    df_establecimiento ["ic_max_hs"] = df_establecimiento.ic_max_establecimiento.apply(lambda x:  min_to_horas(x))
    df_establecimiento = df_establecimiento [["establecimiento","ic_min_hs","ic_max_hs","tiempo_real"]].copy()
    df_establecimiento ["Confianza"] = df_establecimiento.tiempo_real.apply(lambda x: "Si" if x > 30 else "No")
    df_establecimiento = df_distancias.merge(df_establecimiento,"left",on="establecimiento")
    df_establecimiento = df_establecimiento[["establecimiento","ic_min_hs","ic_max_hs","Confianza"]]
    df_establecimiento.ic_min_hs.fillna("-",inplace=True)
    df_establecimiento.ic_max_hs.fillna("-",inplace=True)
    df_establecimiento.Confianza.fillna("Nuevo",inplace=True)
    df_establecimiento.establecimiento = establecimiento_upper
    df_establecimiento["Descanso"] = df_distancias.duracion_cdescanso
    df_establecimiento.columns = ["Establecimiento","T.T Min", "T.T Max","Confianza","Descanso"]
except Exception as ex:
    logger.warn("Error al calcular los min max por establecimiento estadistico", exc_info=True)
    quit()
#Intervalo de confianza algoritmico Establecimiento
try:
    df_subzonas_2 = df_distancias[["establecimiento","distancia","duracion_cdescanso"]].copy()
    df_subzonas_2["duracion_max"] = ((df_subzonas_2["distancia"]/vel_min)*60)
    df_subzonas_2["duracion_min"] = ((df_subzonas_2["distancia"] / vel_max) * 60)
    df_subzonas_2["duracion_min"] += df_subzonas_2.apply(lambda x:  ((x["duracion_min"])/hs_manejo)*hs_descanso
                        if (x["distancia"] > (hs_manejo*60)) 
                        else x["duracion_min"], axis=1)
    df_subzonas_2["duracion_max"] += df_subzonas_2.apply(lambda x:  ((x["duracion_max"])/hs_manejo)*hs_descanso
                        if (x["distancia"] > (hs_manejo*60))
                        else x["duracion_max"],
                        axis=1)
    df_subzonas_2 ["ic_hs_min"] = df_subzonas_2["duracion_min"].apply(lambda x:  min_to_horas(x) )
    df_subzonas_2 ["ic_hs_max"] = df_subzonas_2["duracion_max"].apply(lambda x:   min_to_horas(x) )
    df_subzonas_2 = df_subzonas_2  [["establecimiento","ic_hs_min", "ic_hs_max","duracion_cdescanso"]]
    df_subzonas_2.columns = ["Establecimiento","T.T Min","T.T Max","Descanso"]
    df_subzonas_2.Establecimiento = establecimiento_upper
    df_establecimiento.loc[df_establecimiento.Confianza == "Nuevo",["T.T Min","T.T Max"]] = df_subzonas_2 [["T.T Min" ,"T.T Max"]]
except Exception as ex:
    logger.warning("Error al calcular min max algoritmico por establecimiento")
    quit()

    # Intevalo de Confianza algoritmico por zona
try:
    subzonas_algoritmico =  df_distancias.pivot_table(values="distancia", 
                            index="zone",
                            aggfunc=["min","max"])
    subzonas_algoritmico.columns = subzonas_algoritmico.columns.to_flat_index()
    subzonas_algoritmico = subzonas_algoritmico.reset_index()
    subzonas_algoritmico.columns = ["zone", "distancia_min","distancia_max"]
    subzonas_algoritmico["duracion_max"] = ((subzonas_algoritmico["distancia_max"]/vel_min)*60)
    subzonas_algoritmico["duracion_min"] = ((subzonas_algoritmico["distancia_min"] / vel_max)*60)
    subzonas_algoritmico["duracion_min"] += subzonas_algoritmico.apply(lambda x:  ((x["duracion_min"])/hs_manejo)*hs_descanso
                                                                    if (x["distancia_min"] > (hs_manejo*60)) 
                                                                    else x["duracion_min"], 
                                                                    axis=1)
    subzonas_algoritmico["duracion_max"] += subzonas_algoritmico.apply(lambda x:  ((x["duracion_max"])/hs_manejo)*hs_descanso
                        if (x["distancia_max"] > (hs_manejo*60))
                        else x["duracion_max"],
                        axis=1)
    subzonas_algoritmico.loc[subzonas_algoritmico.duracion_min < 14 ,"duracion_min"] = 15 
    subzonas_algoritmico ["ic_hs_min"] = subzonas_algoritmico["duracion_min"].apply(lambda x:  ic_to_horas(x) )
    subzonas_algoritmico ["ic_hs_max"] = subzonas_algoritmico["duracion_max"].apply(lambda x:   ic_to_horas(x) )
    subzonas_algoritmico = subzonas_algoritmico[["zone","ic_hs_min","ic_hs_max"]]
    subzonas_algoritmico.columns = ["Zona","T.T Min", "T.T Max"]
except Exception as ex:
    logger.warning("Error al calcular min max algoritmico por zona")
    quit()
# Output
logger.info("Output")
def output_writer (output_name, sheet_names, df, title):
    """
    Descripcion
    -------------------------------------------------------
    Funcion que da un formato de tabla prefinido para excel
    
    Parametros
    ------------------------------------------------------
    output_name = str, Path del archivo
    sheet_names = list, Nombres de las hojas 1 por DataFrame
    df = list, DataFrame para guardar
    title = list, Titulos de cada hoja

    """
    result = 'Exito al Guardar'
    try:
        writer = pd.ExcelWriter(output_name, engine = "xlsxwriter")

        for i in range(len(df)):
            df[i].to_excel(writer, sheet_name=sheet_names[i], startrow=2, header=False, index=False)

            workbook = writer.book
            worksheet = writer.sheets[sheet_names[i]]

            max_row, max_col = df[i].shape

            header_format = workbook.add_format({'font_name' : 'Times New Roman',
                                                 'bold': True, 
                                                 'font_color': 'black',
                                                 'font_size' : 14,
                                                 'align' : 'center',
                                                 'valign' : 'vcenter',
                                                 'border' : 1
                                                })

            cell_format = workbook.add_format({'font_name' : 'Times New Roman',
                                               'font_color' : 'black',
                                               'font_size' : 12,
                                               'align' : 'center',
                                               'valign' : 'vcenter',
                                              })

            title_format = workbook.add_format({'font_name' : 'Times New Roman',
                                          'bold': True, 
                                          'font_color': 'black',
                                          'font_size' : 14,
                                          'align' : 'center',
                                          'valign' : 'vcenter',
                                          'border' : 1,
                                          'bg_color' : "#4F81BD"
                                                })

            columns_settings = [{'header': column, 
                                 'header_format': header_format
                                } for column in df[i].columns]

            worksheet.add_table(1, 0, max_row+1, max_col - 1, {'columns': columns_settings
                                                                })
            worksheet.set_column(first_col = 0, last_col = max_col -1 , width = 15, cell_format = cell_format)

            worksheet.set_row (1,30)

            worksheet.merge_range(0,0,0,max_col-1, title[i],title_format)
            worksheet.set_row (0,30)

        writer.save()
    except Exception as ex:
        result = 'Error al Guardar'+ str(ex.__class__) 
        logger.warning("Error al guardar archivos de salida",exc_info=True)

    print (result)

resumen_segmentacion["Duracion Minima"] = resumen_segmentacion["Duracion Minima"].apply(lambda x: min_to_horas(x))
resumen_segmentacion["Duracion Maxima"] = resumen_segmentacion["Duracion Maxima"].apply(lambda x: min_to_horas(x))

df_subzonas = df_subzonas [["subZone","ic_min_hs","ic_max_hs","Confianza"]].copy()
df_subzonas.columns = ["SubZone","T.T Min", "T.T Max","Confianza"]
#se rellenas las subzonas no calculadas estadisticamente con calculo algoritmico
df_subzonas.loc [df_subzonas["T.T Min"] == "-", ["T.T Min","T.T Max"]]  = subzonas_algoritmico[["T.T Min", "T.T Max"]]

df_distancias = df_distancias [["establecimiento",
                "zone",
                "state",
                "city",
                "latitud",
                "longitud",
                "distancia",
                "duracion",
                "duracion_cdescanso"]]
df_distancias.establecimiento = establecimiento_upper

df_distancias.columns = ["Establecimiento",
                         "SubZona",
                         "Provincia",
                         "Ciudad",
                         "Latitud",
                         "Longitud",
                         "Distancia a Planta",
                         "Duracion Google",
                          "Descanso"]

fecha = str(date.today())
dir_out = "Output"
existe_directorio = False
try:
    for directorio in os.listdir(dir_out):
        if directorio == fecha:
            existe_directorio = True
            dir_out = os.path.join (dir_out,fecha)
            break
    if existe_directorio == False:
        dir_out = os.path.join (dir_out,fecha)
        os.mkdir(dir_out)

    last = 0
    output_name = ""
    if (len (os.listdir(dir_out)) == 0):
        output_name = fecha+"-"+"ESTIMACIONTTMINMAX_0.xlsx"
    else:
        for file in os.listdir(dir_out):
            if file != ('.ipynb_checkpoints'):
                num = int(file.split("_")[1].split(".")[0])
                if num > last:
                    last = num
        output_name = fecha+"-"+"ESTIMACIONTTMINMAX_{:d}".format(last+1)+'.xlsx'
    output_name = os.path.join(dir_out,output_name)

    os.listdir(dir_out)

    dfs = [df_subzonas,
       subzonas_algoritmico,
       df_establecimiento,
       df_subzonas_2,
       df_distancias,
       resumen_segmentacion,
       df_maestro_establecimientos]
    sheet_names = ["T.T Subzonas Est",
                   "T.T Subzonas Const",
                   "T.T Establecimientos Est",
                   "T.T Establecimientos Const",
                   "Establecimientos",
                   "Segmentacion",
                   "Maestro"]
    titles = ["T.T Subzonas Estadistico",
              "T.T Subzonas Cosntante",
              "T.T Establecimientos Estadistico",
              "T.T Establecimientos Constante",
              "Datos Establecimientos",
              "Resumen Segmentacion",
              "Establecimientos Historicos"]
    output_writer(output_name,sheet_names,dfs,titles)

    output_writer(os.path.join("Output","duplicados.xlsx"),["Duplicado"],[df_duplicado],["Establecimientos Duplicados"])

    index_drop = df_maestro_establecimientos.loc[df_maestro_establecimientos["Duplicado"]!=0].index
    df_maestro_establecimientos.drop(index_drop,inplace=True)
    df_maestro_establecimientos.to_excel(name_maestro_establecimientos,index=False)
except Exception as ex:
    logger.warning("Error al guardar los archivos",exc_info=True)
    quit()



#Movemos los archivos a procesados
try:
    if (hay_archivos_establecimientos_nuevo):
        os.rename(archivo_establecimientos_nuevo, archivo_establecimientos_nuevo.replace("PENDIENTE","PROCESADO"))

except Exception as ex:
    logger.warning("Error al mover los archivos a procesados.",exc_info=True)
    quit()





# Ejecutamos el envío por Mail
try:
    os.system("python Mail.py")

except Exception as ex:
    logger.warning("Error al enviar los archivos por mail.",exc_info=True)
    quit()





stop = timeit.default_timer()
logger.info(f"Fin de Ejecucion de ejecucion {stop-start}")