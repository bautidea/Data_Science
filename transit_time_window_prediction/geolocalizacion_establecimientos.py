import requests 
import json
import pandas as pd
import os
import os.path as path
import googlemaps
from datetime import datetime
import timeit
import logging
import base64
from dotenv import load_dotenv




os.chdir("C:\\Python Scripts\\Bayer_distribucion_estimador_transit_times")


def configure ():
    load_dotenv()
env = configure()
start = timeit.default_timer()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(fmt ='%(asctime)s - %(levelname)s - %(message)s', datefmt='%d-%m-%Y %H:%M')
handler = logging.FileHandler("logging.log")
handler.setLevel(logging.DEBUG)
handler.setFormatter(formatter)
logger.addHandler(handler)


logger.info("----Inicio Geolocalizacion de establecimientos----")



ruta_carpeta_establecimientos = "C:\\QlikView\\PROD\\DATA\\EXCEL\\XLXS\\PENDIENTE\\INBOUND\\ROJAS\\ESTABLECIMIENTOS"

hay_archivos_establecimientos_nuevo = (len(os.listdir(ruta_carpeta_establecimientos)) > 0)

archivo_establecimientos_nuevo = ""

if not hay_archivos_establecimientos_nuevo:
    logger.info("No hay nuevo archivo de establecimientos. Utilizando los establecimientos existentes")
    stop = timeit.default_timer()
    logger.info(f"Fin de ejecucion{stop-start}")
    exit(0)

else:
    archivo_establecimientos_nuevo = ruta_carpeta_establecimientos+"\\"+os.listdir(ruta_carpeta_establecimientos)[0]





client = os.getenv("CLIENT")
decode = base64.b64decode(client)
gmaps = googlemaps.Client(key=decode.decode("utf-8"))
bayer_lat = -34.2031478
bayer_lon = -60.6939654
#carga de datos
try:
    df_cordenadas = pd.read_excel(archivo_establecimientos_nuevo)
except Exception as ex:
    logger.warning("Error en la carga de datos",exc_info=True)

def buscar_destino (lat,lon):
    try:
        result = gmaps.distance_matrix( origins = {"lat":bayer_lat, "lng":bayer_lon},
                                    destinations ={"lat":lat, "lng":lon},
                                    language = "es-ar",

                                   )
   
        destino = result["destination_addresses"][0]
        distancia = result["rows"][0]["elements"][0]["distance"]["text"]
        duracion = result["rows"][0]["elements"][0]["duration"]["text"]
        distancia_value = result["rows"][0]["elements"][0]["distance"]["value"]
        duracion_value = result["rows"][0]["elements"][0]["duration"]["value"]
    except:
        destino,distancia,distancia_value,duracion,duracion_value = 0,0,0,0,0
        logger.warning(f"Error no se encuentra localizacion lat: {lat}, lon: {lon}")

    return destino,distancia,distancia_value,duracion,duracion_value
#Inicio Geolocalizacion de Erstablecimientos
df_cordenadas["datos"] = df_cordenadas.apply(lambda row: buscar_destino(row["Lat"],row["Long"])
                                             ,axis=1
                                            )
#Carga de datos en dataFrame original
logger.info("Carga de datos")
df_cordenadas["destino"] = df_cordenadas.apply(lambda row: row["datos"][0], axis = 1)
df_cordenadas["distancia"] = df_cordenadas.apply(lambda row: row["datos"][1],axis = 1)
df_cordenadas["distancia_value_mts"] = df_cordenadas.apply(lambda row: row["datos"][2],axis = 1)
df_cordenadas["duracion"] = df_cordenadas.apply(lambda row: row["datos"][3],axis = 1)
df_cordenadas["duracion_value_seg"] = df_cordenadas.apply(lambda row: row["datos"][4],axis = 1)
df_cordenadas["ciudad"] = df_cordenadas.destino.str.split(",",expand=True)[0]
df_cordenadas["provincia"] = df_cordenadas.destino.str.split(",",expand=True)[1]
df_cordenadas.drop(["datos","destino"],axis=1,inplace=True)
df_cordenadas.columns = ["zone", "establecimiento","latitud","longitud","distancia","distancia_value_mts","duracion","duracion_min","city","state"]
try:
    df_cordenadas.duracion_min = df_cordenadas.duracion_min / 60
    df_cordenadas.distancia = df_cordenadas.distancia_value_mts/1000
    df_cordenadas.distancia = df_cordenadas.distancia.round().astype(int)
except Exception as ex:
    logger.warning("Error en calculo de duracion o distancia", exc_info=True)

try:
    df_cordenadas.to_excel("Input/establecimientos.xlsx")
except Exception as ex:
    logger.warning("Error al guardar datos", exc_info=True)
stop = timeit.default_timer()
logger.info(f"Fin de ejecucion{stop-start}")
