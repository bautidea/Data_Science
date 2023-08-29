#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime
import argparse
import shutil
import sys
import os

from docplex.mp.model import Model


# In[2]:


#Seteo para que no muestre los Warnings de valores asignados en un slice de una copia de un DF. (Simplifica lectura de Logs)
pd.options.mode.chained_assignment = None  # default='warn'


# In[3]:


#CAMBIAR CUANDO SE QUIERA CORRER DESDE ACA O CUANDO SE QUIERA DEESCARGAR UNA NUEVA VERSION PARA CORRER AUTOMATICAMENTE

#ap = argparse.ArgumentParser()
#ap.add_argument("-m", "--manual", required=False, help="Indicar si es corrida manual corta (1) o larga (2)", default= 0)
#args = vars(ap.parse_args())

args = {'manual': '1'}


# ## Paths

# In[4]:


# Definimos los paths a los cuales vamos a ir a buscar los excels necesarios
# y donde vamos a depositar el output.

path_excels = 'C:/FTPS/malvina.bernal/Archivos_Variables'


path_fijos = 'C:/FTPS/malvina.bernal/fijos'


carpeta_a_escuchar = 'C:/FTPS/malvina.bernal'


archivos = {'yva': None, 'yq10': None, 'embolse': None,'recepciones': None, 'yvatr': None}


# ## Funciones Útiles

# In[5]:


# Esta función suma dias habiles a una fecha que le pase

def proximo_dia_habil(fecha, dias_a_sumar): 
    proximo_dia_habil = fecha
    while dias_a_sumar != 0:
        proximo_dia_habil += datetime.timedelta(days=1)
        if proximo_dia_habil.weekday() >= 5:
            continue
        dias_a_sumar -= 1       
    return proximo_dia_habil


# Esta función arma el nombre del output dependiendo del día actual y el numero de corrida

def armar_output():
        i = 0
        path_intermedio = 'C:/FTPS/malvina.bernal/Output/' + str(datetime.date.today())
        try:
            os.mkdir(path_intermedio)
                
        except FileExistsError:
                pass
            
        while True:
            path_out = path_intermedio + '/run_' + str(i)
            try:
                os.mkdir(path_out)
                break
            except FileExistsError:
                i += 1
                continue
        return path_out
    
    
# Esta función toma los archivos que se van a usar para el modelo y los copia en la carpeta output


def copiar_archivos(path_out):
        shutil.copy(path_excels + '/' + archivos['yva'], path_out + '/' + archivos['yva'])
        shutil.copy(path_excels + '/' + archivos['yq10'], path_out + '/' + archivos['yq10'])
        shutil.copy(path_fijos + '/' + archivos['embolse'], path_out + '/' + archivos['embolse'])
        shutil.copy(path_excels + '/' + archivos['yvatr'], path_out + '/' + archivos['yvatr'])
        inputs.to_csv(path_out + '/inputs.csv', sep=';', index= False)
        

# Esta función mira los archivos fijos y los identifica para leerlos posteriormente


def identificar_archivos_fijos(path, archivos):
    for file in os.listdir(path):
        try:
            split1 = file.split('_')[1]
            if split1 == 'EMBOLSE':
                    archivos['embolse'] = file
            elif split1 == 'RECEPCIONCENTROS':
                archivos['recepciones'] = file
            else: 
                pass
        except: pass
        
        
# Esta función mira los archivos variables y los identifica para posterior lectura

# CAMBIO ACÁ
def identificar_archivos(path, archivos):
    for file in os.listdir(path):
        split1 = file.split('_')[1]
        if split1 == 'YVADKLT':
            archivos['yva'] = file
        elif split1 == 'YQ10':
            archivos['yq10'] = file
        elif split1 == 'YVATRDKLT':
            archivos['yvatr'] = file
        else: 
            pass
        

# Funcion que se usa en el proceso de verificar que tenemos todos los archivos necesarios


def tengo_archivos(archivos):
    return all(archivos.values())


# ## Inputs

# In[6]:


# BUSCO LOS ARCHIVOS INPUT

identificar_archivos(path_excels, archivos)

identificar_archivos_fijos(path_fijos, archivos)

if not(tengo_archivos(archivos)):
    print('Falta YVA o YQ10 o Embolse')
    sys.exit()

path_out = armar_output()


# In[7]:


dateparse = lambda x: pd.datetime.strptime(x, '%d.%m.%Y')


# In[8]:


# Lectura de la YVA
# CAMBIO ACÁ
data = pd.read_excel(path_excels + '/' + archivos['yva'], sheet_name= 'Sheet1')#, parse_dates=['Fecha_doc', 'Fe_SM_real','FePrefEnt_Cab', 'Fe_entrega','FechaFact'], date_parser=dateparse)

fecha_ultimo_pedido = data['Fe_entrega'].max()


# In[9]:


# Si la corrida es manual se toman los inputs originales

if int(args['manual']) == 1:
    
    inputs = pd.read_csv('C:/FTPS/malvina.bernal/inputs.csv', sep=';', encoding='mbcs')


# Corrida automatica a largo plazo

elif int(args['manual']) == 2:
    
    inputs = pd.read_csv('C:/FTPS/malvina.bernal/inputs.csv', sep=';', encoding='mbcs')
    
    inputs['VALOR'].iloc[0] = datetime.date.today()
    
    inputs['VALOR'].iloc[1] = fecha_ultimo_pedido


# Si es automatica se inputa el dia actual y se toman 90 días hábiles en adelante

else:
    
    inputs = pd.read_csv('C:/FTPS/malvina.bernal/inputs.csv', sep=';', encoding='mbcs')
    
    inputs['VALOR'].iloc[0] = datetime.date.today()
    
    inputs['VALOR'].iloc[1] = proximo_dia_habil(inputs['VALOR'].iloc[0], 90)


# In[10]:


inputs


# In[11]:


# Copio los archivos en la carpeta de outputs
copiar_archivos(path_out) 


# In[12]:


# INPUTS QUE SE VAN A PODER CAMBIAR

fecha_de_hoy = pd.to_datetime(inputs['VALOR'].iloc[0], format='%d/%m/%Y') #Fecha del dia en que se hace la optimizacion

ultimo_dia = pd.to_datetime(inputs['VALOR'].iloc[1], format='%d/%m/%Y') #ultimo dia que se toma como ventana

dias_habiles_disponibles = np.busday_count(fecha_de_hoy.date(), ultimo_dia.date()) # Dias habiles disponibles para despachar o hacer REAP

USAR_FECHA_SIEMBRA_TEMPRANA = True if inputs['VALOR'].iloc[2] == 'SI' else False

USAR_FECHA_SIEMBRA_TARDIA = True if inputs['VALOR'].iloc[3] == 'SI' else False

capacidad_despacho_por_dia = int(inputs['VALOR'].iloc[4]) #Cuantos SSU se pueden despachar por dia (segun camiones)

capacidad_reap_por_dia = int(inputs['VALOR'].iloc[5])

ultima_campaña_despachable = inputs['VALOR'].iloc[6]

calcular_fechas_erroneas = True if inputs['VALOR'].iloc[7] == 'SI' else False

nombre_output = path_out + '/OPTIMIZADODISTRIBUCION_' + fecha_de_hoy.strftime("%Y%m%d") + '_' + ultimo_dia.strftime("%Y%m%d") + '_ROJAS' + '.xlsx'

correr_otd = True if inputs['VALOR'].iloc[8] == 'SI' else False


# In[13]:


# Lectura de todos los archivos necesarios

yq10 = pd.read_excel(path_excels + '/' + archivos['yq10'], parse_dates=['Fecha DE'], date_parser=dateparse)

embolse = pd.read_excel(path_fijos + '/'  + archivos['embolse'], skiprows=4)

prop = pd.read_csv('C:/FTPS/malvina.bernal/fijos/proporcion_de_material_por_hibrido')

yvatr = pd.read_excel(path_excels + '/'  + archivos['yvatr'])#, sheet_name='YVA TR GRAL')

prioridades = pd.read_excel('C:/FTPS/malvina.bernal/fijos/Prioridades Distribución FEFO.xlsx')

zonas = pd.read_excel('C:/FTPS/malvina.bernal/fijos/Detalle Distrib - Final.xlsx', skiprows=3)

recep = pd.read_excel(path_fijos + '/'  + archivos['recepciones'])

# CAMBIO ACÁ

# maestro = pd.read_excel('C:/FTPS/malvina.bernal/fijos/Grupo de clientes.xlsx', sheet_name= 'Base clientes')


# ## Embolse

# In[14]:


embolse.columns


# In[15]:


embolse.drop(columns= ['Unnamed: 0'], inplace= True)

embolse = embolse.dropna(subset=['DESTINOS'])

#zonas.drop(columns= ['Unnamed: 0'], inplace= True)


# In[16]:


# Hay embolses que se dividen en los destinos por lo que hay que separarlos

embolse = embolse[embolse['DESTINOS'] != 6035] #Filtramos el 6035 porque nos dijeron que no iba más

separadas = embolse[(embolse['DESTINOS'] != 6009) & (embolse['DESTINOS'] != 6244) & (embolse['DESTINOS'] != 6592)]


# In[17]:


# Para los embolses que se mandan a ambos centros en distintas proporciones, armo una linea para cada centro

if len(separadas) > 0:
    separadas['DESTINOS'] = separadas['DESTINOS'].astype(str)
    for ind, row in separadas.iterrows():

            copia = row

            div1, centro1, div2, centro2 = row['DESTINOS'].split(' ')

            if centro1 == '6009':

                centro2 = '6244'

            else: 

                centro1 = '6244'

            embolse.iloc[ind, embolse.columns.get_loc('DESTINOS')] = centro1

            embolse.iloc[ind, embolse.columns.get_loc('SSU estimados')] *= (float(div1[0:2]) / 100)

            copia['DESTINOS'] = centro2

            copia['SSU estimados'] *= (float(div2[0:2]) / 100)

            embolse = embolse.append(copia)


    embolse['DESTINOS'] = embolse['DESTINOS'].replace({'6244': 6244, '6009': 6009})


#Me quedo con las Filas del embolse que estan entre el primer dia y el ultimo a optimizar

embolse['Fecha disponibilidad'] = pd.to_datetime(embolse['Fecha disponibilidad'])

embolse = embolse[(embolse['Fecha disponibilidad'] >= fecha_de_hoy) 
                  & (embolse['Fecha disponibilidad'] <= ultimo_dia)]

embolse = embolse[['Híbridos', 'Unidad', 'SSU estimados', 'Fecha disponibilidad', 'DESTINOS']]

embolse = embolse.groupby(['Híbridos','Fecha disponibilidad']).sum().reset_index()


# In[18]:


def estimar_SKU(hy, ssu_total, prop, disponibilidad, destino): # prop es el dataframe que esta metido en el csv "proporcion_de_material_por_hibrido"
    df = prop[['HY','Material','PROPORCION_DEL_HY']]
    df1 = df[df['HY'] == hy]
    df1['SSU_A_OBTENER'] = (ssu_total* df1['PROPORCION_DEL_HY']).round()
    df1['Fecha disponibilidad'] = disponibilidad
    df1['DESTINOS'] = destino
    return df1


# In[19]:


def estimar_embolse(embolse, prop):
    df = pd.DataFrame()
    for x in range(len(embolse)):
        data = estimar_SKU(embolse.iloc[x,0], embolse.iloc[x,2], prop, embolse.iloc[x,1],embolse.iloc[x,3])
        df = df.append(data, ignore_index= True) 
    return df


# In[20]:


# ACA PODRÍA PRIMERO SEPARAR POR FECHA Y DESPUES HACER EL IF

if len(embolse) != 0:
    
    embolse['Fecha disponibilidad'] = pd.to_datetime(embolse['Fecha disponibilidad']).dt.date    
    
    embolse_estimado = estimar_embolse(embolse, prop)
    
    mat_embolse = embolse_estimado['Material']
    
    ssu_embolse = embolse_estimado['SSU_A_OBTENER']
    
    destinos = embolse_estimado['DESTINOS']
    
    fechas_embolse = pd.to_datetime(embolse_estimado['Fecha disponibilidad'])
    
    hibridos_embolse = embolse_estimado['HY']
        
    yq10_embolse = pd.DataFrame({'HY': hibridos_embolse, 'MATERIAL': mat_embolse,'CENTRO': destinos, 'SSU' : ssu_embolse, 'FECHA_DISPONIBILIDAD': fechas_embolse, 'PRIORIDAD_FEFO': 17 })

else:
    
    yq10_embolse = pd.DataFrame(columns=['HY','MATERIAL','CENTRO', 'SSU', 'FECHA_DISPONIBILIDAD', 'PRIORIDAD_FEFO'])


# ## YQ10

# In[21]:


yq10.columns


# In[22]:


#Elimino las lineas que no tienen habilitacion DAV 1 ni DAV 2

yq10.dropna(subset=['DAV Number 1 Argentina', 'DAV Number 2 Argentina'], how='all', inplace=True)


yq10 = yq10[((yq10['Ce.'] == 6009) | (yq10['Ce.'] == 6244) | (yq10['Ce.'] == 6592)) 
            & (yq10['Alm.'] == 'WM01')]

yq10 = yq10[(yq10['Cód.'] == 'LIB1') | (yq10['Cód.'] == 'LIB2') | (yq10['Cód.'] == 'LIB3') 
            | ((yq10['Cód.'] == 'DETA') & ((pd.to_datetime(yq10['Fecha DE']).dt.day > 15)) & (pd.to_datetime(yq10['Fecha DE']).dt.month >= 10) )]


yq10 = yq10[(yq10['Ubicación'].str[0:2] != 'RO') & (yq10['Ubicación'].str[0:3] != 'MAN') & (yq10['Ubicación'] != 'STORECV') 
           & (yq10['Ubicación'] != 'VIRTUAL') & (yq10['Ubicación'] != 'MFDENTRG') & (yq10['Ubicación'] != 0)]

yq10 = yq10[yq10['UMB'] == 'BOL']


# In[23]:


lotes_habilitados = list(np.sort(yq10['Lote'].str[0].unique()))

#SI AL LLEGAR A LA Z VUELVE A EMPEZAR POR LA A, ESTE CODIGO NO FUNCIONA BIEN

yq10 = yq10[(yq10['Lote'].str[0] > ultima_campaña_despachable)]


# In[24]:


yq10.reset_index(drop= True, inplace=True)
yq10['HY'] = ''
for idx, row in yq10.iterrows():
    yq10['HY'].iloc[idx] = row['Denominación'].split(' ')[2]


# In[25]:


def pasar_a_ssu(yq10_):
    
    equivSSU = 80
    
    df1 = yq10_.copy(deep=True)
    df1['TamanioBolsa'] = 1
    
    for idx, row in df1.iterrows():
        df1['TamanioBolsa'].iloc[idx] = row['Denominación'].split(' ')[4][:2]
        
    df1 = df1[(df1['TamanioBolsa'] == '80') |( df1['TamanioBolsa'] == '60') | (df1['TamanioBolsa'] == '40')]
    df1['TamanioBolsa'] = df1['TamanioBolsa'].astype(int)
    
    df1['SSU'] = df1['LibrUtiliz'] * (df1['TamanioBolsa'] / equivSSU)
    
    return df1
   


# In[26]:


yq10.reset_index(drop=True, inplace=True)
yq10 = pasar_a_ssu(yq10)


# In[27]:


yq10_final = pd.DataFrame({'HY': yq10['HY'], 'MATERIAL': yq10['Material'],'CENTRO': yq10['Ce.'], 'SSU' : yq10['SSU'], 
                           'FECHA_DISPONIBILIDAD': fecha_de_hoy, 'DE': yq10['Cód.'], 'AÑO_MAX_CAMP': yq10['Fecha DE'].dt.year, 'Campaña': yq10['Lote'].str[0],
                           'Lote': yq10['Lote'], 'Ubicación': yq10['Ubicación'] , 'Peso_bolsa': yq10['SG Peso neto promedio p. bolsa'] })


yq10_final = yq10_final.merge(prioridades, how='left', on= ['DE', 'Campaña', 'AÑO_MAX_CAMP'])

#Hago esto porque las prioridades no tienen en cuenta la campaña U

yq10_final['Prioridad'].fillna(16, inplace= True)


# Agrego distintivo para materiales provenientes del embolse

yq10_final['Campaña'] = yq10_final['Campaña'].fillna('Embolse')

yq10_final.rename(columns={'Prioridad': 'PRIORIDAD_FEFO'}, inplace= True)

yq10_final = yq10_final.drop(columns=['AÑO_MAX_CAMP'])


if len(yq10_embolse) != 0:
    yq10_final = yq10_final.append(yq10_embolse, ignore_index= True)


yq10_final['PRIORIDAD_FEFO'].replace([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17], [17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1], inplace= True)

# Normalizo la prioridad FEFO a valores entre 0 y 1

yq10_final['PRIORIDAD_FEFO'] = (yq10_final['PRIORIDAD_FEFO'] - 1) / 16

yq10_final['FECHA_DISPONIBILIDAD'] = pd.to_datetime(yq10_final['FECHA_DISPONIBILIDAD'])

yq10_final['DE'].fillna(value= 'Emb', inplace= True)

yq10_final.head()


# In[28]:


# ME FIJO QUE NO SE DUPLIQUE EL STOCK
len(yq10) + len(yq10_embolse) == len(yq10_final)


# In[29]:


mat_unique = list(yq10_final['MATERIAL'].unique())

cantidades = []

for m in mat_unique:
    
    cantidades.append(yq10_final['SSU'][(yq10_final['MATERIAL'] == m)].sum())


# In[30]:


yq10_nueva = pd.DataFrame({'Material': mat_unique, 'Cantidad_Total': cantidades})
yq10_nueva


# ## Prioridades

# In[31]:


def asignar_fechas_entrega(ytr):
    asignados = ytr[(ytr['Lote'].isna()) & (ytr['Fe_SM_real'].isna())]
    maximo = capacidad_despacho_por_dia / 2
    dia = 1
    suma = 0
    if(len(asignados) > 0):
        for indice in range(len(asignados)):
            asignados['Doc_venta'].iloc[indice] = str(asignados['Doc_venta'].iloc[indice]) + '-' + str(indice)
            if (suma + asignados['Ctd_entr'].iloc[indice]) < maximo:
                suma += asignados['Ctd_entr'].iloc[indice]
                asignados['Fe_entrega'].iloc[indice] = proximo_dia_habil(fecha_de_hoy, dia)
            else:
                suma = asignados['Ctd_entr'].iloc[indice]
                dia += 1
                asignados['Fe_entrega'].iloc[indice] = proximo_dia_habil(fecha_de_hoy, dia)

        asignados.rename(columns={'Ctd_fact': 'Ctd_fact', 'Ctd_conf': 'Ctd_conf.',
                                  'Ctd_entr': 'Ctd_entr', 'Fe_SM_real': 'Fe_SM_real'}, inplace=True)
        asignados['Status'] = 'Liberado'
        asignados['Prioridad'] = 500000
        asignados['ERROR_FECHA_ENTREGA'] = 'NO'
        asignados['Recepcion'] = 'Recibe'
        asignados = asignados[['Doc_venta', 'ClVt', 'Solic', 'Nomb_Dest_Mcia', 'GClt', 'Dest_Mcia', 'Oficina_de_Venta',
           'Material', 'SSU', 'Fe_entrega', 'Recepcion','ERROR_FECHA_ENTREGA', 'Status', 'FechaFact', 'FePrefEnt_Cab', 'BqEn', 'Prioridad']]
        asignados.rename(columns={'Nomb_Dest_Mcia': 'Cliente'}, inplace=True)
    return asignados


# In[32]:


# Funciones para "calcular" prioridades
anio_actual = datetime.date.today().year

anio_anterior = anio_actual - 1

prio1 = 100000

prio2 = 10000

prioridad_pedido = 1000

def obtener_prioridad(potencia):
    return prioridad_pedido / (2 ** potencia)


# REGLA DE LAS FILAS 3 Y 4 (Tipo Pedido)
def prioridad_Clvt(df):
    regla1 = (df['ClVt'] == 'ZMKG') | (df['ClVt'] == 'ZBKG')
    regla2 = (df['ClVt'] == 'ZROG') | (df['ClVt'] == 'ZCHG')
    regla3 = (df['ClVt'] == 'ZSCG') | (df['ClVt'] == 'ZRQG') | (df['ClVt'] == 'ZVQG') 
    df['Prioridad_Clvt'] = 0
    df['Prioridad_Clvt'][regla1] = prio1
    df['Prioridad_Clvt'][regla2] = prio2
    df['Prioridad_Clvt'][regla3] = obtener_prioridad(1)
    return df


# REGLA DE LA FILA 5 (Fecha Factura)
def prioridad_ff(df):
    df['Prioridad_ff'] = 0
    df['Prioridad_ff'][df['FechaFact'].dt.year == anio_anterior] = obtener_prioridad(2)
    df['Prioridad_ff'][(df['FechaFact'].dt.year == anio_actual) & (df['FechaFact'].dt.month <= 4)] = obtener_prioridad(3)
    df['Prioridad_ff'][(df['FechaFact'].dt.year == anio_actual) & (df['FechaFact'].dt.month > 4)] = obtener_prioridad(4)
    return df


#REGLA FILA 6 (Tipo Cliente)

def prioridad_tc(df):
    df['Prioridad_tc'] = 0
    df['Prioridad_tc'][(df['GClt'] == 'EK') & (df['GClt'] == 'ES')] = obtener_prioridad(5)
    df['Prioridad_tc'][(df['GClt'] == 'EL') & (df['GClt'] == 'E5')] = obtener_prioridad(6) 
    df['Prioridad_tc'][(df['GClt'] == 'EF')] = obtener_prioridad(7)
    df['Prioridad_tc'][(df['GClt'] == 'E2')] = obtener_prioridad(8)
    df['Prioridad_tc'][(df['GClt'] == 'EC')] = obtener_prioridad(9)
    df['Prioridad_tc'][(df['GClt'] == 'E7')] = obtener_prioridad(10)
    df['Prioridad_tc'][(df['GClt'] == 'EY')] = obtener_prioridad(11)
    return df


# REGLA DE LA FILA 7 (Fecha Deseada)
def prioridad_fd(df):
    df['Prioridad_fd'] = 0
    df['Prioridad_fd'][df['FePrefEnt_Cab'].dt.year == anio_anterior] = obtener_prioridad(12)
    df['Prioridad_fd'][(df['FePrefEnt_Cab'].dt.year == anio_actual) & (df['FePrefEnt_Cab'].dt.month < 4)] = obtener_prioridad(13)
    df['Prioridad_fd'][(df['FePrefEnt_Cab'].dt.year == anio_actual) & (df['FePrefEnt_Cab'].dt.month < 7)] = obtener_prioridad(14)
    df['Prioridad_fd'][(df['FePrefEnt_Cab'].dt.year == anio_actual) & (df['FePrefEnt_Cab'].dt.month < 10)] = obtener_prioridad(15)
    df['Prioridad_fd'][(df['FePrefEnt_Cab'].dt.year == anio_actual) & (df['FePrefEnt_Cab'].dt.month > 10)] = obtener_prioridad(16)
    return df


def prioridad_yva(df):
    df = prioridad_Clvt(df)
    df= prioridad_ff(df)
    df= prioridad_tc(df)
    df = prioridad_fd(df)
    df['Prioridad'] = df['Prioridad_Clvt'] + df['Prioridad_fd'] + df['Prioridad_ff'] + df['Prioridad_tc']
    df.drop(columns=['Prioridad_Clvt', 'Prioridad_fd', 'Prioridad_ff', 'Prioridad_tc'], inplace= True)
    
    return df


# ## YVA (DATA)

# In[33]:


# Obtengo las fechas de Inicio y Fin de Siembra temprana y tardía

feZonas = zonas[['Sales Office description', 'Inicio ', 'Fin', 'Inicio .1', 'Fin.1']]

feZonas.rename(columns= {'Sales Office description': 'Oficina_de_Venta', 'Inicio ':'Inicio Siembra Temprana',
                         'Fin':'Fin Siembra Temprana', 'Inicio .1':'Inicio Siembra Tardia',
                         'Fin.1':'Fin Siembra Tardia' },inplace=True)


# In[34]:


# Obtengo la fecha a partir de la cuál cada centro puede recibir mercadería

recep2 = recep[['ID', 'Recepción de bolsas']]

recep2.rename(columns={'ID': 'Dest_Mcia', 'Recepción de bolsas': 'Fecha Recepcion'}, inplace= True)


# In[35]:


# Agrego a la YVA las fechas de siembra y de recepción

data.rename(columns= {'Fe_entrega': 'Fe_entrega'},inplace=True)

data = data.merge(feZonas, how='left', on= 'Oficina_de_Venta')

data = data.merge(recep2, how='left', on= 'Dest_Mcia')


# In[36]:


# Completamos las fechas de recepción vacias y cambiamos las que no reciben

recibe_siempre = pd.to_datetime(str(fecha_de_hoy.year) + '-01-01')

no_recibe = pd.to_datetime(str(fecha_de_hoy.year) + '-12-31')

# Si estan en nulo reciben siempre entoces lleno con 1 de enero

data['Fecha Recepcion'].fillna(value= recibe_siempre, inplace=True)

data['Recepcion'] = data['Fecha Recepcion']

# Los que por alguna razon no reciben, tienen la razón en la columna, por lo que lo cambio
# a fin de año.

data['Fecha Recepcion'] = pd.to_datetime(data['Fecha Recepcion'], errors='coerce')

reciben = data['Fecha Recepcion'][(data['Fecha Recepcion'].notna())].index.to_list()

data.Recepcion.iloc[reciben] = 'Recibe'

data['Fecha Recepcion'].fillna(no_recibe, inplace= True)


# In[37]:


# Marco los pedidos que tienen error en la fecha de entrega = Fecha de entrega < Fecha de Recepción

data['ERROR_FECHA_ENTREGA'] = 'NO'

data['ERROR_FECHA_ENTREGA'][(pd.to_datetime(data['Fe_entrega']) > pd.to_datetime(data['Inicio Siembra Temprana'])) | 
                            (pd.to_datetime(data['Fe_entrega']) < pd.to_datetime(data['Fecha Recepcion']))] = 'SI'

# Separo aquellos pedidos flagueados con error en fecha de entrega porque en el output los voy a mostrar separados para que sean analizados

error_fecha = data[data['ERROR_FECHA_ENTREGA'] == 'SI']

error_fecha = error_fecha[['Doc_venta', 'Dest_Mcia', 'Material', 'SSU', 'Fe_entrega', 'Inicio Siembra Temprana', 'Fin Siembra Temprana',
                           'Inicio Siembra Tardia', 'Fin Siembra Tardia', 'Recepcion', 'Fecha Recepcion', 'Status','BqEn', 'ERROR_FECHA_ENTREGA']]

if not(calcular_fechas_erroneas):
    
    data = data[data['ERROR_FECHA_ENTREGA'] != 'SI']
    
else: 
    
    data['Fe_entrega'][pd.to_datetime(data['Fe_entrega']) > pd.to_datetime(data['Inicio Siembra Temprana'])] = pd.to_datetime(data['Inicio Siembra Temprana']) - pd.Timedelta(value= 7, unit='D')


# In[38]:


if USAR_FECHA_SIEMBRA_TEMPRANA:
    
    yvaDK = data[['Doc_venta', 'ClVt', 'Solic', 'Nomb_Dest_Mcia', 'GClt', 'Dest_Mcia', 'Material', 'SSU', 'Oficina_de_Venta', 'Fe_entrega', 'Inicio Siembra Temprana', 'Fin Siembra Temprana', 'Inicio Siembra Tardia', 'Fin Siembra Tardia', 'Recepcion', 'Fecha Recepcion', 'FechaFact', 'FePrefEnt_Cab', 'Status','BqEn', 'ERROR_FECHA_ENTREGA']]
    
    yvaDK['Fe_entrega'] = yvaDK['Inicio Siembra Temprana']
    
    #ultimo_dia = yvaDK['Fin Siembra Temprana'].max() -> Comento esta linea porque genera que entren todos los pedidos

elif USAR_FECHA_SIEMBRA_TARDIA:
    
    yvaDK = data[['Doc_venta', 'ClVt', 'Solic', 'Nomb_Dest_Mcia', 'GClt', 'Dest_Mcia', 'Material', 'SSU', 'Oficina_de_Venta', 'Fe_entrega', 'Inicio Siembra Temprana', 'Fin Siembra Temprana', 'Inicio Siembra Tardia', 'Fin Siembra Tardia', 'Recepcion', 'Fecha Recepcion', 'FechaFact', 'FePrefEnt_Cab', 'Status','BqEn', 'ERROR_FECHA_ENTREGA']]
    
    yvaDK['Fe_entrega'] = yvaDK['Inicio Siembra Tardia']
    
    #ultimo_dia = yvaDK['Fin Siembra Tardia'].max()

else:
    
    yvaDK = data[['Doc_venta', 'ClVt', 'Solic', 'Nomb_Dest_Mcia', 'GClt', 'Dest_Mcia', 'Material', 'SSU', 'Oficina_de_Venta', 'Fe_entrega', 'Inicio Siembra Temprana', 'Fin Siembra Temprana', 'Inicio Siembra Tardia', 'Fin Siembra Tardia', 'Recepcion', 'Fecha Recepcion', 'Status', 'FechaFact', 'FePrefEnt_Cab','BqEn', 'ERROR_FECHA_ENTREGA']]

yvaDK.rename(columns={'Nomb_Dest_Mcia': 'Cliente'}, inplace= True)


# In[39]:


yvaDK['Fe_entrega'] = pd.to_datetime(yvaDK['Fe_entrega'])

yvaDK['FePrefEnt_Cab'] = pd.to_datetime(yvaDK['FePrefEnt_Cab'])

yvaDK['FechaFact'] = pd.to_datetime(yvaDK['FechaFact'])
yvaDK['Status'].unique()


# In[40]:


# Filtro del YVA los pedidos que se vencen en el tiempo que se eligio

yvaDK = yvaDK[(yvaDK['Fe_entrega'] <= ultimo_dia) & (yvaDK['Fe_entrega'] >= fecha_de_hoy)]
yvaDK


# In[41]:


# Se agrega a la YVA el GClt

# CAMBIO ACÁ
#maestro = maestro[['Solic', 'GClt']]

#yvaDK = yvaDK.merge(maestro, on='Solic', how='left')

# Asigno prioridad a los pedidos

yvaDK = prioridad_yva(yvaDK)

despachos = asignar_fechas_entrega(yvatr)

#despachos = despachos.merge(maestro, on='Solic', how='left')

despachos = despachos.merge(feZonas, how='left', on= 'Oficina_de_Venta')

despachos = despachos.merge(recep2, how='left', on= 'Dest_Mcia')
yvaDK['Status'].unique()


# In[42]:


#Los que tienen fecha de recepción
despachos['Fecha Recepcion'] = pd.to_datetime(data['Fecha Recepcion'], errors='coerce')

despachos['Fecha Recepcion'].fillna(value= fecha_de_hoy, inplace=True)


# In[43]:


#Chequeo que haya envios para que no pinche y concateno

if(len(despachos) > 0):
    despachos = despachos[['Doc_venta', 'ClVt', 'Solic', 'Cliente', 'GClt', 'Dest_Mcia',
       'Material', 'SSU', 'Oficina_de_Venta', 'Fe_entrega', 'Inicio Siembra Temprana',
       'Fin Siembra Temprana', 'Inicio Siembra Tardia', 'Fin Siembra Tardia', 'Recepcion',
       'Fecha Recepcion', 'Status', 'FechaFact', 'FePrefEnt_Cab', 'BqEn',
       'ERROR_FECHA_ENTREGA', 'Prioridad']]
    # Se agregan los despachos a la YVA
    yvaDK = yvaDK.append(despachos, ignore_index=True)


# In[44]:


# Dejo las prioridades entre 1 y 10
#yvaDK['Prioridad'] = 1 + ( (( yvaDK['Prioridad'] - yvaDK['Prioridad'].min()) * (10 - 1)) / (yvaDK['Prioridad'].max() - yvaDK['Prioridad'].min()))


# In[45]:


# Si hay un pedido pendiente con fecha pasada, se le reasigna la fecha de entrega al último día de los que tomé

yvaDK['Fe_entrega'][yvaDK['Fe_entrega'] <= datetime.datetime.today()] = proximo_dia_habil(fecha_de_hoy, 7)#pd.to_datetime('20/06/2021')

# Hago un nuevo DF con el material en stock del material solicitado en el pedido

yvaDK = yvaDK.merge(yq10_nueva,how= 'left', on='Material' )


# In[46]:


# Separo los pedidos liberados de los bloqueados ya que se van a correr de forma separada

liberados = yvaDK[yvaDK['Status'] == 'Liberado']

bloqueados = yvaDK[yvaDK['Status'] != 'Liberado']
# Cambio los NaN por ceros
liberados.fillna(value= {'Cantidad_Total': 0} , inplace= True)

bloqueados.fillna(value= {'Cantidad_Total': 0} , inplace= True)
bloqueados.head()


# #### ----------------------------------------------------COSAS PARA TESTEAR - COMIENZO------------------------------------------------------------------------------

# #### --------------------------------------------COSAS PARA TESTEAR - FIN-------------------------------------------------------------------------------------------------

# In[47]:


# Reseteo los indices para que queden ordenados para usarlos sin problema

liberados.reset_index(inplace= True,drop= True)

yq10_final = yq10_final[yq10_final['SSU'] >= 0]

yq10_final.reset_index(inplace= True, drop= True)


# ## CPLEX

# In[48]:


# Listas que voy a utilizar para armar las variables que necesita el modelo

indices_yq10 = yq10_final.index.to_list()

yq10_final['Indice_Modelo'] = indices_yq10

pedidos_unicos = liberados['Doc_venta'].unique()


# In[49]:


modelo = Model(name="pedidos")


# In[50]:


# Una variable binaria por cada pedido que indica si será despachado o no

binarios = modelo.binary_var_dict(pedidos_unicos)

# Una variable discreta por cada material (producto) que indica cuanto se va a despachar

#cant_material_a_despachar = modelo.integer_var_dict(indices_yq10)
cant_material_a_despachar = modelo.continuous_var_dict(indices_yq10)


# In[51]:


yq10_final['SSU'] = (yq10_final['SSU']).astype(int)


# ## CONSTRAINTS

# In[52]:


materiales_pedidos = liberados['Material'].unique()


# In[53]:


promedio_despacho = liberados['SSU'].mean()


# In[54]:


dias_df = pd.DataFrame({'Fecha' : pd.date_range(fecha_de_hoy, ultimo_dia, freq='D')})

def constraint_3(model,yq10_,yva_):
    
    yvaCons = yva_.copy(deep=True)
    yq10Cons = yq10_.copy(deep=True)
    pedKeys = list(binarios.keys()).copy()
    pedVals = list(binarios.values()).copy()
    stockKeys = list(cant_material_a_despachar.keys()).copy()
    stockVals = list(cant_material_a_despachar.values()).copy()
    
    materiales_pedidos = yvaCons['Material'].unique()
    dfBin = pd.DataFrame({'Doc_venta': pedKeys, 'Despacho': pedVals})
    stockCons = pd.DataFrame({'Indice_Modelo': stockKeys, 'Despacho': stockVals})

    dfResult = pd.DataFrame(columns=['Fecha', 'cumSumStock', 'cumSumDesp', 'Resta', 'RestaMin'])
    
    for m in materiales_pedidos:
        
        pedidos_por_material = yvaCons[['Fe_entrega', 'Doc_venta', 'SSU', 'Material']][yvaCons['Material'] == m]
        pedidos_por_material = pedidos_por_material.merge(dfBin, how='left', on='Doc_venta', copy=False)
        pedidos_por_material['Fe_entrega'] = pd.to_datetime(pedidos_por_material['Fe_entrega'])
        
        pedidos_por_material['cantPorDesp'] = pedidos_por_material['SSU'] * pedidos_por_material['Despacho']
        
        pedidos_mat_dia = dias_df.merge(pedidos_por_material, how='left', left_on='Fecha', right_on ='Fe_entrega').fillna(0)
        
        dfResult_mat = pedidos_mat_dia[['Fecha', 'cantPorDesp']]
        dfResult_mat = dfResult_mat.groupby(by='Fecha').sum().reset_index()
        dfResult_mat['cumSumDesp'] = np.cumsum(dfResult_mat['cantPorDesp'])
        
        
        stock_por_material = yq10Cons[['FECHA_DISPONIBILIDAD', 'Indice_Modelo']][(yq10Cons['MATERIAL'] == m) & (yq10Cons['CENTRO'] == 6009)]
        stock_por_material = stock_por_material.merge(stockCons, how='left', on='Indice_Modelo', copy=False)
        
        stock_mat_dia = dias_df.merge(stock_por_material, how='left', left_on='Fecha', right_on ='FECHA_DISPONIBILIDAD').fillna(0)
        
        dfResult_stock = stock_mat_dia[['Fecha', 'Despacho']]
        dfResult_stock = dfResult_stock.groupby(by='Fecha').sum().reset_index()
        dfResult_stock['cumSumStock'] = np.cumsum(dfResult_stock['Despacho'])
        
        dfResult_p = dfResult_stock[['Fecha', 'cumSumStock']].merge(dfResult_mat[['Fecha', 'cumSumDesp']], how='left', on='Fecha', copy=False)
        dfResult_p['Resta'] = np.nan
        #dfResult_p['RestaMax'] = np.nan
        
        for idx, row in dfResult_p.iterrows():
            dfResult_p['Resta'].iloc[idx] = dfResult_p['cumSumDesp'].iloc[idx] - dfResult_p['cumSumStock'].iloc[idx]
            #dfResult_p['RestaMax'].iloc[idx] = model.max(dfResult_p['Resta'].iloc[idx],0)
        
        dfResult = dfResult.append(dfResult_p, ignore_index=True)
        
    dfResult_final = dfResult.groupby(by='Fecha').sum().reset_index()
    
    for idx, row in dfResult_final.iterrows():
        difDias = np.busday_count(fecha_de_hoy.date(), pd.to_datetime(dfResult_final['Fecha'].iloc[idx]).date())
        reapPosible = capacidad_reap_por_dia * difDias
        #model.add_constraint(dfResult_p['Resta'].iloc[idx] <= reapPosible)
        model.add_constraint(dfResult_final['Resta'].iloc[idx] <= reapPosible)


# In[55]:


# Armo constraint para que la cantidad a despachar de un material no supere la cantidad disponible de ese material
def constraint_1(model,yq10_):
    for ind in list(cant_material_a_despachar.keys()):
        model.add_constraint(cant_material_a_despachar[ind] <= yq10_['SSU'].iloc[ind])
        
        
# Armo una constraint para que la suma de la cantidad a despachar por cada material sea igual a la cantidad que hay que despachar
def constraint_2(model,yq10_,yva_):
    for mat in yq10_['MATERIAL'].unique():
        # Armo un DF filtrado por un material (me trae las distintas calidades de ese material)
        materiales = yq10_[yq10_['MATERIAL'] == mat]
        # Armo un DF de los pedidos filtrado por un material
        yva = yva_[yva_['Material'] == mat]
        pedidos = yva['Doc_venta'].unique()
        # Armo una lista con los indices asi despues los busco en mis variables enteras
        lista_indices = materiales.index
        
        model.add_constraint(model.sum(cant_material_a_despachar[ind] for ind in lista_indices) 
                              == model.sum(binarios[ped] * yva['SSU'][(yva['Doc_venta'] == ped) & (yva['Material'] == mat)].sum()
                                           for ped in pedidos))
        
        
# Constraint para asegurar que no se pueda superar la capacidad de REAP por día
# def constraint_3(model,yq10_,yva_):
#     dias_con_pedido = (yva_['Fe_entrega'].unique()).astype(str)
#     dias_con_pedido = pd.to_datetime(dias_con_pedido)
#     for fecha in dias_con_pedido:
#         diferencia_dias = np.busday_count(fecha_de_hoy.date(), fecha.date())
#         yva = yva_[yva_['Fe_entrega'] == fecha]
#         yq10 = yq10_[yq10_['FECHA_DISPONIBILIDAD'] <= fecha]
#         nro_pedidos = yva['Doc_venta'].unique() #Son los numeros de pedidos de cada día
#         mat_yva = yva['Material'].unique()

#         model.add_constraint(model.sum((binarios[ped] * yva['SSU'][(yva['Doc_venta'] == ped) & (yva['Material'] == mat)].sum()) 
#                    - (yq10['SSU'][(yq10['CENTRO'] == 6009) & (yq10['MATERIAL'] == mat)].sum())
#                    for mat in mat_yva for ped in nro_pedidos) <= (capacidad_reap_por_dia * (diferencia_dias + 1)))   
        

             
# Por cada material, chequea que la suma de las cantidades por pedido de ese material sea menor o igual 
# al stock
def constraint_4(model,yq10_,yva_):
    materiales_pedidos = yva_['Material'].unique()
    for m in materiales_pedidos:
        yva2 = yva_[yva_['Material'] == m]
        dias_distintos = (yva2['Fe_entrega'].unique()).astype(str)
        dias_distintos = pd.to_datetime(dias_distintos)
        for dia in dias_distintos:
            yva_fecha = yva2[yva2['Fe_entrega'] <= dia]
            unicos = yva_fecha['Doc_venta'].unique()
            yq10_fecha = yq10_[(yq10_['FECHA_DISPONIBILIDAD'] <= dia) & (yq10_['MATERIAL'] == m)]
            indices_disp = yq10_fecha['Indice_Modelo'].unique()
            model.add_constraint(model.sum(binarios[p] * (yva_fecha['SSU'][(yva_fecha['Doc_venta'] == p)].sum()) 
                                             for p in unicos) <= (model.sum(cant_material_a_despachar[ind] for ind in indices_disp)))
# Con la ultima linea me aseguro que la suma de lo que se va a sacar, sea menor al stock que hay 


# El problema de esta constraint es que: por ahi se descartan pedidos por no poder satisfacerlos un dia pero se podria despachar de a poco en dias anteriores
# Se asegura que por día, no se pueda despachar mas que el máximo
def constraint_5(model,yva_):
    dias_con_peido = yva_['Fe_entrega'].unique()
    for fecha in dias_con_peido:
        #diferencia_dias = np.busday_count(fecha_de_hoy.date(), fecha.date())
        diferencia_dias = np.busday_count(fecha_de_hoy.date(), pd.to_datetime(fecha).date())
        yva = yva_[yva_['Fe_entrega'] == fecha]
        numeros_pedidos = yva['Doc_venta'].unique()
        model.add_constraint((model.sum(binarios[i] * yva['SSU'][(yva['Doc_venta'] == i)].sum() 
                                          for i in numeros_pedidos)) <= capacidad_despacho_por_dia * diferencia_dias)
        
        
# La suma de SSU de todos los materiales no puede superar la maxima cantidad por dia * cantidad de dias que estoy tomando
#def constraint_6(model,yva_):
#    model.add_constraint((model.sum(binarios[i] * yva_['SSU'][(yva_['Doc_venta'] == i)].sum() 
#                                      for i in binarios)) <= (dias_habiles_disponibles * capacidad_despacho_por_dia))


# In[56]:


dias_distintos = (liberados['Fe_entrega'].unique()).astype(str)
dias_distintos = pd.to_datetime(dias_distintos)


# In[57]:


#%%time
constraint_1(modelo,yq10_final)


# In[58]:


#%%time
constraint_2(modelo,yq10_final,liberados)


# In[59]:


modelo.print_information()


# In[60]:


#%%time
constraint_3(modelo,yq10_final,liberados)


# In[61]:


modelo.print_information()


# In[62]:


#%%time
constraint_4(modelo,yq10_final,liberados)


# In[63]:


#%%time
constraint_5(modelo,liberados)


# ## FUNCION OBJETIVO

# In[64]:


liberados[['Prioridad', 'Doc_venta']].head()


# In[65]:


# Si se decide correr por OTD (On Time Delivery), el modelo va a intentar  despachar la mayor cantidad de pedidos posibles
# teniendo en cuenta que debe sacar los más prioritarios también y utilizando los materiales mas prioritarios

if correr_otd:
    
    modelo.maximize(( 2 * promedio_despacho * modelo.sum(binarios)) 
                    + (modelo.sum(cant_material_a_despachar[k] * yq10_final['PRIORIDAD_FEFO'].iloc[k] for k in cant_material_a_despachar)))
else:
    
    # Si se decide sacar pedidos por prioridad, el modelo intentará sacar los pedidos maás prioritarios con los materiales
    # más prioritarios sin importar si puede sacar más pedidos utilizando otra combinación.
    
    prio_yva = modelo.sum(binarios[i] * liberados['Prioridad'][liberados['Doc_venta'] == i].unique()[0] for i in binarios)
    
    prio_fefo = modelo.sum(cant_material_a_despachar[k] * yq10_final['PRIORIDAD_FEFO'].iloc[k] for k in cant_material_a_despachar)
    
    modelo.maximize(prio_yva + prio_fefo)


# In[66]:


modelo.print_information()


# In[67]:


solucion = modelo.solve()


# In[68]:


print(modelo.get_solve_details())


# In[69]:


soluciones = solucion.get_all_values()


# In[70]:


despachos = soluciones[:len(binarios)]


# In[71]:


despachos_pedidos = pd.DataFrame({'Doc_venta': pedidos_unicos, 'SE_DESPACHA': despachos})


# In[72]:


despachos_pedidos = despachos_pedidos.replace([1.0,0.0], ['SI', 'NO'])


# In[73]:


despachos_pedidos


# In[74]:


liberados = liberados.merge(despachos_pedidos, how= 'left', on= 'Doc_venta')


# In[75]:


despachos_materiales = soluciones[len(binarios):]


# ## Asignaciones

# In[76]:


yq10_final['Campaña'].fillna('Embolse', inplace= True)


# In[77]:


yq10_final['CANT_A_DESPACHAR'] = despachos_materiales


# In[78]:


def clasificar_no_despachados(desp):
    desp['RAZON'] = '-'
    desp.loc[desp['SE_DESPACHA'] == 'NO', 'RAZON'] = 0
    desp.loc[(desp['RAZON'] == 0) & (desp['SSU'] > desp['Cantidad_Total']), 'RAZON'] = 'Sin stock del material'
    for ped in desp['Doc_venta'].unique():
        if (desp['RAZON'][desp['Doc_venta'] == ped] == 'Sin stock del material').any():
            desp.loc[(desp['RAZON'] == 0) & (desp['Doc_venta'] == ped), 'RAZON'] = 'Falta stock de un material en este pedido'
    for mat in desp['Material'][desp['RAZON'] == 0].unique():
        for index , row in desp[(desp['Material'] == mat) & (desp['RAZON'] == 0)].iterrows():
            if ((desp['SSU'][(desp['Material'] == mat) & (desp['SE_DESPACHA'] == 'SI')].sum() + row['SSU']) 
                > row['Cantidad_Total']):
                desp['RAZON'].iloc[index] = 'OTD'
    for ped in desp['Doc_venta'][desp['RAZON'] == 0]:
        if any(desp['RAZON'][desp['Doc_venta'] == ped] == 'OTD'):
            desp['RAZON'][desp['Doc_venta'] == ped] = 'OTD'
        else:
            desp['RAZON'][desp['Doc_venta'] == ped] = 'Stock llega mas tarde o falta de capacidad logística'
    return desp
    


# In[79]:


yq10_final.reset_index(drop=True, inplace=True)
yq10_final['ID'] = yq10_final.index.values
#yq10_final['ID'] = yq10_final['MATERIAL'].astype(str) + yq10_final['DE'] + yq10_final['Lote'] + yq10_final['Ubicación'].astype(str) + yq10_final['CENTRO'].astype(str)


# In[80]:


def asignar_materiales_a_pedidos(liberados_, yq10_):
    resultado = pd.DataFrame(columns=['Doc_venta', 'Cliente', 'Oficina_de_Venta', 'Fe_entrega', 'Fecha Recepcion',
                                     'Dest_Mcia', 'Prioridad', 'Status', 'Material', 'ID_Material_Asignado', 'HY',
                                     'SSU', 'FECHA_DISPONIBILIDAD', 'CENTRO', 'Lote', 'DE', 'Ubicación'])
    liberados_si = liberados_[liberados_['SE_DESPACHA'] == 'SI']
    liberados_si.sort_values(['Fe_entrega','Prioridad'], ascending=[True, False], inplace= True)
    liberados_si.reset_index(drop=True, inplace=True)
    yq10_si = yq10_[yq10_['CANT_A_DESPACHAR'] > 0]
    yq10_si.sort_values(['FECHA_DISPONIBILIDAD', 'PRIORIDAD_FEFO', 'Peso_bolsa', 'Campaña'], ascending=[True, False, False, True], inplace= True)
    for indice in range(len(liberados_si)):
        material = liberados_si['Material'].iloc[indice]
        fecha = liberados_si['Fe_entrega'].iloc[indice]
        mat_posibles = yq10_si[(yq10_si['MATERIAL'] == material) & (yq10_si['FECHA_DISPONIBILIDAD'] <= fecha)]
        count = 0
        suma = 0
        while ((suma < liberados_si['SSU'].iloc[indice]) & (count < 10)):
            mat_posibles = mat_posibles[mat_posibles['CANT_A_DESPACHAR'] > 0]
            if ((mat_posibles['CANT_A_DESPACHAR'].iloc[0] + suma) >= liberados_si['SSU'].iloc[indice]):
                resultado = resultado.append({'Doc_venta': liberados_si['Doc_venta'].iloc[indice],
                                              'Cliente': liberados_si['Cliente'].iloc[indice],
                                              'Oficina_de_Venta': liberados_si['Oficina_de_Venta'].iloc[indice],
                                              'Fe_entrega': liberados_si['Fe_entrega'].iloc[indice],
                                              'Fecha Recepcion': liberados_si['Fecha Recepcion'].iloc[indice],
                                              'Dest_Mcia': liberados_si['Dest_Mcia'].iloc[indice],
                                              'Prioridad': liberados_si['Prioridad'].iloc[indice],
                                              'Status': liberados_si['Status'].iloc[indice],
                                              'Oficina_de_Venta': liberados_si['Oficina_de_Venta'].iloc[indice],
                                              'Material': liberados_si['Material'].iloc[indice],
                                              'ID_Material_Asignado': mat_posibles['ID'].iloc[0],
                                              'SSU': liberados_si['SSU'].iloc[indice] - suma,
                                              'HY': mat_posibles['HY'].iloc[0],
                                              'FECHA_DISPONIBILIDAD': mat_posibles['FECHA_DISPONIBILIDAD'].iloc[0],
                                              'CENTRO': mat_posibles['CENTRO'].iloc[0],
                                              'Lote': mat_posibles['Lote'].iloc[0],
                                              'DE': mat_posibles['DE'].iloc[0],
                                              'Ubicación': mat_posibles['Ubicación'].iloc[0]}, ignore_index= True)
                suma = liberados_si['SSU'].iloc[indice]
                yq10_si['CANT_A_DESPACHAR'][(yq10_si['ID'] == mat_posibles['ID'].iloc[0])] -= (liberados_si['SSU'].iloc[indice] - suma)
                mat_posibles['CANT_A_DESPACHAR'].iloc[0] -= (liberados_si['SSU'].iloc[indice] - suma)
            elif((mat_posibles['CANT_A_DESPACHAR'].iloc[0] + suma) < liberados_si['SSU'].iloc[indice]):
                resultado = resultado.append({'Doc_venta': liberados_si['Doc_venta'].iloc[indice],
                                  'Fe_entrega': liberados_si['Fe_entrega'].iloc[indice],
                                  'Fecha Recepcion': liberados_si['Fecha Recepcion'].iloc[indice],
                                  'Dest_Mcia': liberados_si['Dest_Mcia'].iloc[indice],
                                  'Prioridad': liberados_si['Prioridad'].iloc[indice],
                                  'Status': liberados_si['Status'].iloc[indice],
                                  'Material': liberados_si['Material'].iloc[indice],
                                  'ID_Material_Asignado': mat_posibles['ID'].iloc[0],
                                  'SSU': mat_posibles['CANT_A_DESPACHAR'].iloc[0],
                                  'HY': mat_posibles['HY'].iloc[0],
                                  'FECHA_DISPONIBILIDAD': mat_posibles['FECHA_DISPONIBILIDAD'].iloc[0],
                                  'CENTRO': mat_posibles['CENTRO'].iloc[0],
                                  'Lote': mat_posibles['Lote'].iloc[0],
                                  'DE': mat_posibles['DE'].iloc[0],
                                  'Ubicación': mat_posibles['Ubicación'].iloc[0]}, ignore_index= True)
                yq10_si['CANT_A_DESPACHAR'][(yq10_si['ID'] == mat_posibles['ID'].iloc[0])] -= 0
                suma += mat_posibles['CANT_A_DESPACHAR'].iloc[0]
                mat_posibles['CANT_A_DESPACHAR'].iloc[0] = 0
                count += 1
    return resultado


# In[81]:


liberados['Doc_venta'] = liberados['Doc_venta'].astype(str)
for indice in range(len(liberados)):
    liberados['Doc_venta'].iloc[indice] = liberados['Doc_venta'].iloc[indice].split('-')[0]


# In[82]:


liberados = liberados.reset_index(drop= True)
liberados = clasificar_no_despachados(liberados)
liberados = liberados.drop(columns=['Cantidad_Total'])


# In[83]:


asignacion_liberados = asignar_materiales_a_pedidos(liberados, yq10_final)


# In[84]:


liberados['Fe_entrega'] = pd.to_datetime(liberados['Fe_entrega']).dt.date
liberados['FechaFact'] = pd.to_datetime(liberados['FechaFact']).dt.date
liberados['FePrefEnt_Cab'] = pd.to_datetime(liberados['FePrefEnt_Cab']).dt.date
liberados['Fecha Recepcion'] = pd.to_datetime(liberados['Fecha Recepcion']).dt.date


# In[ ]:





# In[85]:


nombre_output


# In[ ]:





# In[86]:


yq10_liberados = yq10_final.copy(deep=True)
yq10_liberados.sort_values('CANT_A_DESPACHAR', ascending= False, inplace= True)
#liberados.sort_values('SE_DESPACHA', ascending= False, inplace= True)
writer = pd.ExcelWriter(nombre_output, date_format='mm/dd/yyyy',engine='xlsxwriter', datetime_format='mm/dd/yyyy',)
inputs.to_excel(writer, sheet_name='INPUTS', index=False)
liberados.to_excel(writer, sheet_name='yva_liberados', index=False)
yq10_liberados.to_excel(writer, sheet_name='yq10_liberados', index=False)
asignacion_liberados.to_excel(writer, sheet_name='pedido_materiales', index=False)
error_fecha.to_excel(writer, sheet_name='error_fecha', index=False)


# In[87]:


yq10_para_bloqueados = yq10_final.copy(deep=True)
yq10_para_bloqueados.reset_index(inplace= True, drop= True)
yq10_para_bloqueados['SSU'] = yq10_para_bloqueados['SSU'] - yq10_para_bloqueados['CANT_A_DESPACHAR']
yq10_para_bloqueados['CANT_A_DESPACHAR'] = 0


# In[88]:


indices_yq10_bloq = yq10_para_bloqueados.index.to_list()
yq10_para_bloqueados['Indice_Modelo'] = indices_yq10_bloq


# In[89]:


bloqueados.reset_index(inplace=True, drop=True)
bloqueados = bloqueados.drop(columns= 'Cantidad_Total')


# In[90]:


mat_unique_bloq = list(yq10_para_bloqueados['MATERIAL'].unique())
cantidades_bloq = []
for m in mat_unique_bloq:
    cantidades_bloq.append(yq10_para_bloqueados['SSU'][(yq10_para_bloqueados['MATERIAL'] == m)].sum())


# In[91]:


yq10_nueva_bloq = pd.DataFrame({'Material': mat_unique_bloq, 'Cantidad_Total': cantidades_bloq})
yq10_nueva_bloq


# In[92]:


bloqueados = bloqueados.merge(yq10_nueva_bloq,how= 'left', on='Material' )
bloqueados['Cantidad_Total'].fillna(0, inplace= True)


# In[93]:


modelo_2 = Model(name="bloqueados")


# In[94]:


pedidos_bloqueados = bloqueados['Doc_venta'].unique()


# In[95]:


binarios = modelo_2.binary_var_dict(pedidos_bloqueados)


# In[96]:


#cant_material_a_despachar = modelo_2.integer_var_dict(indices_yq10_bloq)
cant_material_a_despachar = modelo_2.continuous_var_dict(indices_yq10_bloq)


# In[97]:


materiales_pedidos = bloqueados['Material'].unique()


# In[98]:


promedio_despacho = bloqueados['SSU'].mean()


# In[99]:


#%%time
constraint_1(modelo_2,yq10_para_bloqueados)


# In[100]:


#%%time
constraint_2(modelo_2,yq10_para_bloqueados,bloqueados)


# In[101]:


#%%time
constraint_3(modelo_2,yq10_para_bloqueados,bloqueados)


# In[102]:


#%%time
constraint_4(modelo_2,yq10_para_bloqueados,bloqueados)


# In[103]:


#%%time
constraint_5(modelo_2,bloqueados)


# In[104]:


if correr_otd:
    modelo_2.maximize((2 * promedio_despacho * modelo_2.sum(binarios)) 
                + (modelo_2.sum(cant_material_a_despachar[k] * yq10_para_bloqueados['PRIORIDAD_FEFO'].iloc[k] for k in cant_material_a_despachar)))
else:
    prio_yva_bloq = modelo_2.sum(binarios[i] * bloqueados['Prioridad'][bloqueados['Doc_venta'] == i].unique()[0] for i in binarios)
    prio_fefo_bloq = modelo_2.sum(cant_material_a_despachar[k] * yq10_para_bloqueados['PRIORIDAD_FEFO'].iloc[k] for k in cant_material_a_despachar)
    modelo_2.maximize(prio_yva_bloq + prio_fefo_bloq)


# In[105]:


modelo_2.print_information()


# In[106]:


solucion_bloq = modelo_2.solve()


# In[107]:


soluciones_bloqueados = solucion_bloq.get_all_values()


# In[108]:


print(modelo_2.get_solve_details())


# In[109]:


despachos_bloqueados = soluciones_bloqueados[:len(binarios)]


# In[110]:


despachos_pedidos_bloqueados = pd.DataFrame({'Doc_venta': pedidos_bloqueados, 'SE_DESPACHA': despachos_bloqueados})


# In[111]:


despachos_pedidos_bloqueados = despachos_pedidos_bloqueados.replace([1.0,0.0], ['SI', 'NO'])


# In[112]:


bloqueados = bloqueados.merge(despachos_pedidos_bloqueados, how= 'left', on= 'Doc_venta')


# In[113]:


despachos_materiales_bloqueados = soluciones_bloqueados[len(binarios):]


# In[114]:


yq10_para_bloqueados['CANT_A_DESPACHAR'] = despachos_materiales_bloqueados


# ## OUTPUT

# In[115]:


yq10_para_bloqueados['FECHA_DISPONIBILIDAD'] = pd.to_datetime(yq10_final['FECHA_DISPONIBILIDAD']).dt.date
yq10_para_bloqueados.sort_values('CANT_A_DESPACHAR', ascending= False, inplace= True)
bloqueados['Fe_entrega'] = pd.to_datetime(bloqueados['Fe_entrega']).dt.date
bloqueados.sort_values('SE_DESPACHA', ascending= False, inplace= True)


# In[116]:


bloqueados['FechaFact'] = pd.to_datetime(bloqueados['FechaFact']).dt.date
bloqueados['FePrefEnt_Cab'] = pd.to_datetime(bloqueados['FePrefEnt_Cab']).dt.date
bloqueados['Fecha Recepcion'] = pd.to_datetime(bloqueados['Fecha Recepcion']).dt.date


# In[117]:


yq10_para_bloqueados.sort_values('CANT_A_DESPACHAR', ascending= False, inplace= True)
yq10_para_bloqueados['ID'] = yq10_para_bloqueados['MATERIAL'].astype(str) + yq10_para_bloqueados['DE'] + yq10_para_bloqueados['Lote'] + yq10_para_bloqueados['Ubicación'].astype(str) + yq10_para_bloqueados['CENTRO'].astype(str)


# In[118]:


bloqueados.sort_values('SE_DESPACHA', ascending= False, inplace= True)


# In[119]:


bloqueados = bloqueados.reset_index(drop= True)
bloqueados = clasificar_no_despachados(bloqueados)


# In[120]:


bloqueados['SSU'][bloqueados['SE_DESPACHA'] == 'SI'].sum()


# In[121]:


yq10_para_bloqueados['CANT_A_DESPACHAR'].sum()


# In[122]:


len(yq10_para_bloqueados['ID'].unique())


# In[123]:


len(yq10_para_bloqueados)


# In[124]:


asignacion_bloqueados = asignar_materiales_a_pedidos(bloqueados, yq10_para_bloqueados)


# In[125]:


bloqueados.to_excel(writer, sheet_name='yva_bloqueados', index=False)
yq10_para_bloqueados.to_excel(writer, sheet_name='yq10_bloqueados', index=False)
asignacion_bloqueados.to_excel(writer, sheet_name='pedido_materiales_bloq', index=False)
#df_kpi.to_excel(writer, sheet_name='KPIs', index=False)


# In[126]:


print('Modelo Finalizado')


# In[127]:


writer.save()
writer.close()


# In[128]:


bloqueados_si = bloqueados[bloqueados['SE_DESPACHA'] == 'SI']
bloqueados_si.sort_values('Prioridad', ascending= False, inplace= True)
yq10_si = yq10_para_bloqueados[yq10_para_bloqueados['CANT_A_DESPACHAR'] > 0]
yq10_si.sort_values(['FECHA_DISPONIBILIDAD', 'PRIORIDAD_FEFO', 'Peso_bolsa', 'Campaña'], ascending=[True, False, False, True], inplace= True)        


# In[ ]:





# In[ ]:




