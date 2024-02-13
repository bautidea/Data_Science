from email.mime.multipart import MIMEMultipart
from email.utils import COMMASPACE, formatdate
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
import datetime
import smtplib
#from smtplib import SMTP_SSL as smSsl
import os
from dotenv import load_dotenv
import base64

def configure ():
    load_dotenv()

env = configure()

def encontrar_archivos():
    archivos = []
    path1= 'Output/' + str(datetime.date.today())
    last = 0
    dirlast = ''
    for run in os.listdir(path1):
        print ("run: " + run)

        try:
            split0 = run.split('_')[1].split(".")[0]
            if int(split0) >= last:
                last = int(split0)
                dirlast = run
        except:
            pass

    last_run = path1 + '/' + dirlast
    
    archivos.append(last_run)
    return archivos
            

gmail_user = base64.b64decode(os.getenv("USER")).decode("utf-8")
gmail_password = base64.b64decode(os.getenv("PASS")).decode("utf-8")

sent_from = gmail_user
to = ['someone@mail.com']
subject = 'Estimacion Transit Time Minimos y Maximos'

archivos_a_enviar = encontrar_archivos()

if(len(archivos_a_enviar) == 0):
	body = 'Hubo un error en la ejecución del modelo, asegurese que todos los archivos necesarios estén cargados.'
else:
	body = 'La ejecución del modelo fue satisfactoria. Enviado Automaticamente desde Python.'

msg = MIMEMultipart()
msg['From'] = sent_from
msg['To'] = COMMASPACE.join(to)
msg['Date'] = formatdate(localtime=True)
msg['Subject'] = subject


for ar in archivos_a_enviar:
    part = MIMEBase('application', "octet-stream")
    part.set_payload(open(ar , "rb").read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', 'attachment; filename="{}"'.format(Path(ar).name))
    msg.attach(part)


try:
    server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    server.ehlo()
    server.login(gmail_user, gmail_password)
    server.sendmail(gmail_user, to, msg.as_string())
    print('Mail Enviado')
    
except Exception as e:
    print( 'Something went wrong...')
    print(e)
