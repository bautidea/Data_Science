import time
import datetime
import os
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

# Código basado en el post:
# http://thepythoncorner.com/dev/how-to-create-a-watchdog-in-python-to-look-for-filesystem-changes/


# Creo las funciones para event handlers. Según el evento, disparo una acción.

# On create, obtengo el momento en el que se recibió la solicitud del usuario, y lo gaurdo en el log. Despues Corro el .bat del modelo.
def on_created(event):
    now = datetime.datetime.now()
    execLog = "On demand Execution detected at " + \
        now.strftime("%Y-%m-%d %H:%M:%S")

    file = open("log.txt", "a")
    file.write(execLog+"\n")
    file.close()

    os.startfile("run_model.bat")
    # os.startfile("ejec.bat")


if __name__ == "__main__":

    # Patrones de nombres de archivos a tener en cuenta, en este caso todos
    patterns = "*"

    # Patrones a ignorar de archivos, en este caso no ignoramos ninguno
    ignore_patterns = ""

    # Ignorar cambios en directorios?
    ignore_directories = False

    # Case Sensitive en nombres?
    case_sensitive = True

    # Creo el event Handler con la configuración previa
    my_event_handler = PatternMatchingEventHandler(
        patterns, ignore_patterns, ignore_directories, case_sensitive)

    # Le asigno a cada evento, las acciones que definimos
    my_event_handler.on_created = on_created
    # my_event_handler.on_modified = on_created

    # directorio a considerar, en este caso, la carpeta donde dejo el archivo desde el servidor de Qlikview
    path = r'./input_bauti/coordenadas_campos'

    # Tener en cuenta los cambios en los subdirectorios?
    go_recursively = False

    # Creo el observer, y le hago un schedule con los gandler de los eventos, el directorio, y
    # la consideración de los subdirectorios o no
    my_observer = Observer()
    my_observer.schedule(my_event_handler, path, recursive=go_recursively)

    # Start to listen
    my_observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        my_observer.stop()
        my_observer.join()
