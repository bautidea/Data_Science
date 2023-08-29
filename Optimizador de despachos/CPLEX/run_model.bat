ECHO "COMIENZA EJECUCION DEL MODELO"
call C:\ProgramData\Anaconda3\Scripts\activate.bat
cd C:\Users\Administrator\Desktop\Listener
python corroborar_archivos.py
cd C:\Users\Administrator\Desktop\CPLEX

set LOGFILE= model_log.txt
call :LOG > %LOGFILE%
exit /B

:LOG
REM python BAYER_Modelo_CPLEX_Unificado.py -m 0
REM ECHO Finalizo El Primer Modelo
REM python BAYER_Modelo_CPLEX_Unificado.py -m 2
REM ECHO Finalizo El Segundo Modelo
python BAYER_Modelo_CPLEX_Unificado.py -m 1
ECHO Finalizo La ejecución del modelo con los inputs enviados
cd C:\Users\Administrator\Desktop\Listener
rem python limpiar_carpeta.py
rem ECHO Los Archivos Variables se han limpiado
python Mail.py



