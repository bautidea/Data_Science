In order to execute docker image you need to run the following commands.
- docker load -i course_approve_detection_dock.tar

After the image is loaded we can now run it, by executing.
- docker run -v /path_to_host/output:/app/output course_approve_detection_dock

Note: Where it says 'path_to_host' you need to specify the directory in which you are
running the container, for example in my case it was 
- docker run -v "C:\Users\juanb\OneDrive\Documentos\0 - De Angelis Juan Bautista\001 - Github\course_approve_detection\docker\output:/app/output" course_approve_detection_dock