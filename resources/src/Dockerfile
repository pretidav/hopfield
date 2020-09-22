FROM python:3.8
RUN mkdir /hopfield
COPY /code ./hopfield/
WORKDIR ./hopfield/
RUN ls
run chmod 777 launcher.sh
run pip install -r requirements.txt
ENTRYPOINT ./launcher.sh
