ARG SRC_IMG
FROM $SRC_IMG
COPY ./trained/* /model/saved
COPY ./logs/* /model/logs
