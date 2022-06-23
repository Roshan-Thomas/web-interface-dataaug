#!/bin/bash

if [ $1 == 'stop' ]
then
	sudo docker stop $2 && sudo docker rm $2
elif [ $1 == 'build' ]
then
	cd ..
	sudo docker build -t $2:latest . && sudo docker run -p 8080:8501 -d $2
fi

exit 0
