#!/bin/bash

image_name="dataaug-webapp"

if [ $2 ]
then
	image_name=$2
fi

if [ $1 == 'stop' ]
then
	sudo docker stop $(sudo docker ps -q --filter ancestor=$image_name) 
	sudo docker rm $(sudo docker ps --filter status=exited -q)
	echo
	sudo docker system prune -f
	echo; echo
elif [ $1 == 'build' ]
then
	cd ..
	sudo docker build -t $image_name:latest . && sudo docker run -p 8080:8501 -d $image_name
fi

exit 0
