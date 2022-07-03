#!/bin/bash

image_name="dataaug-webapp"

if [ $2 ]
then
	image_name=$2
fi

if [ $1 == 'stop' ]
then
	sudo docker stop $2 && sudo docker rm $2 && sudo docker system prune -f
	echo; echo
elif [ $1 == 'build' ]
then
	cd ..
	sudo docker build -t $image_name:latest . && sudo docker run -p 8080:8501 -d $image_name
fi

exit 0
