#!/bin/bash

filename="full_grams_cbow_100_twitter.zip"
mdlFilename="full_grams_cbow_100_twitter.mdl"
npyTrainablesFilename="full_grams_cbow_100_twitter.mdl.trainables.syn1neg.npy"
npyVectorsFilename="full_grams_cbow_100_twitter.mdl.wv.vectors.npy"

if [[ ! -f $filename ]]
then
	echo "$filename does not exist."
	echo " "
	echo " "
	echo "Downloading $filename now..."
	echo " "
	wget https://bakrianoo.ewr1.vultrobjects.com/aravec/full_grams_cbow_100_twitter.zip
	echo "Download $filename complete"
	echo " "
	echo " "
	python3 pyunzip.py full_grams_cbow_100_twitter.zip
	echo "Unzipping Complete"

elif [[ -f $filename ]] && [[ ! -f $mdlFilename ]] && [[ ! -f $npyTrainablesFilename ]] && [[ ! -f $npyVectorsFilename ]]
then
	echo "Unzipping $filename..."
	echo " "
	python3 pyunzip.py full_grams_cbow_100_twitter.zip

else
	echo "$filename exists. All good!! 👍"
fi

echo "Downloading AraVec model complete"
exit 0
