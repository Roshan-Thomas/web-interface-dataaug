#!/bin/bash

cd ..
cd data

filename_300_twitter_cbow="full_grams_cbow_300_twitter.zip"
mdlFilename_300_twitter_cbow="full_grams_cbow_300_twitter.mdl"
npyTrainablesFilename_300_twitter_cbow="full_grams_cbow_300_twitter.mdl.trainables.syn1neg.npy"
npyVectorsFilename_300_twitter_cbow="full_grams_cbow_300_twitter.mdl.wv.vectors.npy"

filename_300_twitter_sg="full_grams_sg_300_twitter.zip"
mdlFilename_300_twitter_sg="full_grams_sg_300_twitter.mdl"
npyTrainablesFilename_300_twitter_sg="full_grams_sg_300_twitter.mdl.trainables.syn1neg.npy"
npyVectorsFilename_300_twitter_sg="full_grams_sg_300_twitter.mdl.wv.vectors.npy"

filename_300_wiki_cbow="full_grams_cbow_300_wiki.zip"
mdlFilename_300_wiki_cbow="full_grams_cbow_300_wiki.mdl"
npyTrainablesFilename_300_wiki_cbow="full_grams_cbow_300_wiki.mdl.trainables.syn1neg.npy"
npyVectorsFilename_300_wiki_cbow="full_grams_cbow_300_wiki.mdl.wv.vectors.npy"

filename_300_wiki_sg="full_grams_sg_300_wiki.zip"
mdlFilename_300_wiki_cbow="full_grams_sg_300_wiki.mdl"
npyTrainablesFilename_300_wiki_cbow="full_grams_sg_300_wiki.mdl.trainables.syn1neg.npy"
npyVectorsFilename_300_wiki_cbow="full_grams_sg_300_wiki.mdl.wv.vectors.npy"

# Download full_grams_cbow_300_twitter.zip if not present on system
if [[ ! -f $filename_300_twitter_cbow ]]
then
	echo "$filename_300_twitter_cbow does not exist."
	echo; echo 
	echo "Downloading $filename_300_twitter_cbow now ... 🚀"
	echo 
	wget https://bakrianoo.ewr1.vultrobjects.com/aravec/full_grams_cbow_300_twitter.zip
	echo "Download $filename_300_twitter_cbow complete ✅"
	echo
	echo "Unzipping $filename_300_twitter_cbow ... 🚀"
	python3 pyunzip.py full_grams_cbow_300_twitter.zip
	echo "Unzipping $filename_300_twitter_cbow Complete ✅"; echo

# Download full_grams_sg_300_twitter.zip if not present on system
elif [[ ! -f $filename_300_twitter_sg ]]
then
	echo "$filename_300_twitter_sg does not exist."
	echo; echo 
	echo "Downloading $filename_300_twitter_sg now ... 🚀"
	echo 
	wget https://bakrianoo.ewr1.vultrobjects.com/aravec/full_grams_sg_300_twitter.zip
	echo "Download $filename_300_twitter_sg complete ✅"
	echo
	echo "Unzipping $filename_300_twitter_sg ... 🚀"
	python3 pyunzip.py full_grams_sg_300_twitter.zip
	echo "Unzipping $filename_300_twitter_sg Complete ✅"; echo

# Download full_grams_cbow_300_wiki.zip if not present on system
elif [[ ! -f $filename_300_wiki_cbow ]]
then
	echo "$filename_300_wiki_cbow does not exist."
	echo; echo 
	echo "Downloading $filename_300_wiki_cbow now ... 🚀"
	echo 
	wget https://bakrianoo.ewr1.vultrobjects.com/aravec/full_grams_cbow_300_wiki.zip
	echo "Download $filename_300_wiki_cbow complete ✅"
	echo
	echo "Unzipping $filename_300_wiki_cbow ... 🚀"
	python3 pyunzip.py full_grams_cbow_300_wiki.zip
	echo "Unzipping $filename_300_wiki_cbow Complete ✅"; echo

# Download full_grams_sg_300_wiki.zip if not present on system
elif [[ ! -f $filename_300_wiki_sg ]]
then
	echo "$filename_300_wiki_sg does not exist."
	echo; echo 
	echo "Downloading $filename_300_wiki_sg now ... 🚀"
	echo 
	wget https://bakrianoo.ewr1.vultrobjects.com/aravec/full_grams_sg_300_wiki.zip
	echo "Download $filename_300_wiki_sg complete ✅"
	echo
	echo "Unzipping $filename_300_wiki_sg ... 🚀"
	python3 pyunzip.py full_grams_sg_300_wiki.zip
	echo "Unzipping $filename_300_wiki_sg Complete ✅"; echo

# Unzip the full_grams_cbow_300_twitter.zip file
elif [[ -f $filename_300_twitter_cbow ]] && [[ ! -f $mdlFilename_300_twitter_cbow ]] && [[ ! -f $npyTrainablesFilename_300_twitter_cbow ]] && [[ ! -f $npyVectorsFilename_300_twitter_cbow ]]
then
	echo "Unzipping $filename_300_twitter_cbow ... 🚀"
	echo
	python3 pyunzip.py full_grams_cbow_300_twitter.zip
	echo "Unzipping $filename_300_twitter_cbow Complete ✅"; echo

# Unzip the full_grams_sg_300_twitter.zip file
elif [[ -f $filename_300_twitter_sg ]] && [[ ! -f $mdlFilename_300_twitter_sg ]] && [[ ! -f $npyTrainablesFilename_300_twitter_sg ]] && [[ ! -f $npyVectorsFilename_300_twitter_sg ]]
then
	echo "Unzipping $filename_300_twitter_sg ... 🚀"
	echo
	python3 pyunzip.py full_grams_sg_300_twitter.zip
	echo "Unzipping $filename_300_twitter_sg Complete ✅"; echo

# Unzip the full_grams_cbow_300_wiki.zip file
elif [[ -f $filename_300_wiki_cbow ]] && [[ ! -f $mdlFilename_300_wiki_cbow ]] && [[ ! -f $npyTrainablesFilename_300_wiki_cbow ]] && [[ ! -f $npyVectorsFilename_300_wiki_cbow ]]
then
	echo "Unzipping $filename_300_wiki_cbow ... 🚀"
	echo
	python3 pyunzip.py full_grams_cbow_300_wiki.zip
	echo "Unzipping $filename_300_wiki_cbow Complete ✅"; echo

# Unzip the full_grams_sg_300_wiki.zip file
elif [[ -f $filename_300_wiki_sg ]] && [[ ! -f $mdlFilename_300_wiki_sg ]] && [[ ! -f $npyTrainablesFilename_300_wiki_sg ]] && [[ ! -f $npyVectorsFilename_300_wiki_sg ]]
then
	echo "Unzipping $filename_300_wiki_sg ... 🚀"
	echo
	python3 pyunzip.py full_grams_sg_300_wiki.zip
	echo "Unzipping $filename_300_wiki_sg Complete ✅"; echo

else
	echo "$filename_300_twitter_cbow exists and unzipped. All good!! 👍"
	echo "$filename_300_twitter_sg exists and unzipped. All good!! 👍"
	echo "$filename_300_wiki_cbow exists and unzipped. All good!! 👍"
	echo "$filename_300_wiki_sg exists and unzipped. All good!! 👍"
fi

echo "Downloading AraVec models complete! 🍾🍾🍾"; echo

exit 0
