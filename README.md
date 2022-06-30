# Web Interface for Data Augmentation Project (QCRI)

This project aims to understand the various outputs of natural language processing for an Arabic sentence with the pre-existing models available. 

We use the models to augment a sentence and gauge the output using cosine similarity. We used 13 models from [HuggingFace ](https://huggingface.co/) to do the data augmentation. This repo is the code for the web interface hosting all those above methods. The project is based on Python and uses the package `streamlit` to host it on the web. 

Data Augmentation Techniques / Machine Learning Models Used:

* [AraBERT](https://huggingface.co/aubmindlab/bert-base-arabert) 
* [QARiB](https://huggingface.co/qarib/bert-base-qarib) 
* [XLM-RoBERTa](https://huggingface.co/xlm-roberta-base) 
* [AraBART](https://huggingface.co/moussaKam/AraBART)
* [CAMeLBERT-Mix NER](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-mix-ner) 
* [Arabic BERT (Large)](https://huggingface.co/asafaya/bert-large-arabic) 
* [ARBERT](https://huggingface.co/UBC-NLP/ARBERT) 
* [MARBERTv2](https://huggingface.co/UBC-NLP/MARBERTv2) 
* [AraELECTRA](https://huggingface.co/aubmindlab/araelectra-base-generator) 
* [AraGPT2](https://huggingface.co/aubmindlab/aragpt2-base) 
* [W2V (AraVec)](https://github.com/bakrianoo/aravec)
* [Text-to-Text Augmentation](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt)
* [Back Translation](https://huggingface.co/Helsinki-NLP)

![Web Interface Screenshot](/public/web-interface-screenshot.png)

# Setup (Local Machine)
1. Clone the Github repo to your local machine and follow the steps below.

    ```sh
    # Install pipenv (to run environments)
    # If you already have pip installed...
    pip install pipenv

    # If you have Fedora 28
    sudo dnf install pipenv

    # If you have Homebrew (MacOS) [This is discouraged]
    brew install pipenv
    ```

2. Start a new `pipenv` environment 
    ```sh
    pipenv shell
    ```

3. Install the required packages using `pip`
    ```sh
    pip install -r requirements.txt
    ```

4. You will need to download the W2V model from the script that is given in the `/scripts` folder.
    ```sh
    cd scripts
    ./aravec_download.sh
    ```

5. Once all the packages are installed, you should be able to run the app on Streamlit locally
    ```sh
    # main.py is the file where the main code resides
    streamlit run main.py
    ```

# Deploying on a `Docker` Container

1. Build the docker app
    ```sh
    sudo docker build -t dataaug-webapp:latest .
    ```

2. Run the web app using docker with port 8051 and then forwarding it to port 8080 on the server
    ```sh
    sudo docker run -p 8080:8501 dataaug-webapp
    ```

3. Run the web app in the background
    ```sh
    sudo docker run -p 8080:8501 -d dataaug-webapp
    ```

### Build and Run in one command
```sh
sudo docker build -t dataaug-webapp:latest . && sudo docker run -p 8080:8501 -d dataaug-webapp
```

### Deleting unused docker containers
```sh
sudo docker rm $(sudo docker ps --filter status=exited -q)
```

## Scripts

The scripts include two commands now to make the process of building and running docker containers easier.

First, change the directory to `/scripts`
```
cd scripts
```
Then run either of the two commands based on what you want to do:
```sh
# To stop and remove a container (find the name of the container first using 'sudo docker ps' command)
./dockercommands.sh stop <DOCKER_CONTAINER_NAME>

# To build and run a new container 
./dockercommands.sh build <DOCKER_IMAGE_NAME>
```

## Contribution

If you want to make any changes to the repo, follow these steps:
* Fork it to your own Github profile
* Make the changes 
* Create a pull request
* And, I will review the request and accept it.
