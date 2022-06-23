# Web Interface for Data Augmentation Project (QCRI)

The project is based on Python and is using the package `streamlit` to host it on the web. 

## Setup (Local Machine)
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

4. You will need to download the W2V model from the script that is given in the 'data' folder.
    ```sh
    cd data
    ./script.sh
    ```

5. Once all the packages are installed, you should be able to run the app on Streamlit locally
    ```sh
    # main.py is the file where the main code resides
    streamlit run main.py
    ```

## Deploying on a `Docker` Container

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

If you plan on making any changes to the repo, follow these steps:
* Fork it to your own Github profile
* Make the changes 
* Create a pull request
* And, I will review the request and accept it.
