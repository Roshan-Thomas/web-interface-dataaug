# Web Interface for Data Augmentation Project (QCRI)

The project is based on Python and is using the package `streamlit` to host it on the web. 

View on [Google Colab](https://colab.research.google.com/drive/1G_MKT4gnDDoX-hzyFABeornv-gubkRLy?usp=sharing)

## Setup (Local Machine)
Clone the Github repo to your local machine and follow the steps below.

```sh
# Install pipenv (to run environments)
# If you already have pip installed...
pip install pipenv

# If you have Fedora 28
sudo dnf install pipenv

# If you have Homebrew (MacOS) [This is discouraged]
brew install pipenv
```

Start a new `pipenv` environment 
```sh
pipenv shell
```

Install the required packages from Pipfile
```
pipenv install
```

You will need to download the W2V model from the script that is given in the 'data' folder.
```bash
cd data
./script.sh
```

Once all the packages are installed, you should be able to run the app on Streamlit locally
```sh
# main.py is the file where 
# the main code resides
streamlit run main.py
```

## Deploying on a Docker Container

Build the docker app
```sh
sudo docker build -t dataaug-webapp:latest .
```

Run the web app using docker with port 8051 and then forwarding it to port 8080 on the server
```sh
sudo docker run -p 8080:8501 dataaug-webapp
```

Run the web app in the background
```sh
sudo docker run -p 8080:8501 -d dataaug-webapp
```

## Contribution

If you plan on making any changes to the repo, follow these steps:
* Fork it to your own Github profile
* Make the changes 
* Create a pull request
* And, I will look at the request and accept it.
