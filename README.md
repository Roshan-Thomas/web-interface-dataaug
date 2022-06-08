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

Unzip the file into the data folder
```bash
python3 pynuzip.py full_grams_cbow_100_twitter.zip
```

Once all the packages are installed, you should be able to run the app on Streamlit locally
```sh
# main.py is the file where 
# the main code resides
streamlit run main.py
```

## Contribution

If you plan on making any changes to the repo, follow these steps:
* Fork it to your own Github profile
* Make the changes 
* Create a pull request
* And, I will look at the request and accept it.
