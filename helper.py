import streamlit as st 
from transformers import (MarianMTModel, MarianTokenizer)
import tensorflow as tf
import re
import requests 
import json
from camel_tools.utils.charsets import AR_LETTERS_CHARSET

### ------------------------ Helper Functions ----------------------------- ###
@st.cache(allow_output_mutation=True)
def load_translator(model_name):
  """
  Loads the translation model from HuggingFace. It downloads and caches the model
  for future use by the app.

  Input Parameters
  ================
  model_name => HuggingFace link of the model (typically like this: Helsinki-NLP/opus-mt-ar-en).

  Return Parameters
  =================
  model => Loaded model for translation.
  tokenizer => Tokenizer for the translation model.
  """

  tokenizer = MarianTokenizer.from_pretrained(model_name)
  model = MarianMTModel.from_pretrained(model_name)
  return model, tokenizer

def translate_user_text_input(user_input):
  """
  Function translates the user text input from arabic to english using a Helsinki-NLP model
  from HuggingFace. The model translates the sentence and then sends it to the frontend for 
  displaying.

  Input Parameters
  ================
  user_input => Arabic sentence for translation (given by the user).

  Return Parameters
  =================
  translated_sentence => Returns the English translation of the arabic sentence back to the front-end.
  """
  
  model, tokenizer = load_translator("Helsinki-NLP/opus-mt-ar-en")
  translated = model.generate(**tokenizer(user_input, return_tensors="pt", padding=True))
  translated_sentence = ""
  for t in translated:
    translated_sentence = tokenizer.decode(t, skip_special_tokens=True)
  return translated_sentence

def is_replacable(token, pos_dict):
  """
  Helper function which tells the augmenting functions if we can removes all the uncessary 
  parts of a sentence and keeps only the NOUNS, VERBS and ADJECTIVES of the sentences for 
  augmentation. The parts of the sentence are coming from the POS tagger from Farasa API.

  Input Parameters
  ================
  token => Token from POS tagging.
  pos_dict => Parts of the Speech in the sentence.

  Return Parameters
  =================
  True/False (bool value) => tells the user whether there are POS which can be replaced.
  """

  if ner(token) != 'O':
     return False
  if token in pos_dict:
    if bool(set(pos_dict[token].split("+")) & set(['NOUN','V','ADJ'])):
      return True
  return False

def spl(text):
  """
  Helper function to split the sentence into three parts, first part, second part and augmented
  word. This helps to color the augmented words in the front-end.

  The augmented word is marked by a star from the augmenting functions and the star is removed
  and then sentence is split into 3 to be processed in the frontend.

  Input Parameters
  ================
  text => Sentence for splitting.

  Return Parameters
  =================
  rep => Augmented Word (Will be coloured in the frontend).
  fhalf => First half of the sentence.
  shalf => Second half of the sentence (after the augmented word).
  """

  fhalf = []
  shalf = []
  rep = ""
  first = True 
  for w in text.split():
    if "*" in w:
      rep = w.replace("*","").replace("_"," ")
      first = False
    elif first:
      fhalf.append(w)
    else:
      shalf.append(w)
  fhalf = " ".join(fhalf)
  shalf = " ".join(shalf)
  return rep, fhalf, shalf

def ner(text):
  """
  Helper function to check if the sentence contains a named entinty such as country, 
  famous person, names, places etc. so they are removed as those words contian meanings
  and cant be augmented.

  Input Parameters
  ================
  text => Input text

  Return Parameters
  =================
  result => Sentece with the named entities marked 
  """

  url = 'https://farasa.qcri.org/webapi/ner/'
  api_key = "KMxvdPGsKHXQAbRXGL"
  payload = {'text': text, 'api_key': api_key}
  data = requests.post(url, data=payload)
  result = json.loads(data.text)
  return result['text'][0].split("/")[1]

def models_data(file_name):
  """
  Helper function to open a JSON file and return the data to the front-end.

  Input Parameter
  ===============
  file_name => File name of the JSON file to be opened

  Return Parameter
  ================
  data => Data from the file returned to the frontend
  """

  f = open(file_name)
  data = json.load(f)
  return data

def seperate_punct(text):
  """
  Helper function to seperate the punctuations and not the arabic characters. The arabic characters are
  imported from the AR_LETTERS_CHARSET. 

  Input Parameters
  ================
  text => Takes a sentence for seperating the punctuations.

  Return Parameters
  =================
  ret => Returns the sentence without punctuations.
  """

  text = text.strip()
  text = " ".join(text.split())
  ret = ""
  for i,l in enumerate(text):
    if not i == len(text) - 1:
      if l in AR_LETTERS_CHARSET and text[i+1] != " " and not text[i+1] in AR_LETTERS_CHARSET:
        ret += l + " "
      elif not l in AR_LETTERS_CHARSET and text[i+1] != " " and text[i+1] in AR_LETTERS_CHARSET:
        ret += l + " "
      else:
        ret += l
    else:
      ret += l
  ret = " ".join(ret.split())
  return ret

def clean(text):
  """
  Helper function which process the arabic text and removes any and all punctutaion 
  and keeps only the arabic text. 

  Input Parameters
  ================
  text => Sentence for processing.

  Return Parameters
  =================
  text => Returns the clean text after processing.
  """

  punc = """،.:!?؟!:.,''!"#$%&'()*+, -./:;<=>?@[\]^_`{|}~"""
  for l in text:
    if l in punc and l != " ":
      text = text.replace(l,"")
  # keep only arabic text
  text = " ".join(re.findall(r'[\u0600-\u06FF]+', text))
  return text

def strip_punc(text):
  """
  Helper function to strip the sentence of punctuations and replace it with a space.

  Input Parameters
  ================
  text => A sentence to remove punctuations from.

  Return Parameter
  ================
  text => A sentence without any punctuations and all of them are replaced with a space.
  """

  remove = ""
  for l in reversed(text):
    if l in AR_LETTERS_CHARSET:
      break
    elif not l in AR_LETTERS_CHARSET:
      remove += l
  return text.replace(remove[::-1],"")

@st.cache(allow_output_mutation=True)
def convert_df_to_csv(df):
  return df.to_csv().encode('utf-8-sig')

### ----------------- End of Helper Functions ----------------------------- ###