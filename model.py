import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st 
import json
import requests
from transformers import GPT2LMHeadModel, pipeline, GPT2TokenizerFast
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import pipeline
import gensim
import re
from random import choice
import time
import nlpaug.augmenter.word as naw
import tensorflow as tf

### ------------------------ Back Translation ------------------------------ ###

# st.cache(allow_output_mutation=True)

@st.experimental_memo
def load_models_bt(from_model_name, to_model_name):
  device = 'cpu'
  if tf.test.gpu_device_name():
    device = 'cuda'
  back_translation = naw.BackTranslationAug(
      from_model_name=from_model_name,
      to_model_name=to_model_name,
      device=device, # needs to be changed to 'cpu'  when running on the server
  )
  return (back_translation)

def back_translate(selected_languages, text):
  """
  List of Languages Used
    - ar-en	(English)
    - ar-fr	(French)	
    - ar-tr	(Turkish)
    - ar-ru	(Russian)
    - ar-pl	(Polish)
    - ar-it	(Italian)
    - ar-es	(Spanish)
    - ar-el (Greek)
    - ar-de (German)
    - ar-he (Hebrew)
  """

  all_sentences = []
  available_languages = ['ar-en', 'ar-fr', 'ar-tr', 'ar-ru', 'ar-pl', 'ar-it', 'ar-es', 'ar-el', 'ar-de', 'ar-he']

  loading_state_bt = st.text("Loading & Augmenting Back Translating Models...")
  tic = time.perf_counter()

  for model in available_languages:
    model_name = model.split('-')
    back_translation = load_models_bt(f'Helsinki-NLP/opus-mt-{model_name[0]}-{model_name[1]}',
                                      f'Helsinki-NLP/opus-mt-{model_name[1]}-{model_name[0]}')
    bt_sentence = back_translation.augment(text)
    all_sentences.append(bt_sentence)

  toc = time.perf_counter()
  loading_state_bt.write("Loading & Augmenting Back Translating Models done ✅: " + str(toc-tic) + " seconds")

  return all_sentences

### ---------------------- End of Back Translation ------------------------- ###

### ------------------------------- W2V ------------------------------------ ###

@st.cache(allow_output_mutation=True)
def load_w2v(model_path):
  try:
      model = gensim.models.KeyedVectors.load_word2vec_format(model_path,binary=True,unicode_errors='ignore')
  except:
      model = gensim.models.Word2Vec.load(model_path)
  return model

def w2v(model,sentence):
  org_text = sentence
  sentence = process(sentence)
  l = []
  augs = []
  if len(sentence.split()) < 15 and len(sentence.split()) > 2:
    for i,token in enumerate(sentence.split()):
      try:
        word_vectors = model.wv
        if token in word_vectors.key_to_index:
           exist = True
        else:
           exist = False
      except:
        if token in model:
          exist = True
        else:
          exist = False
      if is_replacable(token,pos(sentence)):
        if exist:
          try:
            most_similar = model.wv.most_similar( token, topn=5 )
          except:
            most_similar = model.most_similar( token, topn=5 )
          for term, score in most_similar:
                if term != token:
                    term = "*" + term
                    s = sentence.split()
                    s[i] = term
                    aug = " ".join(s)
                    if not aug.replace(".","").replace("،","").replace("!","").replace("؟","").replace(":","") in augs:
                      augs.append(aug.replace(".","").replace("،","").replace("!","").replace("؟","").replace(":",""))
                      org_text = " ".join(org_text.split())
                      l.append(org_text.replace(sentence,aug))
  return l

def aug_w2v(model_path,text):   # text here is a list of sentences
  loading_state_w2v = st.text("Loading W2V...")
  tic = time.perf_counter()
  model = load_w2v(model_path)
  toc = time.perf_counter()
  loading_state_w2v.text("Loading W2V done ✅: " + str(toc-tic) + " seconds")
  augment_state_w2v = st.text("Augmenting with W2V...")
  tic = time.perf_counter()
  if isinstance(text, str):
    ret = w2v(model,text)
    toc = time.perf_counter()
    augment_state_w2v.text("Augmenting with W2V done ✅: " + str(toc-tic) + " seconds")
    return ret
  else:
    all_sentences = []
    for sentence in text:
       sentence = sentence.strip()
       all_sentences.append([sentence,w2v(model,sentence)])
    toc = time.perf_counter()
    augment_state_w2v.text("Augmenting with W2V done ✅: " + str(toc-tic) + " seconds")
    return all_sentences


### ------------------------ End of W2V ------------------------------------ ###

### ------------------------------- BERT ----------------------------------- ###

@st.cache(allow_output_mutation=True)
def load_bert(model):
  model = pipeline('fill-mask', model=model)
  return model

def bert(model, sentence):  # Contextual word embeddings
  org_text = sentence
  sentence = process(sentence)
  l = []
  augs = []
  if len(sentence.split()) < 15 and len(sentence.split()) > 2:
    for n,token in enumerate(sentence.split()):
        if is_replacable(token,pos(sentence)):
          s = sentence.split()
          try:
            s[n] = "<mask>"
            masked_text = " ".join(s)
            pred = model(masked_text , top_k = 5)
          except:
            s[n] = "[MASK]"
            masked_text = " ".join(s)
            pred = model(masked_text , top_k = 5)
          for i in pred:
            if isinstance(i, dict):
              output = i['token_str']
              if not output == token:
                if not len(output) < 2 and not "+" in output and not "[" in output:
                  term = "*"+i['token_str']
                  s = sentence.split()
                  s[n] = term
                  aug = " ".join(s)
                  if not aug.replace(".","").replace("،","").replace("!","").replace("؟","").replace(":","") in augs:
                        augs.append(aug.replace(".","").replace("،","").replace("!","").replace("؟","").replace(":",""))
                        org_text = " ".join(org_text.split())
                        l.append(org_text.replace(sentence,aug))
  return l

def aug_bert(model,text):  # text here is a list of sentences
  loading_state_bert = st.text("Loading AraBERT...")
  tic = time.perf_counter()
  model = load_bert(model)
  toc = time.perf_counter()
  loading_state_bert.text("Loading AraBERT done ✅: " + str(toc-tic) + " seconds")
  augment_state_bert = st.text("Augmenting with AraBERT...")
  tic = time.perf_counter()
  if isinstance(text, str):
    ret = bert(model, text)
    toc = time.perf_counter()
    augment_state_bert.text("Augmenting with AraBERT done ✅: " + str(toc-tic) + " seconds")
    return ret
  else:
    all_sentences = []
    for sentence in text:
      sentence = sentence.strip()
      all_sentences.append([sentence,bert(model, sentence)])
    toc = time.perf_counter()
    augment_state_bert.text("Augmenting with AraBERT done ✅: " + str(toc-tic) + " seconds")
    return all_sentences

### ------------------------ End of BERT ----------------------------------- ###

### -------------------------------- GPT ------------------------------------ ###

@st.cache(allow_output_mutation=True)
def load_GPT(model_name):
  model = GPT2LMHeadModel.from_pretrained(model_name)
  tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
  generation_pipeline = pipeline("text-generation",model=model,tokenizer=tokenizer)
  return model , tokenizer , generation_pipeline

def GPT(model,tokenizer , generation_pipeline ,sentence):
  org_text = sentence
  sentence = process(sentence)
  l = []
  if len(sentence.split()) < 16:
    input_ids = tokenizer.encode(sentence, return_tensors="pt")
    for n in range(1,4):
      for i in range(5):
        pred = generation_pipeline(sentence,
          return_full_text = True,
          pad_token_id=tokenizer.eos_token_id,
          num_beams=10 ,
          max_length=len(input_ids[0]) + n,
          top_p=0.9,
          repetition_penalty = 3.0,
          no_repeat_ngram_size = 3)[0]['generated_text'].replace("."," ").replace("،"," ").replace(":"," ").strip()
        pred = " ".join(pred.split())
        if not pred in l:
          l.append(org_text.replace(sentence,pred))
  return l

def aug_GPT(model_name,text):  # text here can be list of sentences or on string sentence
  loading_state_gpt = st.text("Loading AraGPT2...")
  tic = time.perf_counter()
  model , tokenizer , generation_pipeline = load_GPT(model_name)
  toc = time.perf_counter()
  loading_state_gpt.write("Loading AraGPT2 done ✅: " + str(toc-tic) + " seconds")
  augment_state_gpt = st.text("Augmenting with AraGPT2...")
  tic = time.perf_counter()
  if isinstance(text, str):
    ret = GPT(model,tokenizer , generation_pipeline ,text)
    toc = time.perf_counter()
    augment_state_gpt.text("Augmenting with AraGPT2 done ✅: " + str(toc-tic) + " seconds")
    return ret
  else:
    all_sentences = []
    for sentence in text:
      sentence = sentence.strip()
      all_sentences.append([sentence,GPT(model,tokenizer , generation_pipeline ,sentence)])
    toc = time.perf_counter()
    augment_state_gpt.text("Augmenting with AraGPT2 done ✅: " + str(toc-tic) + " seconds")
    return all_sentences

### ------------------------ End of GPT ------------------------------------ ###  

### ----------------------- Text-to-Text ----------------------------------- ###

@st.experimental_memo
def load_m2m(model_name): ## use the facebook/m2m100-12B-last-ckpt
  if "m2m" in model_name:
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    token = "ar"
  else:
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    token = "ar_AR"
  return model,tokenizer,token

def m2m(model,tokenizer,token,sentence):
    org_text = sentence
    sentence = process(sentence)
    if len(sentence.split()) < 15 and len(sentence.split()) > 2:
      if token == "ar":
        id = tokenizer.get_lang_id(token)
      else:
        id = tokenizer.lang_code_to_id[token]
      tokenizer.src_lang = token
      encoded_hi = tokenizer(sentence, return_tensors="pt")
      generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=id)
      aug = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
      clean_aug = aug.replace(".","").replace("،","").replace("!","").replace("؟","").replace(":","")
      if not clean_aug == org_text.replace(".","").replace("،","").replace("!","").replace("؟","").replace(":",""):
        if " ".join(re.findall(r'[\u0600-\u06FF]+', clean_aug)) == clean_aug:
          return [aug]
    return []

def aug_m2m(model_name,text):
    loading_state_m2m = st.text("Loading M2M...")
    tic = time.perf_counter()
    model , tokenizer , token = load_m2m(model_name)
    toc = time.perf_counter()
    loading_state_m2m.text("Loading M2M done ✅: " + str(toc-tic) + " seconds")
    augment_state_m2m = st.text("Augmenting with M2M...")
    tic = time.perf_counter()
    if isinstance(text, str):
      ret = m2m(model,tokenizer , token ,text)
      toc = time.perf_counter()
      augment_state_m2m.text("Augmenting with AraBERT done ✅: " + str(toc-tic) + " seconds")
      return ret
    else:
      all_sentences = []
      for sentence in text:
        sentence = sentence.strip()
        all_sentences.append([sentence,m2m(model,tokenizer , token ,sentence)])
      toc = time.perf_counter()
      augment_state_m2m.text("Augmenting with AraBERT done ✅: " + str(toc-tic) + " seconds")
      return all_sentences

### ------------------------ End of Text-to-Text --------------------------- ###

### ------------------------ General Functions ----------------------------- ###

def process(text):
  # remove any punctuations in the text
  punc = """،.:!?؟!:.,''!"#$%&'()*+, -./:;<=>?@[\]^_`{|}~"""
  text = text.strip()
  if text[-1] in punc:
      text = text[0:-1]
  text = text.strip()
  # keep only arabic text
  text = " ".join(re.findall(r'[\u0600-\u06FF]+', text))
  return text


def is_replacable(token,pos_dict):
  # removes all the unncessary part of a sentence and keeps only the main ones for augmentation 
  if ner(token) != 'O':
     return False
  if token in pos_dict:
    if bool(set(pos_dict[token].split("+")) & set(['NOUN','V','ADJ'])):
      return True
  return False

def spl(text):
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
  url = 'https://farasa.qcri.org/webapi/ner/'
  api_key = "KMxvdPGsKHXQAbRXGL"
  payload = {'text': text, 'api_key': api_key}
  data = requests.post(url, data=payload)
  result = json.loads(data.text)
  return result['text'][0].split("/")[1]

### ----------------- End of General Functions ----------------------------- ###

### -------------------- Random Sentence Generator ------------------------- ###

@st.experimental_memo
def delete_unncessary_lines(file_name:str):
  sentences = []
  with open(file_name, 'r') as input:
    with open("./data/temp.txt", "w") as output:
      for line in input:
        if "#" not in line.strip("\n"):
          output.write(line)
  os.replace('./data/temp.txt', file_name)


def random_sentence(file_name:str):
  sentences = []
  temp_list = []  # temporary list (hold temp values)
  res = []        # temporary list (hold temp values)
  delete_unncessary_lines(file_name)

  with open(file_name, 'r') as f:
    x = f.readlines()
    for line in x:
      temp_list += line.split(" .")
    
    for ele in temp_list:
        if ele.strip():
            res.append(ele)

    for i in range(len(res)):
      if len(res[i].split()) < 15:
        sentences.append(res[i].strip())

  selected_sentence = choice(sentences)
  return selected_sentence

### ------------------- End of Random Sentence Generator ------------------- ###

### ------------------------- Farasa API ----------------------------------- ###

def pos(text):
  url = 'https://farasa.qcri.org/webapi/pos/'
  api_key = "KMxvdPGsKHXQAbRXGL"
  payload = {'text': text, 'api_key': api_key}
  data = requests.post(url, data=payload)
  result = json.loads(data.text)
  text  = text.split()
  pos_dict  = {}
  for n in range(len(result["text"])):
    i = result["text"][n]
    if "+" == i['surface'][0]:
      word = "".join(s.strip() for s in result["text"][n-1]['surface'].split("+"))
      word = word + i['surface'].replace("+","").strip()
      if word in text:
        pos_dict[word] = result["text"][n-1]['POS']
    if "+" == i['surface'][-1]:
      word = "".join(s.strip() for s in result["text"][n+1]['surface'].split("+"))
      word = i['surface'].replace("+","").strip() + word
      if word in text:
       pos_dict[word] = result["text"][n+1]['POS']
    else:
      word = "".join(s.strip() for s in i['surface'].split("+"))
      if word in text:
        pos_dict[word] = i['POS']
  return pos_dict

### ------------------------- End of Farasa API ---------------------------- ###