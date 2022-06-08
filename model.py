import streamlit as st 
import json
import requests
from transformers import GPT2LMHeadModel, pipeline, GPT2TokenizerFast
import gensim
import re
import time

def process(text):
  # remove any punctuations in the text
  punc = "،.:!?"
  text = text.strip()
  if text[-1] in punc:
      text = text[0:-1]
  text = text.strip()
  # keep only arabic text
  text = " ".join(re.findall(r'[\u0600-\u06FF]+', text))
  return text

### -------------------------------- GPT ------------------------------------ ###

def load_GPT(model_name):
  model = GPT2LMHeadModel.from_pretrained(model_name)
  tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
  generation_pipeline = pipeline("text-generation",model=model,tokenizer=tokenizer)
  return model , tokenizer , generation_pipeline

def GPT(model,tokenizer , generation_pipeline ,sentence):
  org_text = sentence
  sentence = process(sentence)
  l = []
  if len(sentence.split()) < 11:
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

### ------------------------------- W2V ------------------------------------ ###

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
  if len(sentence.split()) < 11:
    for token in sentence.split():
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
      if is_replacable(token):
        if exist:
          try:
            most_similar = model.wv.most_similar( token, topn=5 )
          except:
            most_similar = model.most_similar( token, topn=5 )
          for term, score in most_similar:
                if term != token:
                    term = term.replace("_"," ")
                    aug = sentence.replace(token,term)
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

def load_bert(model):
  model = pipeline('fill-mask', model= model)
  return model

def bert(model, sentence):  # Contextual word embeddings
  org_text = sentence
  sentence = process(sentence)
  l = []
  if len(sentence.split()) < 11:
    for token in sentence.split():
        if is_replacable(token):
          masked_text = sentence.replace(token,"[MASK]")
          pred = model(masked_text , top_k = 20)
          for i in pred:
            if isinstance(i, dict):
              output = i['token_str']
              if not len(output) < 2 and not "+" in output and not "[" in output:
                aug = sentence.replace(token, i['token_str'])
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

def is_replacable(token):
   if token in ["يا","و"]:
     return False
   return True