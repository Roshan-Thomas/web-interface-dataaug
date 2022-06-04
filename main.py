import streamlit as st 

import json
import requests
from transformers import GPT2LMHeadModel, pipeline, GPT2TokenizerFast
import gensim

# for now only doing arabert and aragpt2 models 
# TODO: aravec model

st.title("Web Interface")
st.write("Loading data...")

@st.cache(suppress_st_warning=True)
def text_generation(model_name, text):
    punc = "،.:!?"
    text = text.strip()
    if text[-1] in punc:
        text = text[0:-1]
    text = text.strip()
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    generation_pipeline = pipeline("text-generation",model=model,tokenizer=tokenizer)
    l = []
    #method 1
    for n in range(1,4):
        for i in range(5):
            pred = generation_pipeline(text,
                return_full_text = True,
                pad_token_id=tokenizer.eos_token_id,
                num_beams=10 ,
                max_length=len(text.split()) + n,
                top_p=0.9,
                repetition_penalty = 3.0,
                no_repeat_ngram_size = 3)[0]['generated_text'].replace("."," ").replace("،"," ").replace(":"," ").strip()
            pred = " ".join(pred.split())
            if not pred in l:
                l.append(pred)
    # method 2
    text = " ".join(text.split()[0:-1])
    for n in range(1,4):
        for i in range(5):
            pred = generation_pipeline(text,
                return_full_text = True,
                pad_token_id=tokenizer.eos_token_id,
                num_beams=10 ,
                max_length=len(text.split()) + n,
                top_p=0.9,
                repetition_penalty = 3.0,
                no_repeat_ngram_size = 3)[0]['generated_text'].replace("."," ").replace("،"," ").replace(":"," ").strip()
            pred = " ".join(pred.split())
            if not pred in l:
                l.append(pred)

    return l

# TODO: w2v function

@st.cache(suppress_st_warning=True)
def fill_mask(model, text):
    model = pipeline('fill-mask', model= model)
    l = []
    for token in text.split()[1:]:
        masked_text = text.replace(token,"[MASK]")
        pred = model(masked_text , top_k = 20)
        for i in pred:
            output = i['token_str']
            if not len(output) < 2 and not "+" in output and not "[" in output:
                l.append(text.replace(token, i['token_str']))
    return l

# TODO: aravec models

text = "انه موضوع شيق"

# BERT-based fill mask models (4)
arabert = fill_mask('aubmindlab/bert-base-arabert',text)
arabertv2 = fill_mask('aubmindlab/bert-large-arabertv2',text)
arabertv02 = fill_mask('aubmindlab/bert-large-arabertv02',text)
arabertv01 = fill_mask('aubmindlab/bert-base-arabertv01',text)

# GPT2-based text generation models
aragpt2 = text_generation('aubmindlab/aragpt2-medium',text) ## use the mega model

output = "org: " + text + "\n"

gpt = list(set(aragpt2))
for aug in gpt:
    st.write("GPT2-based: " + aug + "\n")

bert = list(set(arabert) | set(arabertv2) | set(arabertv02) | set(arabertv01))
for aug in bert:
    st.write("BERT-based: " + aug + "\n")

st.write("\n\nDone!!")