### ---------------------------- Imports ----------------------------------- ###
from transformers import (
    GPT2LMHeadModel, pipeline, GPT2TokenizerFast,
    pipeline, M2M100ForConditionalGeneration,
    M2M100Tokenizer, MBartForConditionalGeneration,
    MBart50TokenizerFast, AutoTokenizer, AutoModel)
import streamlit as st
from helper import (is_replacable, seperate_punct,
                    clean, strip_punc, convert_df_to_csv)
import torch
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import nlpaug.augmenter.word as naw
from random import choice
import gensim
import gensim.downloader
from statistics import mean
import numpy as np
import pandas as pd
import string
import time
import requests
import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


### ----------------------- End of Imports --------------------------------- ###


### ------------------------ Back Translation ------------------------------ ###

@st.experimental_memo
def load_models_bt(from_model_name, to_model_name):
    """
    This function loads the Back Translation Models from Huggingface. It also checks if
    the system has GPU or not, and if it has GPU then the model will use GPU, and if it 
    doesnt have GPU then it will only use the CPU Resources.

    Here we use an experimental cache system of Streamlit as it has shown to be faster 
    when loading the models the second time. It initially takes longer than @st.cache but
    in the long term its faster. 

    Input Parameters
    ===========
    from_model_name => Translating model which translated from arabic > english
    to_model_name => Translating model with translates the sentences from english > arabic
    """

    device = 'cpu'
    if tf.test.gpu_device_name():
        device = 'cuda'
    back_translation = naw.BackTranslationAug(
        from_model_name=from_model_name,
        to_model_name=to_model_name,
        device=device,  # needs to be changed to 'cpu'  when running on the server
    )
    return (back_translation)


def double_back_translate(text):
    """
    This function does double back translation, so it does ar > en > ar_1 > en > ar_2. This gives
    us two augmented arabic sentences. 

    The list of Languages Used:
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

    Input Parameters
    ================
    text => This is the user inputed text which is to be back translated using the multiple languages
    """

    all_sentences = []
    available_languages = ['ar-en', 'ar-fr', 'ar-tr', 'ar-ru',
                           'ar-pl', 'ar-it', 'ar-es', 'ar-el', 'ar-de', 'ar-he']

    loading_state_bt = st.text(
        "Loading & Augmenting Back Translating Models...")
    tic = time.perf_counter()

    for model in available_languages:
        model_name = model.split('-')
        back_translation = load_models_bt(f'Helsinki-NLP/opus-mt-{model_name[0]}-{model_name[1]}',
                                          f'Helsinki-NLP/opus-mt-{model_name[1]}-{model_name[0]}')
        bt_sentence_1 = back_translation.augment(text)
        bt_sentence_2 = back_translation.augment(bt_sentence_1)
        all_sentences.append(bt_sentence_1)
        all_sentences.append(bt_sentence_2)

    toc = time.perf_counter()
    loading_state_bt.write(
        "Loading & Augmenting Back Translating Models done ✅: " + str(round(toc-tic, 3)) + " seconds")

    return all_sentences

### ---------------------- End of Back Translation ------------------------- ###


### ------------------------------- W2V ------------------------------------ ###


@st.cache(allow_output_mutation=True)
def load_w2v(ar_model, en_model):
    """
    This function loads the word-to-vector (w2v) model from our local directories. It also is a cached 
    function using streamlit that only needs to be loaded once and the next time someone
    uses the function it loads immedietly because it is present in the cache.

    Input Parameters
    ================
    model_path => The local path of the w2v model to be imported

    Return Parameters
    =================
    model => The loaded model 
    """

    try:
        ar_model = gensim.models.KeyedVectors.load_word2vec_format(
            ar_model, binary=True, unicode_errors='ignore')
    except:
        ar_model = gensim.models.Word2Vec.load(ar_model)
    en_model = gensim.downloader.load(en_model)
    return ar_model, en_model


def w2v(ar_model, en_model, sentence):
    """
    This function uses a word-to-vector (w2v) model and a sentence (user inputed) and cleans it and then 
    augments the sentence based on the pretrained w2v model.

    Input Parameters
    ================
    model => w2v model (from Local Directories)
    sentence => A sentence to augment using the w2v model (typically the user inputed sentence from the frontend)

    Return Parameters
    =================
    l => A list of the augmented sentences
    """

    cleaned = clean(sentence)
    sentence = seperate_punct(sentence)
    l = []
    augs = []
    if len(sentence.split()) < 15 and len(sentence.split()) > 2:
        for i, token in enumerate(sentence.split()):
            pos_dict = pos(cleaned)
            if token in cleaned and is_replacable(token, pos_dict):
                if pos_dict[token] == 'FOREIGN':
                    model_to_use = en_model
                else:
                    model_to_use = ar_model
                try:
                    word_vectors = model_to_use.wv
                    if token in word_vectors.key_to_index:
                        exist = True
                    else:
                        exist = False
                except:
                    if token in model_to_use:
                        exist = True
                    else:
                        exist = False
                    if exist:
                        try:
                            most_similar = model_to_use.wv.most_similar(
                                token, topn=5)
                        except:
                            most_similar = model_to_use.most_similar(
                                token, topn=5)
                for term, score in most_similar:
                    if term != token:
                        term = "*" + term
                        s = sentence.split()
                        s[i] = term
                        aug = " ".join(s)
                        if not clean(aug) in augs:
                            augs.append(clean(aug))
                            aug = " ".join(aug.split())
                            l.append(aug)
    return l


def aug_w2v(ar_model, en_model, text, model_name):
    """
    This function is the main augmenting code of the word-to-vector (w2v) model. This function calls the 
    load_w2v() and w2v() functions. This function also calculates the total time it takes to
    load the model and augment the sentence and it outputs it so the user has a better idea of 
    how long it takes for the model to run. 

    This function also has the capability to take an input of multiple sentences (in a list)
    and augment them as well. 

    Note: Right now we are only using the function to augment one sentence, but it can also augment
    an entire list of sentences.

    Input Parameters
    ================
    model_path => The local path of the w2v Model.
    text => A sentence to be augmented (typically the user's input).
    model_name => Custom name of the model (to be displayed to user).

    Return Parameters
    =================
    all_sentences => Returns all the augmented sentence from the w2v model.
    """

    loading_state_w2v = st.text(f"Loading {model_name}...")
    tic = time.perf_counter()
    ar_model, en_model = load_w2v(ar_model, en_model)
    toc = time.perf_counter()
    loading_state_w2v.text("Loading W2V done ✅: " +
                           str(round(toc-tic, 3)) + " seconds")
    augment_state_w2v = st.text(f"Augmenting with {model_name}...")
    tic = time.perf_counter()
    if isinstance(text, str):
        ret = w2v(ar_model, en_model, text)
        toc = time.perf_counter()
        augment_state_w2v.text(
            f"Augmenting with {model_name} done ✅: " + str(round(toc-tic, 3)) + " seconds")
        return ret
    else:
        all_sentences = []
        for sentence in text:
            sentence = sentence.strip()
            all_sentences.append([sentence, w2v(ar_model, en_model, sentence)])
        toc = time.perf_counter()
        augment_state_w2v.text(
            f"Augmenting with {model_name} done ✅: " + str(round(toc-tic, 3)) + " seconds")
        return all_sentences

### ------------------------ End of W2V ------------------------------------ ###


### ------------------------------- BERT ----------------------------------- ###


@st.cache(allow_output_mutation=True)
def load_bert(model):
    """
    Loads the BERT model and downloades it from HuggingFace using the transformers
    package. This function is cached by streamlit so it can be loaded faster the next
    time someone calls it.

    Input Parameters
    ================
    model => HuggingFace path for the model (typically like this: aubmindlab/bert-large-arabertv2)

    Return Parameters
    =================
    model => Returns the loaded model which can be used for augmenting 
    """

    model = pipeline('fill-mask', model=model)
    return model


def bert(model, sentence):
    """
    This function uses the BERT model to augment the sentence. It first reads the sentence 
    and then sees if its less that 15 words and greater than 2 words and then proceeds to 
    process the sentence. Here the BERT models places a mask on a single word and then the 
    model predicts the rest of the sentence.

    The augmentation technique used is 'Contextual word embeddings'.

    Input Parameters
    ================
    model => HuggingFace path for the model (typically like this: aubmindlab/bert-large-arabertv2)
    sentence => A sentence for the model to augment (typically it is the user inputted sentence)

    Return Parameters
    =================
    l => Returns the augmented sentences in a list
    """

    cleaned = clean(sentence)
    sentence = seperate_punct(sentence)
    l = []
    augs = []
    if len(sentence.split()) < 15 and len(sentence.split()) > 2:
        for n, token in enumerate(sentence.split()):
            if token in cleaned and is_replacable(token, pos(sentence)):
                s = sentence.split()
                try:
                    s[n] = "<mask>"
                    masked_text = " ".join(s)
                    pred = model(masked_text, top_k=5)
                except:
                    s[n] = "[MASK]"
                    masked_text = " ".join(s)
                    pred = model(masked_text, top_k=5)
                for i in pred:
                    if isinstance(i, dict):
                        output = i['token_str']
                        if not output == token:
                            if not len(output) < 2 and clean(output) == output:
                                term = "*"+i['token_str']
                                s = sentence.split()
                                s[n] = term
                                aug = " ".join(s)
                                if not clean(aug) in augs:
                                    augs.append(clean(aug))
                                    aug = " ".join(aug.split())
                                    l.append(aug)
    return l


def multi_bert(model, sentence):
    """
    This function augments a given sentence two times using the same bert model
    so as to give more data augmented sentences.

    Input Parameters
    ================
    ar_model => Arabic Model
    en_model => English Model
    sentence => User inputed sentence

    Output Parameters
    =================
    ret => A list of all augmented sentences
    """

    l = bert(model, sentence)
    ret = []
    for i in l:
        ret += bert(model, i)
    return ret


def aug_bert(bert_model, text, model_name):
    """
    This function is the display function of the BERT model where we call the 
    load_bert() and bert() functions to process the given sentence or list of 
    sentences to produce a list of augmented sentences.

    Input Parameters
    ================
    model => HuggingFace link for the BERT model (typically like this: aubmindlab/bert-large-arabertv2)
    text => Sentence for the model to augment (typically the user inputed sentence) 
    model_name => A string which gives the display name of the model (This is because multiple BERT models use the same function)

    Return Parameters
    =================
    all_sentences => Returns all the augmented sentences to the frontend
    """

    loading_state_bert = st.text(f"Loading {model_name}...")
    tic = time.perf_counter()
    model = load_bert(bert_model)
    toc = time.perf_counter()
    loading_state_bert.text(
        f"Loading {model_name} done ✅: " + str(round(toc-tic, 3)) + " seconds")
    augment_state_bert = st.text(f"Augmenting with {model_name}...")
    tic = time.perf_counter()
    if isinstance(text, str):
        ret = multi_bert(model, text)
        toc = time.perf_counter()
        augment_state_bert.text(
            f"Augmenting with {model_name} done ✅: " + str(round(toc-tic, 3)) + " seconds")
        return ret
    else:
        all_sentences = []
        for sentence in text:
            sentence = sentence.strip()
            all_sentences.append(
                [sentence, multi_bert(model, sentence)])
        toc = time.perf_counter()
        augment_state_bert.text(
            f"Augmenting with {model_name} done ✅: " + str(round(toc-tic, 3)) + " seconds")
        return all_sentences

### ------------------------ End of BERT ----------------------------------- ###


### -------------------------------- GPT ------------------------------------ ###


@st.cache(allow_output_mutation=True)
def load_GPT(model_name):
    """
    Loads the AraGPT2 model from HuggingFace where it downloads it using the package
    'transformers' and caches it so the augmentation can take place. 

    Input Parameters
    ================
    model_name => Hugging Face link of the model (typically like this: aubmindlab/bert-large-arabertv2).

    Return Parameters
    =================
    model => Loaded GPT2 model.
    tokenizer => Tokenizer for the GPT2 model.
    generation_pipeline => Pipeline for text generation by the GPT2 model.
    """

    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    generation_pipeline = pipeline(
        "text-generation", model=model, tokenizer=tokenizer)
    return model, tokenizer, generation_pipeline


def GPT(model, tokenizer, generation_pipeline, sentence):
    """
    This function uses the GPT2 model to augment text. It takes in a sentence with less
    than 15 words (as no. of words increase the no. of outputed sentences also increases)
    and more than 2 words and generates a sentence. It usually removes the last word and 
    generates 2 or 3 more words based on the context of the sentence.

    Input Parameters
    ================
    model => Model path from HuggingFace (typically like this: aubmindlab/bert-large-arabertv2).
    tokenizer => Tokenizer for the AraGPT2 model.
    generation_pipeline => Pipeline for text generation by the GPT2 model.
    sentence => A sentence for augmenting (typically user inputed sentence).

    Return Parameters
    =================
    l => Returns a list of the augmented sentences.
    """

    org_text = sentence
    sentence = clean(sentence)
    l = []
    if len(sentence.split()) < 15 and len(sentence.split()) > 2:
        input_ids = tokenizer.encode(sentence, return_tensors="pt")
        for n in range(1, 4):
            for i in range(2):
                pred = generation_pipeline(sentence,
                                           return_full_text=False,
                                           pad_token_id=tokenizer.eos_token_id,
                                           num_beams=10,
                                           max_length=len(input_ids[0]) + n,
                                           top_p=0.9,
                                           repetition_penalty=3.0,
                                           no_repeat_ngram_size=3)[0]['generated_text'].replace(".", " ").replace("،", " ").replace(":", " ").strip()
                pred = " ".join(pred.split()).strip()
                if not pred == "":
                    pred = "*" + pred.replace(" ", "_")
                    aug = strip_punc(org_text) + " " + pred
                    org_text = " ".join(org_text.split())
                    pred = org_text.replace(strip_punc(org_text), aug)
                    if not pred in l and not pred == org_text:
                        l.append(pred)
    return l


def aug_GPT(model_name, text):
    """
    This is the display function of the GPT2 model where the load_GPT() and GPT() functions are 
    called to augment either a sentence or a list of sentences. This function also calculates the 
    time it takes for loading and augmenting sentences with the AraGPT2 model so the user can see it.

    Input Parameters
    ================
    model_name => HuggingFace link for the model (typically like this: aubmindlab/bert-large-arabertv2).
    text => A sentence or list of sentences (typically the user inputed sentence).

    Return Parameters
    =================
    all_sentences => Returns all the augmented sentences by the GPT2 model.
    """

    loading_state_gpt = st.text("Loading AraGPT2...")
    tic = time.perf_counter()
    model, tokenizer, generation_pipeline = load_GPT(model_name)
    toc = time.perf_counter()
    loading_state_gpt.write("Loading AraGPT2 done ✅: " +
                            str(round(toc-tic, 3)) + " seconds")
    augment_state_gpt = st.text("Augmenting with AraGPT2...")
    tic = time.perf_counter()
    if isinstance(text, str):
        ret = GPT(model, tokenizer, generation_pipeline, text)
        toc = time.perf_counter()
        augment_state_gpt.text(
            "Augmenting with AraGPT2 done ✅: " + str(round(toc-tic, 3)) + " seconds")
        return ret
    else:
        all_sentences = []
        for sentence in text:
            sentence = sentence.strip()
            all_sentences.append(
                [sentence, GPT(model, tokenizer, generation_pipeline, sentence)])
        toc = time.perf_counter()
        augment_state_gpt.text(
            "Augmenting with AraGPT2 done ✅: " + str(round(toc-tic, 3)) + " seconds")
        return all_sentences

### ------------------------ End of GPT ------------------------------------ ###


### ----------------------- Text-to-Text ----------------------------------- ###


@st.experimental_memo
def load_m2m(model_name):
    """
    Loads the m2m (Text-to-Text) model from HuggingFace to the streamlit app. It 
    downloaded and caches the model so it can be used further for whichever user
    uses it.

    Input Parameters
    ================
    model_name => HuggingFace link for the model (typically like this: aubmindlab/bert-large-arabertv2).

    Return Parameters
    =================
    model => Loaded m2m model.
    tokenizer => Tokenizer for the m2m model.
    token => token is 'ar' or 'ar_AR' based on which m2m model is used .
    """

    if "m2m" in model_name:
        model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        token = "ar"
    else:
        model = MBartForConditionalGeneration.from_pretrained(model_name)
        tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
        token = "ar_AR"
    return model, tokenizer, token


def m2m(model, tokenizer, token, sentence):
    """
    Uses the m2m model to augment a sentence if its less than 15 words and more than 2 words long.
    It uses the text-to-text data augmentation technique to generate sentences.

    Input Parameters
    ================
    model => HuggingFace link for the model (typically like this: aubmindlab/bert-large-arabertv2).
    tokenizer => Tokenizer for the m2m model.
    token => Token for the m2m model (decided in the load_m2m() function).
    sentence => Sentence to be augmented by the m2m model.

    Return Parameters
    =================
    aug => Returns the augmented sentence to the frontend.
    """

    org_text = sentence
    sentence = clean(sentence)
    if len(sentence.split()) < 15 and len(sentence.split()) > 2:
        if token == "ar":
            id = tokenizer.get_lang_id(token)
        else:
            id = tokenizer.lang_code_to_id[token]
        tokenizer.src_lang = token
        encoded_hi = tokenizer(sentence, return_tensors="pt")
        generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=id)
        aug = tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True)[0]
        if not clean(aug) == clean(org_text):
            if not any(p in aug for p in string.ascii_letters):
                return [aug]
    return []


def aug_m2m(model_name, text):
    """
    This is the display function for the m2m model where the load_m2m() and m2m() 
    functions are called. The function also calculates the time it takes to load 
    the model and augment the sentence and display it to the user as well. 

    Input Parameters
    ================
    model_name => HuggingFace link of the model (typically like this: aubmindlab/bert-large-arabertv2).
    text => Sentence or a list of sentences (typically user inputed sentence).

    Return Parameters
    =================
    all_sentences => Returns a list of all the augmented sentences.
    """

    loading_state_m2m = st.text("Loading M2M...")
    tic = time.perf_counter()
    model, tokenizer, token = load_m2m(model_name)
    toc = time.perf_counter()
    loading_state_m2m.text("Loading M2M done ✅: " +
                           str(round(toc-tic, 3)) + " seconds")
    augment_state_m2m = st.text("Augmenting with M2M...")
    tic = time.perf_counter()
    if isinstance(text, str):
        ret = m2m(model, tokenizer, token, text)
        toc = time.perf_counter()
        augment_state_m2m.text(
            "Augmenting with AraBERT done ✅: " + str(round(toc-tic, 3)) + " seconds")
        return ret
    else:
        all_sentences = []
        for sentence in text:
            sentence = sentence.strip()
            all_sentences.append(
                [sentence, m2m(model, tokenizer, token, sentence)])
        toc = time.perf_counter()
        augment_state_m2m.text(
            "Augmenting with AraBERT done ✅: " + str(round(toc-tic, 3)) + " seconds")
        return all_sentences

### ------------------------ End of Text-to-Text --------------------------- ###


### -------------------- Random Sentence Generator ------------------------- ###


def random_sentence(file_name: str):
    """
    Function to choose sentences with less than 15 words and
    choose a random sentence from the generated list and send 
    it to the frontend.

    The function also removes stray sentences with only a quotation
    mark.

    Input Parameters
    ================
    file_name (str) => Name of the file.

    Return Parameters
    =================
    selected_sentence => A randomly choosen sentence to be sent to the frontend.
    """

    sentences = []
    temp_list = []  # temporary list (hold temp values)
    res = []        # temporary list (hold temp values)

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

    # delete sentences with only quotation marks in the sentences list
    for sent in sentences:
        if sent == '"':
            sentences.remove(sent)

    selected_sentence = choice(sentences)
    return selected_sentence

### ------------------- End of Random Sentence Generator ------------------- ###


### ------------------------- Farasa API ----------------------------------- ###


def pos(text):
    """
    Function calls the Farasa API and get the Parts of Speech Tagger to read the sentence
    and split it into the different parts of the sentence and returns a dictionary.

    Input Parameters
    ================
    text => Sentence to be parsed with the Farasa API

    Return Parameters
    =================
    pos_dict => Dictionary of the parts of speech
    """

    url = 'https://farasa.qcri.org/webapi/pos/'
    api_key = "KMxvdPGsKHXQAbRXGL"
    payload = {'text': text, 'api_key': api_key}
    data = requests.post(url, data=payload)
    result = json.loads(data.text)
    text = text.split()
    pos_dict = {}
    for n in range(len(result["text"])):
        i = result["text"][n]
        if "+" == i['surface'][0]:
            word = "".join(s.strip()
                           for s in result["text"][n-1]['surface'].split("+"))
            word = word + i['surface'].replace("+", "").strip()
            if word in text:
                pos_dict[word] = result["text"][n-1]['POS']
        if "+" == i['surface'][-1]:
            word = "".join(s.strip()
                           for s in result["text"][n+1]['surface'].split("+"))
            word = i['surface'].replace("+", "").strip() + word
            if word in text:
                pos_dict[word] = result["text"][n+1]['POS']
        else:
            word = "".join(s.strip() for s in i['surface'].split("+"))
            if word in text:
                pos_dict[word] = i['POS']
    return pos_dict


def farasa_pos_output(text):
    """
    Function to use the Farasa API to read only the NOUNS, VERBS and ADJECTIVES in
    a sentence so it can be processed by the augmentation functions.

    Input Parameters
    ================
    text => Sentence to be processed by the Farasa API

    Return Parameters
    =================
    ret => Processed sentence where the NOUNS, VERBS and ADJECTIVES are marked
    """

    url = 'https://farasa.qcri.org/webapi/pos/'
    api_key = "KMxvdPGsKHXQAbRXGL"
    payload = {'text': text, 'api_key': api_key}
    data = requests.post(url, data=payload)
    result = json.loads(data.text)
    ret = ""
    for i in result['text']:
        if bool(set(i['POS'].split("+")) & set(['NOUN', 'V', 'ADJ'])):
            ret += "*" + i['POS'] + " "
        else:
            ret += i['POS'] + " "
    return ret.strip()

### ------------------------- End of Farasa API ---------------------------- ###


### ---------------------- Similarity Checker----- ------------------------- ###


@st.cache(allow_output_mutation=True)
def load_similarity_checker_model(model_name):
    """
    Load a Similarity calculator model from HuggingFace. The model is downloaded and cached 
    for futher use. Once cached it can be used multiple times without having to be downloaded
    everytime.

    Input Parameters
    ================
    model_name => Hugging Face link for the model (typically like this: sentence-transformers/bert-base-nli-mean-tokens).

    Return Parameters
    =================
    tokenizer => Tokenizer for the model.
    Model => Loaded model for the similarity calculator.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model


def similarity_checker(sentences, user_text_input):
    """
    Similarity calculator to calculate the cosine similarity between the original sentence and
    augmented sentence. It first encodes the augmented sentences and then using PyTorch and 
    Tensorflow it creates vectors. Then it calculates the cosine similarity and generates a list
    of the similarities.

    Input Parameters
    ================
    sentences => List of augmented sentences
    user_text_input => Sentence given by the user (Original sentence)

    Return Parameters
    =================
    cos_similarity => List of cosine similarities rounded up to 6 decimal places 
    average_similarity => Average similarity for the entire model (rounded to 6 decimal places)
    """

    tokenizer, model = load_similarity_checker_model(
        'sentence-transformers/bert-base-nli-mean-tokens')
    if len(sentences) > 0 and not any(x is None for x in sentences):
        tokens = {'input_ids': [], 'attention_mask': []}
        sentences.insert(0, user_text_input)
        for sentence in sentences:
            # tokenize sentence and append to dictionary lists
            new_tokens = tokenizer.encode_plus(sentence, max_length=128, truncation=True,
                                               padding='max_length', return_tensors='pt')
            tokens['input_ids'].append(new_tokens['input_ids'][0])
            tokens['attention_mask'].append(new_tokens['attention_mask'][0])

        # reformat list of tensors into single tensor
        tokens['input_ids'] = torch.stack(tokens['input_ids'])
        tokens['attention_mask'] = torch.stack(tokens['attention_mask'])

        outputs = model(**tokens)
        embeddings = outputs.last_hidden_state
        attention_mask = tokens['attention_mask']
        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()

        masked_embeddings = embeddings * mask
        summed = torch.sum(masked_embeddings, 1)
        summed_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / summed_mask

        # Convert from PyTorch tensor to numpy array
        mean_pooled = mean_pooled.detach().numpy()

        # Calculate cosine similarity
        cos_similarity = cosine_similarity([mean_pooled[0]], mean_pooled[1:])

        # Calculate average of similarities
        if len(sentences) > 1:
            average_similarity = mean(cos_similarity[0])
        else:
            average_similarity = 0

        return (np.around(cos_similarity[0], decimals=6), average_similarity)


def display_similarity_table(sentences_list, similarity_list, model_name):
    """
    Function to display the similarity table using streamlit. The function checks if there
    are sentences in the list and then prints a pandas DataFrame of the sentences and their
    coressponding similarities. The function also styles the similarities in a range of greens
    to show the highest and lowest similarities in the table; indicated by white(lowest) and dark
    green (highest).

    Input Parameters
    ================
    sentences_list => List of augmented sentences.
    similarity_list => List of the cosine similarities 
    """

    if len(sentences_list) > 0 and not any(x is None for x in sentences_list):
        data = list(zip(sentences_list, similarity_list))
        df = pd.DataFrame(data, columns=['Sentences', 'Similarity Score'])
        csv_file = convert_df_to_csv(df)
        st.download_button(
            label=f"Download {model_name} results as a CSV",
            data=csv_file,
            file_name=f'{model_name}-output.csv',
            mime='text/csv',
        )
        st.table(df[1:].style.background_gradient(cmap='Greens'))

### -------------------- End of Similarity Checker ------------------------- ###
