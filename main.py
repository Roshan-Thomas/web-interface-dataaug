import streamlit as st 
from model import (aug_bert, aug_w2v, double_back_translate, 
                  random_sentence, spl, aug_m2m, aug_GPT,
                  load_bert, load_GPT, load_m2m, load_w2v, 
                  models_data, farasa_pos_output, translate_user_text_input,
                  display_similarity_table, similarity_checker)
from citations import citations

## ----------------------------------------------- Page Config --------------------------------------------- ##

st.set_page_config(
     page_title="Data Augmentation",
     page_icon='📈'
 )

## Session states - Initialization
if 'user_input' not in st.session_state:
  st.session_state['user_input'] = 'وبذلك تشتد المنافسة بين فايبر وبرنامج سكايب الذي يقدم خدمات مماثلة'

## --------------------------------------------- End of Page Config ---------------------------------------- ##

## ------------------------------------------------- Introduction ------------------------------------------ ##


st.title("Data augmentation - Web Interface")
st.markdown(
  """
  Welcome to our data augmentation web interface.

  ### What is Data Augmentation?

  It is a set of techniques used in data analysis to increase the amount of 
  data by adding slightly modified copies of already existing data or newly
  created synthetic data from existing data. Read more [here](https://en.wikipedia.org/wiki/Data_augmentation).

  We are using thirteen machine learning models to do data augmentation on Arabic text: 
      
      * AraBERT (Machine Learning Model)
      * QARiB (Machine Learning Model)
      * XLM-RoBERTa (Machine Learning Model)
      * AraBART (Machine Learning Model)
      * CAMeLBERT-Mix NER (Machine Learning Model)
      * Arabic BERT (Large) (Machine Learning Model)
      * ARBERT (Machine Learning Model)
      * MARBERTv2 (Machine Learning Model)
      * AraELECTRA (Machine Learning Model)
      * AraGPT2 (Machine Learning Model)
      * W2V (Machine Learning Model)
      * Text-to-Text Augmentation
      * Back Translation
  """
)

## --------------------------------------- End of Introduction --------------------------------------------- ##


## ---------------------------------------- 'Test the App' ------------------------------------------------- ##

test_app_container = st.container()

with test_app_container:
  st.markdown("# Test out our app here :blush::")
  # test_text = "وبذلك تشتد المنافسة بين فايبر وبرنامج سكايب الذي يقدم خدمات مماثلة" # text to be used for testing purposes only

  text_input_container = st.empty()
  translated_input_container = st.empty()
  farasa_pos_container = st.empty()

  user_text_input = text_input_container.text_input("Enter your text here (AR):", 
                                                    placeholder="وبذلك تشتد المنافسة بين فايبر وبرنامج سكايب الذي يقدم خدمات مماثلة")

  
  random_sentence_generator = st.checkbox('Use a Random Sentence (AR)?')
  if random_sentence_generator:
    text_input_container.empty()
    user_text_input = random_sentence('./data/WikiNewsTruth.txt')
    st.session_state.user_input = user_text_input
    text_input_container.text_input("Enter your text here (AR):", value=user_text_input)
    st.markdown("""*Note: If you want to generate a new sentence, uncheck and recheck the 'Use a Random Sentence (AR)?' checkbox.*""")

  if user_text_input:
    # Farasa POS output
    farasa_pos_container.markdown(f"*Parts of Speech:* {farasa_pos_output(user_text_input)}")

    ## Translate the sentence from arabic to english for the user
    # translated_input_container.markdown(f"*Translated sentence (EN):* {translate_user_text_input(user_text_input)}")

    ## Read the models.json to see which all models to be run. Change the flags to run only certain models. (1 = ON; 0 = OFF)
    data = models_data('./data/models.json')
    model_text_data = models_data('./data/models_data.json')

    ## ---------------------------- aubmindlab/bert-large-arabertv2 ----------------------- ##
    if data['arabert']:
      bert_container = st.container()
      with bert_container:
        st.markdown(model_text_data["arabert"]["header"])
        st.markdown(model_text_data["arabert"]["text"])

        sentences_bert = aug_bert(model_text_data["arabert"]["url"], st.session_state['user_input'])
       
        similarity_list = similarity_checker(sentences_bert, user_text_input)
        with st.expander(model_text_data["arabert"]["results"]):
          display_similarity_table(sentences_bert, similarity_list)
          st.markdown(model_text_data["common"]["word-info-expander"])
          
    ## -------------------------- qarib/bert-base-qarib ----------------------------------- ##
    if data['qarib-bert']:
      qarib_bert_container = st.container()
      with qarib_bert_container:
        st.markdown(model_text_data["qarib-bert"]["header"])
        st.markdown(model_text_data["qarib-bert"]["text"])

        sentences_qarib_bert = aug_bert(model_text_data["qarib-bert"]["url"], st.session_state['user_input'])

        similarity_list = similarity_checker(sentences_qarib_bert, user_text_input)
        with st.expander(model_text_data["qarib-bert"]["results"]):
          display_similarity_table(sentences_qarib_bert, similarity_list)
          st.markdown(model_text_data["common"]["word-info-expander"])
      
    ## ----------------------------- xlm-roberta-base ------------------------------------- ##
    if data['xlm-roberta-bert']:
      xlm_bert_container = st.container()
      with xlm_bert_container:
        st.markdown(model_text_data["xlm-roberta-bert"]["header"])
        st.markdown(model_text_data["xlm-roberta-bert"]["text"])

        sentences_xlm_bert = aug_bert(model_text_data["xlm-roberta-bert"]["url"], st.session_state['user_input'])

        similarity_list = similarity_checker(sentences_xlm_bert, user_text_input)
        with st.expander(model_text_data["xlm-roberta-bert"]["results"]):
          display_similarity_table(sentences_xlm_bert, similarity_list)
          st.markdown(model_text_data["common"]["word-info-expander"])

    ## ----------------------------- moussaKam/AraBART ------------------------------------ ##
    if data['arabart']:
      arabart_bert_container = st.container()
      with arabart_bert_container:
        st.markdown(model_text_data["arabart"]["header"])
        st.markdown(model_text_data["arabart"]["text"])

        sentences_arabart_bert = aug_bert(model_text_data["arabart"]["url"], st.session_state['user_input'])

        similarity_list = similarity_checker(sentences_arabart_bert, user_text_input)
        with st.expander(model_text_data["arabart"]["results"]):
          display_similarity_table(sentences_arabart_bert, similarity_list)
          st.markdown(model_text_data["common"]["word-info-expander"])

    ## ---------------------- CAMeL-Lab/bert-base-arabic-camelbert-mix -------------------- ##
    if data['camelbert']:
      camelbert_bert_container = st.container()
      with camelbert_bert_container:
        st.markdown(model_text_data["camelbert"]["header"])
        st.markdown(model_text_data["camelbert"]["text"])

        sentences_camelbert_bert = aug_bert(model_text_data["camelbert"]["url"], st.session_state['user_input'])

        similarity_list = similarity_checker(sentences_camelbert_bert, user_text_input)
        with st.expander(model_text_data["camelbert"]["results"]):
          display_similarity_table(sentences_camelbert_bert, similarity_list)
          st.markdown(model_text_data["common"]["word-info-expander"])
        
    ## --------------------------- asafaya/bert-large-arabic ------------------------------ ##
    if data['bert-large-arabic']:
      large_arabic_bert_container = st.container()
      with large_arabic_bert_container:
        st.markdown(model_text_data["bert-large-arabic"]["header"])
        st.markdown(model_text_data["bert-large-arabic"]["text"])

        sentences_large_arabic_bert = aug_bert(model_text_data["bert-large-arabic"]["url"], st.session_state['user_input'])

        similarity_list = similarity_checker(sentences_large_arabic_bert, user_text_input)
        with st.expander(model_text_data["bert-large-arabic"]["results"]):
          display_similarity_table(sentences_large_arabic_bert, similarity_list)
          st.markdown(model_text_data["common"]["word-info-expander"])
        
    ## --------------------------------- UBC-NLP/ARBERT ----------------------------------- ##
    if data['ubc-arbert']:
      ubc_arbert_bert_container = st.container()
      with ubc_arbert_bert_container:
        st.markdown(model_text_data["ubc-arbert"]["header"])
        st.markdown(model_text_data["ubc-arbert"]["text"])

        sentences_ubc_arbert_bert = aug_bert(model_text_data["ubc-arbert"]["url"], st.session_state['user_input'])

        similarity_list = similarity_checker(sentences_ubc_arbert_bert, user_text_input)
        with st.expander(model_text_data["ubc-arbert"]["results"]):
          display_similarity_table(sentences_ubc_arbert_bert, similarity_list)
          st.markdown(model_text_data["common"]["word-info-expander"])

    ## --------------------------------- UBC-NLP/MARBERTv2 -------------------------------- ##
    if data['ubc-marbertv2']:
      ubc_marbertv2_bert_container = st.container()
      with ubc_marbertv2_bert_container:
        st.markdown(model_text_data["ubc-marbertv2"]["header"])
        st.markdown(model_text_data["ubc-marbertv2"]["text"])

        sentences_ubc_marbertv2_bert = aug_bert(model_text_data["ubc-marbertv2"]["url"], st.session_state['user_input'])

        similarity_list = similarity_checker(sentences_ubc_marbertv2_bert, user_text_input)
        with st.expander(model_text_data["ubc-marbertv2"]["results"]):
          display_similarity_table(sentences_ubc_marbertv2_bert, similarity_list)
          st.markdown(model_text_data["common"]["word-info-expander"])
         
    ## ------------------------ aubmindlab/araelectra-base-generator ---------------------- ##
    if data['araelectra']:
      araelectra_bert_container = st.container()
      with araelectra_bert_container:
        st.markdown(model_text_data["araelectra"]["header"])
        st.markdown(model_text_data["araelectra"]["text"])

        sentences_araelectra_bert = aug_bert(model_text_data["araelectra"]["url"], st.session_state['user_input'])

        similarity_list = similarity_checker(sentences_araelectra_bert, user_text_input)
        with st.expander(model_text_data["araelectra"]["results"]):
          display_similarity_table(sentences_araelectra_bert, similarity_list)
          st.markdown(model_text_data["common"]["word-info-expander"])

    ## ------------------------------------- araGPT2 -------------------------------------- ##
    if data['aragpt2']:
      gpt2_container = st.container()
      with gpt2_container:
            st.markdown(model_text_data["aragpt2"]["header"])
            st.markdown(model_text_data["aragpt2"]["text"])
            sentences_gpt = aug_GPT(model_text_data["aragpt2"]["url"], st.session_state['user_input'])

            similarity_list = similarity_checker(sentences_gpt, user_text_input)
            with st.expander(model_text_data["aragpt2"]["results"]):
              display_similarity_table(sentences_gpt, similarity_list)
              st.markdown(model_text_data["common"]["word-info-expander"])

    ## ------------------------------------- AraVec -------------------------------------- ##
    if data['aravec']:
      w2v_container = st.container()
      with w2v_container:
        st.markdown(model_text_data["aravec"]["header"])
        st.markdown(model_text_data["aravec"]["text"])
        sentences_w2v = aug_w2v('./data/full_grams_cbow_100_twitter.mdl', st.session_state['user_input'])

        similarity_list = similarity_checker(sentences_w2v, user_text_input)
        with st.expander(model_text_data["aravec"]["results"]):
          display_similarity_table(sentences_w2v, similarity_list)
          st.markdown(model_text_data["common"]["word-info-expander"])

    ## ------------------------------------- Back- Translation -------------------------------------- ##
    if data['double-back-translation']:
      back_translation_container = st.container()
      with back_translation_container:
        st.markdown(model_text_data["double-back-translation"]["header"])
        available_languages = ['ar-en', 'ar-fr', 'ar-tr', 'ar-ru', 'ar-pl', 'ar-it', 'ar-es', 'ar-el', 'ar-de', 'ar-he']
        back_translated_sentences = []
        st.markdown(model_text_data["double-back-translation"]["text"])
        st.markdown(model_text_data["double-back-translation"]["text-2"])

        back_translated_sentences = double_back_translate(available_languages, st.session_state['user_input'])
        similarity_list = similarity_checker(back_translated_sentences, user_text_input)
        with st.expander(model_text_data["double-back-translation"]["results"]):
          display_similarity_table(back_translated_sentences, similarity_list)

    ## ------------------------------------- Text-to-Text -------------------------------------- ##
    if data['m2m']:
      text_to_text_container = st.container()
      with text_to_text_container:
        st.markdown(model_text_data["m2m"]["header"])
        st.markdown(model_text_data["m2m"]["text"])
        sentences_m2m = aug_m2m(model_text_data["m2m"]["url"], st.session_state['user_input'])

        similarity_list = similarity_checker(sentences_m2m, user_text_input)
        with st.expander(model_text_data["m2m"]["results"]):
          display_similarity_table(sentences_m2m, similarity_list)
          st.markdown(model_text_data["common"]["word-info-expander"])

## ---------------------------------------- End of 'Test the App' ------------------------------------------ ##


## ---------------------------------------------- Citations ------------------------------------------------ ##

# st.write("-------------------------------------------------")
# citations()

## ------------------------------------------ End of Citations --------------------------------------------- ##
