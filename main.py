import streamlit as st 
from model import aug_bert, aug_w2v, back_translate, random_sentence, spl, aug_m2m, aug_GPT
from model import load_bert, load_GPT, load_m2m, load_w2v, models_data, farasa_pos_output, translate_user_text_input
from citations import citations

## ----------------------------------------------- Page Config --------------------------------------------- ##

st.set_page_config(
     page_title="Data Augmentation",
     page_icon='📈'
 )

## Session states - Initialization
if 'user_input' not in st.session_state:
  st.session_state['user_input'] = 'وبذلك تشتد المنافسة بين فايبر وبرنامج سكايب الذي يقدم خدمات مماثلة'
if 'farasa_output' not in st.session_state:
  st.session_state['farasa_output'] = 'وبذلك تشتد المنافسة بين فايبر وبرنامج سكايب الذي يقدم خدمات مماثلة'

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


## ---------------------------------------- Test the App --------------------------------------------------- ##

test_app_container = st.container()

with test_app_container:
  st.markdown("# Test out our app here :blush::")
  # test_text = "وبذلك تشتد المنافسة بين فايبر وبرنامج سكايب الذي يقدم خدمات مماثلة" # text to be used for testing purposes only

  text_input_container = st.empty()
  user_text_input = text_input_container.text_input("Enter your text here (AR):", 
                                                    placeholder="وبذلك تشتد المنافسة بين فايبر وبرنامج سكايب الذي يقدم خدمات مماثلة")

  st.session_state.farasa_output = st.text(farasa_pos_output(user_text_input))

  random_sentence_generator = st.checkbox('Use a Random Sentence (AR)?')
  if random_sentence_generator:
    text_input_container.empty()
    user_text_input = random_sentence('./data/WikiNewsTruth.txt')
    st.session_state.user_input = user_text_input
    text_input_container.text_input("Enter your text here (AR):", value=user_text_input)
    st.markdown("""*Note: If you want to generate a new sentence, uncheck and recheck the 'Use a Random Sentence (AR)?' checkbox.*""")
    st.session_state.farasa_output.text(farasa_pos_output(user_text_input))

  if user_text_input:
    ## Translate the sentence from arabic to english for the user
    translate_user_text_input(user_text_input)

    ## Read the models.json to see which all models to be run. Change the flags to run only certain models. (1 = ON; 0 = OFF)
    data = models_data('./data/models.json')
    model_text_data = models_data('./data/models_data.json')

    ## ---------------------------- aubmindlab/bert-large-arabertv2 ----------------------- ##
    if data['arabert']:
      bert_container = st.container()
      with bert_container:
        st.markdown(model_text_data["arabert"]["header"])
        st.markdown(model_text_data["arabert"]["text"])
        st.markdown(model_text_data["common"]["bert-output"], unsafe_allow_html=True)

        sentences_bert = aug_bert(model_text_data["arabert"]["url"], st.session_state['user_input'])

        output_bert = ""
        for sent in sentences_bert:
          rep, fhalf, shalf = spl(sent)
          output_bert += f"""<p>
                          <span style="color:#ffffff">{fhalf}</span>
                          <span style="color:#7CFC00">{rep}</span> 
                          <span style="color:#ffffff">{shalf}</span>
                          </p> """

        with st.expander(model_text_data["arabert"]["results"]):
          st.markdown(output_bert, unsafe_allow_html=True)
    
    ## -------------------------- qarib/bert-base-qarib ----------------------------------- ##
    if data['qarib-bert']:
      qarib_bert_container = st.container()
      with qarib_bert_container:
        st.markdown(model_text_data["qarib-bert"]["header"])
        st.markdown(model_text_data["qarib-bert"]["text"])
        st.markdown(model_text_data["common"]["bert-output"], unsafe_allow_html=True)

        sentences_qarib_bert = aug_bert(model_text_data["qarib-bert"]["url"], st.session_state['user_input'])

        output_qarib_bert = ""
        for sent in sentences_qarib_bert:
          rep, fhalf, shalf = spl(sent)
          output_qarib_bert += f"""<p>
                          <span style="color:#ffffff">{fhalf}</span>
                          <span style="color:#7CFC00">{rep}</span> 
                          <span style="color:#ffffff">{shalf}</span>
                          </p> """

        with st.expander(model_text_data["qarib-bert"]["results"]):
          st.markdown(output_qarib_bert, unsafe_allow_html=True)

    ## ----------------------------- xlm-roberta-base ------------------------------------- ##
    if data['xlm-roberta-bert']:
      xlm_bert_container = st.container()
      with xlm_bert_container:
        st.markdown(model_text_data["xlm-roberta-bert"]["header"])
        st.markdown(model_text_data["xlm-roberta-bert"]["text"])
        st.markdown(model_text_data["common"]["bert-output"], unsafe_allow_html=True)

        sentences_xlm_bert = aug_bert(model_text_data["xlm-roberta-bert"]["url"], st.session_state['user_input'])

        output_xlm_bert = ""
        for sent in sentences_xlm_bert:
          rep, fhalf, shalf = spl(sent)
          output_xlm_bert += f"""<p>
                          <span style="color:#ffffff">{fhalf}</span>
                          <span style="color:#7CFC00">{rep}</span> 
                          <span style="color:#ffffff">{shalf}</span>
                          </p> """

        with st.expander(model_text_data["xlm-roberta-bert"]["results"]):
          st.markdown(output_xlm_bert, unsafe_allow_html=True)

    ## ----------------------------- moussaKam/AraBART ------------------------------------ ##
    if data['arabart']:
      arabart_bert_container = st.container()
      with arabart_bert_container:
        st.markdown(model_text_data["arabart"]["header"])
        st.markdown(model_text_data["arabart"]["text"])
        st.markdown(model_text_data["common"]["bert-output"], unsafe_allow_html=True)

        sentences_arabart_bert = aug_bert(model_text_data["arabart"]["url"], st.session_state['user_input'])

        output_arabart_bert = ""
        for sent in sentences_arabart_bert:
          rep, fhalf, shalf = spl(sent)
          output_arabart_bert += f"""<p>
                          <span style="color:#ffffff">{fhalf}</span>
                          <span style="color:#7CFC00">{rep}</span> 
                          <span style="color:#ffffff">{shalf}</span>
                          </p> """

        with st.expander(model_text_data["arabart"]["results"]):
          st.markdown(output_arabart_bert, unsafe_allow_html=True)


    ## ---------------------- CAMeL-Lab/bert-base-arabic-camelbert-mix -------------------- ##
    if data['camelbert']:
      camelbert_bert_container = st.container()
      with camelbert_bert_container:
        st.markdown(model_text_data["camelbert"]["header"])
        st.markdown(model_text_data["camelbert"]["text"])
        st.markdown(model_text_data["common"]["bert-output"], unsafe_allow_html=True)

        sentences_camelbert_bert = aug_bert(model_text_data["camelbert"]["url"], st.session_state['user_input'])

        output_camelbert_bert = ""
        for sent in sentences_camelbert_bert:
          rep, fhalf, shalf = spl(sent)
          output_camelbert_bert += f"""<p>
                          <span style="color:#ffffff">{fhalf}</span>
                          <span style="color:#7CFC00">{rep}</span> 
                          <span style="color:#ffffff">{shalf}</span>
                          </p> """

        with st.expander(model_text_data["camelbert"]["results"]):
          st.markdown(output_camelbert_bert, unsafe_allow_html=True)

    ## --------------------------- asafaya/bert-large-arabic ------------------------------ ##
    if data['bert-large-arabic']:
      large_arabic_bert_container = st.container()
      with large_arabic_bert_container:
        st.markdown(model_text_data["bert-large-arabic"]["header"])
        st.markdown(model_text_data["bert-large-arabic"]["text"])
        st.markdown(model_text_data["common"]["bert-output"], unsafe_allow_html=True)

        sentences_large_arabic_bert = aug_bert(model_text_data["bert-large-arabic"]["url"], st.session_state['user_input'])

        output_large_arabic_bert = ""
        for sent in sentences_large_arabic_bert:
          rep, fhalf, shalf = spl(sent)
          output_large_arabic_bert += f"""<p>
                          <span style="color:#ffffff">{fhalf}</span>
                          <span style="color:#7CFC00">{rep}</span> 
                          <span style="color:#ffffff">{shalf}</span>
                          </p> """

        with st.expander(model_text_data["bert-large-arabic"]["results"]):
          st.markdown(output_large_arabic_bert, unsafe_allow_html=True)

    ## --------------------------------- UBC-NLP/ARBERT ----------------------------------- ##
    if data['ubc-arbert']:
      ubc_arbert_bert_container = st.container()
      with ubc_arbert_bert_container:
        st.markdown(model_text_data["ubc-arbert"]["header"])
        st.markdown(model_text_data["ubc-arbert"]["text"])
        st.markdown(model_text_data["common"]["bert-output"], unsafe_allow_html=True)

        sentences_ubc_arbert_bert = aug_bert(model_text_data["ubc-arbert"]["url"], st.session_state['user_input'])

        output_ubc_arbert_bert = ""
        for sent in sentences_ubc_arbert_bert:
          rep, fhalf, shalf = spl(sent)
          output_ubc_arbert_bert += f"""<p>
                          <span style="color:#ffffff">{fhalf}</span>
                          <span style="color:#7CFC00">{rep}</span> 
                          <span style="color:#ffffff">{shalf}</span>
                          </p> """

        with st.expander(model_text_data["ubc-arbert"]["results"]):
          st.markdown(output_ubc_arbert_bert, unsafe_allow_html=True)

    ## --------------------------------- UBC-NLP/MARBERTv2 -------------------------------- ##
    if data['ubc-marbertv2']:
      ubc_marbertv2_bert_container = st.container()
      with ubc_marbertv2_bert_container:
        st.markdown(model_text_data["ubc-marbertv2"]["header"])
        st.markdown(model_text_data["ubc-marbertv2"]["text"])
        st.markdown(model_text_data["common"]["bert-output"], unsafe_allow_html=True)

        sentences_ubc_marbertv2_bert = aug_bert(model_text_data["ubc-marbertv2"]["url"], st.session_state['user_input'])

        output_ubc_marbertv2_bert = ""
        for sent in sentences_ubc_marbertv2_bert:
          rep, fhalf, shalf = spl(sent)
          output_ubc_marbertv2_bert += f"""<p>
                          <span style="color:#ffffff">{fhalf}</span>
                          <span style="color:#7CFC00">{rep}</span> 
                          <span style="color:#ffffff">{shalf}</span>
                          </p> """

        with st.expander(model_text_data["ubc-marbertv2"]["results"]):
          st.markdown(output_ubc_marbertv2_bert, unsafe_allow_html=True)

    ## ------------------------ aubmindlab/araelectra-base-generator ---------------------- ##
    if data['araelectra']:
      araelectra_bert_container = st.container()
      with araelectra_bert_container:
        st.markdown(model_text_data["araelectra"]["header"])
        st.markdown(model_text_data["araelectra"]["text"])
        st.markdown(model_text_data["common"]["bert-output"], unsafe_allow_html=True)

        sentences_araelectra_bert = aug_bert(model_text_data["araelectra"]["url"], st.session_state['user_input'])

        output_araelectra_bert = ""
        for sent in sentences_araelectra_bert:
          rep, fhalf, shalf = spl(sent)
          output_araelectra_bert += f"""<p>
                          <span style="color:#ffffff">{fhalf}</span>
                          <span style="color:#7CFC00">{rep}</span> 
                          <span style="color:#ffffff">{shalf}</span>
                          </p> """

        with st.expander(model_text_data["araelectra"]["results"]):
          st.markdown(output_araelectra_bert, unsafe_allow_html=True)

    ## ------------------------------------- araGPT2 -------------------------------------- ##
    if data['aragpt2']:
      gpt2_container = st.container()
      with gpt2_container:
            st.markdown(model_text_data["aragpt2"]["header"])
            st.markdown(model_text_data["aragpt2"]["text"])
            sentences_gpt = aug_GPT(model_text_data["aragpt2"]["url"], st.session_state['user_input'])

            output_gpt = ""
            for sent in sentences_gpt:
              rep, fhalf, shalf = spl(sent)
              output_gpt += f"""<p>
                          <span style="color:#ffffff">{fhalf}</span>
                          <span style="color:#D22B2B">{rep}</span> 
                          <span style="color:#ffffff">{shalf}</span>
                          </p> """

            with st.expander(model_text_data["aragpt2"]["results"]):
              st.markdown(output_gpt, unsafe_allow_html=True)

    ## ------------------------------------- AraVec -------------------------------------- ##
    if data['aravec']:
      w2v_container = st.container()
      with w2v_container:
        st.markdown(model_text_data["aravec"]["header"])
        st.markdown(model_text_data["aravec"]["text"])
        st.markdown(model_text_data["common"]["aravec-output"], unsafe_allow_html=True)
        sentences_w2v = aug_w2v('./data/full_grams_cbow_100_twitter.mdl', st.session_state['user_input'])
        
        output_w2v = ""
        for sent in sentences_w2v:
          rep, fhalf, shalf = spl(sent)
          output_w2v += f"""<p>
                          <span style="color:#ffffff">{fhalf}</span>
                          <span style="color:#FFBF00">{rep}</span> 
                          <span style="color:#ffffff">{shalf}</span>
                        </p> """

        with st.expander(model_text_data["aravec"]["results"]):
          st.markdown(output_w2v, unsafe_allow_html=True)

    ## ------------------------------------- Back- Translation -------------------------------------- ##

    if data['back-translation']:
      back_translation_container = st.container()
      with back_translation_container:
        st.markdown(model_text_data["back-translation"]["header"])
        available_languages = ['ar-en', 'ar-fr', 'ar-tr', 'ar-ru', 'ar-pl', 'ar-it', 'ar-es', 'ar-el', 'ar-de', 'ar-he']
        back_translated_sentences = []
        st.markdown(model_text_data["back-translation"]["text"])

        back_translated_sentences = back_translate(available_languages, st.session_state['user_input'])
        with st.expander(model_text_data["back-translation"]["results"]):
            st.write(back_translated_sentences)

    ## ------------------------------------- Text-to-Text -------------------------------------- ##

    if data['m2m']:
      text_to_text_container = st.container()
      with text_to_text_container:
        st.markdown(model_text_data["m2m"]["header"])
        st.markdown(model_text_data["m2m"]["text"])
        sentences_m2m = aug_m2m(model_text_data["m2m"]["url"], st.session_state['user_input'])
        sentences_m2m_2 = aug_m2m(model_text_data["m2m"]["url-2"], st.session_state['user_input'])

        output_m2m = ""
        output_m2m_2 = ""

        for sent in sentences_m2m:
          output_m2m = f""" <p>
                        <span style="color:#ffffff">MBART_Large: </span> 
                        <span style="color:#ffffff">{sent}</span>
                        </p> """
        
        for sent in sentences_m2m_2:
          output_m2m_2 = f"""<p>
                          <span style="color:#ffffff">M2M100: </span>
                          <span style="color:#ffffff">{sent}</span>
                          </p> """

        with st.expander(model_text_data["m2m"]["results"]):
          st.markdown(output_m2m, unsafe_allow_html=True)
          st.markdown(output_m2m_2, unsafe_allow_html=True)

## ---------------------------------------- End of Test the App -------------------------------------------- ##


## ---------------------------------------------- Citations ------------------------------------------------ ##

# st.write("-------------------------------------------------")
# citations()

## ------------------------------------------ End of Citations --------------------------------------------- ##
