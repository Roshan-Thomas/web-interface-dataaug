import streamlit as st 
from model import aug_bert, aug_w2v, back_translate, random_sentence, spl, aug_m2m, aug_GPT
from model import load_bert, load_GPT, load_m2m, load_w2v, models_data, farasa_pos_output
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

  We are using four methods to do data augmentation on Arabic text: 
      
      * AraBERT (Machine Learning Model)
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

    ## Read the models.json to see which all models to be run. Change the flags to run only certain models. (1 = ON; 0 = OFF)
    data = models_data('./data/models.json')

    ## ------------------------------------- araBERT -------------------------------------- ##
    if data['arabert']:
      bert_container = st.container()
      with bert_container:
        st.subheader("AraBERT Data Augmentation")
        st.markdown("""
                    AraBERT is an Arabic pre-trained language model based on Google's 
                    BERT architecture. BERT is a fully connected deep neural network 
                    trained to predict two main things: a masked word in a sentence 
                    and the probability that the two sentences flow with each other. 
                    We give BERT a sentence with a masked word and using the context 
                    of the sentence, BERT predicts the masked word.

                    The outputs which can be seen shows the highlighted words (in green) 
                    that are changed by the model. 
                    """
                  )

        sentences_bert = aug_bert('aubmindlab/bert-large-arabertv2', st.session_state['user_input'])

        output_bert = ""
        for sent in sentences_bert:
          rep, fhalf, shalf = spl(sent)
          output_bert += f"""<p>
                          <span style="color:#ffffff">{fhalf}</span>
                          <span style="color:#7CFC00">{rep}</span> 
                          <span style="color:#ffffff">{shalf}</span>
                          </p> """

        with st.expander("Open to see AraBERT results"):
          st.markdown(output_bert, unsafe_allow_html=True)
    
    ## ------------------------------------- araGPT2 -------------------------------------- ##

    if data['aragpt2']:
      gpt2_container = st.container()
      with gpt2_container:
            st.subheader("AraGPT2 Data Augmentation")
            st.markdown(""" 
                        AraGPT2 is a pre-trained transformer for the Arabic Language 
                        Generation. The model successfully uses synthetic news 
                        generation and zero-shot question answering and has a 98% 
                        accuracy in detecting model-generated text. They are publicly 
                        available, and you can read the paper 
                        [here](https://arxiv.org/abs/2012.15520).
                        """)
            sentences_gpt = aug_GPT('aubmindlab/aragpt2-medium', st.session_state['user_input'])

            output_gpt = ""
            for sent in sentences_gpt:
              rep, fhalf, shalf = spl(sent)
              output_gpt += f"""<p>
                          <span style="color:#ffffff">{fhalf}</span>
                          <span style="color:#D22B2B">{rep}</span> 
                          <span style="color:#ffffff">{shalf}</span>
                          </p> """

            with st.expander("Open to see AraGPT2 results"):
              st.markdown(output_gpt, unsafe_allow_html=True)

    ## ------------------------------------- AraVec -------------------------------------- ##

    if data['aravec']:
      w2v_container = st.container()
      with w2v_container:
        st.subheader("AraVec (W2V) Data Augmentation")
        st.markdown(""" 
                    AraVec (W2V) is a pre-trained distributed word representation 
                    (word embedding) open-source project aiming to provide the Arabic 
                    NLP research community with free-to-use and powerful word 
                    embedding models. The recent version of AraVec provides 16-word 
                    embedding models built on top of two different Arabic content 
                    domains; Tweets and Wikipedia Arabic articles. This app uses the 
                    Twitter-trained model to augment the text. You can read more 
                    about it in this paper 
                    [here](https://www.researchgate.net/publication/319880027_AraVec_A_set_of_Arabic_Word_Embedding_Models_for_use_in_Arabic_NLP).

                    The outputs which can be seen shows the highlighted words (in yellow) 
                    that are changed by the model.
                    """)
        sentences_w2v = aug_w2v('./data/full_grams_cbow_100_twitter.mdl', st.session_state['user_input'])
        
        output_w2v = ""
        for sent in sentences_w2v:
          rep, fhalf, shalf = spl(sent)
          output_w2v += f"""<p>
                          <span style="color:#ffffff">{fhalf}</span>
                          <span style="color:#FFBF00">{rep}</span> 
                          <span style="color:#ffffff">{shalf}</span>
                        </p> """

        with st.expander("Open to see W2V results"):
          st.markdown(output_w2v, unsafe_allow_html=True)

    ## ------------------------------------- Back- Translation -------------------------------------- ##

    if data['back-translation']:
      back_translation_container = st.container()
      with back_translation_container:
        st.markdown("### Back Translation Augmentation")
        available_languages = ['ar-en', 'ar-fr', 'ar-tr', 'ar-ru', 'ar-pl', 'ar-it', 'ar-es', 'ar-el', 'ar-de', 'ar-he']
        back_translated_sentences = []
        st.markdown(
            """
            The languages we are back translating from are: *English (en), French (fr), 
            Turkish (tr), Russian (ru), Polish (pl), Italian (it), Spanish (es), 
            Greek (el), German (de), and Hebrew (he)*.
            """
        )

        back_translated_sentences = back_translate(available_languages, st.session_state['user_input'])
        with st.expander("Open to see Back Translation results"):
            st.write(back_translated_sentences)

    ## ------------------------------------- Text-to-Text -------------------------------------- ##

    if data['m2m']:
      text_to_text_container = st.container()
      with text_to_text_container:
        st.markdown("### Text-to-Text Augmentation")
        st.markdown("""
                    MBART is a sequence-to-sequence denoising auto-encoder 
                    pretrained on large-scale monolingual corpora in many languages 
                    using the BART objective. mBART is one of the first methods 
                    for pretraining a complete sequence-to-sequence model by 
                    denoising full texts in multiple languages. At the same time, 
                    previous approaches have focused only on the encoder, decoder, 
                    or reconstructing parts of the text. You can read more about it 
                    from this paper [here](https://arxiv.org/abs/2001.08210).
                    """)
        sentences_m2m = aug_m2m('facebook/mbart-large-50-many-to-many-mmt', st.session_state['user_input'])

        output_m2m = ""

        for sent in sentences_m2m:
          output_m2m = f""" <p>
                              <span style="color:#ffffff">{sent}</span>
                              </p> """
        with st.expander("Open to see Text-to-Text Results"):
          st.markdown(output_m2m, unsafe_allow_html=True)

## ---------------------------------------- End of Test the App -------------------------------------------- ##


## ---------------------------------------------- Citations ------------------------------------------------ ##

# st.write("-------------------------------------------------")
# citations()

## ------------------------------------------ End of Citations --------------------------------------------- ##
