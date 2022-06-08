import streamlit as st 
from model import aug_bert, aug_GPT, aug_w2v

st.title("Data augmentation with AraGPT2, AraBERT, and W2V")
st.markdown(
    """
    Welcome to our data augmentation web interface. The app takes approximately 
    _one and a half minute (1.5)_ to load our machine learning models and then 
    augment the sentence. So please hold on as we process your sentence. 
    
    
    Why not get a cup of coffee while its processing? 😊
    """
)

user_text_input = st.text_input("Enter Text here (AR):")
# test_text = "RT @USER: رحمك الله يا صدام يا بطل ومقدام. URL	NOT_OFF	NOT_HS" # text to be used for testing purposes only

if user_text_input:
  bert_container = st.container()
  gpt2_container = st.container()
  w2v_container = st.container()

  with bert_container:
    st.subheader("AraBERT Data Augmentation")
    sentences_bert = aug_bert('aubmindlab/bert-large-arabertv2', user_text_input)
    with st.expander("Open to see AraBERT results"):
      st.write(sentences_bert)

  with gpt2_container:
    st.subheader("AraGPT2 Data Augmentation")
    sentences_gpt = aug_GPT('aubmindlab/aragpt2-medium', user_text_input)
    with st.expander("Open to see AraGPT2 results"):
      st.write(sentences_gpt)

  with w2v_container:
    st.subheader("W2V Data Augmentation")
    sentences_w2v = aug_w2v('./data/full_grams_cbow_100_twitter.mdl', user_text_input)
    with st.expander("Open to see W2V results"):
      st.write(sentences_w2v)
    
