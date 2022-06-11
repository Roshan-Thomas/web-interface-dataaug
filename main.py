import streamlit as st 
from model import aug_bert, aug_GPT, aug_w2v, back_translate

st.set_page_config(
     page_title="Data Augmentation",
     page_icon='📈'
 )

st.title("Data augmentation - Web Interface")
st.markdown(
    """
    Welcome to our data augmentation web interface.

    ### What is Data Augmentation?

    It is a set of techniques used in data analysis to increase the amount of 
    data by adding slightly modified copies of already existing data or newly
    created synthetic data from existing data. Read more [here.](https://en.wikipedia.org/wiki/Data_augmentation)

    We are using four methods to do data augmentation on Arabic text: 
        
        * AraBERT (Machine Learning Model)
        * AraGPT2 (Machine Learning Model)
        * W2V (Machine Learning Model)
        * Back Translation
    """
)

test_app_container = st.container()

with test_app_container:
  st.markdown("# Test out our app here :blush::")
  user_text_input = st.text_input("Enter your text here (AR):")
  # test_text = "RT @USER: رحمك الله يا صدام يا بطل ومقدام. URL	NOT_OFF	NOT_HS" # text to be used for testing purposes only

  if user_text_input:
    bert_container = st.container()
    gpt2_container = st.container()
    w2v_container = st.container()
    back_translation_container = st.container()

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

      back_translated_sentences = back_translate(available_languages, user_text_input)
      with st.expander("Open to see Back Translation results"):
          st.write(back_translated_sentences)

st.write("-------------------------------------------------")

st.header("Citations")
with st.expander("Expand to see the citations"):
  aragpt2_citation = '''
    @inproceedings{antoun-etal-2021-aragpt2,
      title = "{A}ra{GPT}2: Pre-Trained Transformer for {A}rabic Language Generation",
      author = "Antoun, Wissam  and
        Baly, Fady  and
        Hajj, Hazem",
      booktitle = "Proceedings of the Sixth Arabic Natural Language Processing Workshop",
      month = apr,
      year = "2021",
      address = "Kyiv, Ukraine (Virtual)",
      publisher = "Association for Computational Linguistics",
      url = "https://www.aclweb.org/anthology/2021.wanlp-1.21",
      pages = "196--207",
    }
  '''
  arabert_citation = '''
    @inproceedings{antoun2020arabert,
      title={AraBERT: Transformer-based Model for Arabic Language Understanding},
      author={Antoun, Wissam and Baly, Fady and Hajj, Hazem},
      booktitle={LREC 2020 Workshop Language Resources and Evaluation Conference 11--16 May 2020},
      pages={9}
    }
  '''
  w2v_citation = '''
    @article{article,
      author = {Mohammad, Abu Bakr and Eissa, Kareem and El-Beltagy, Samhaa},
      year = {2017},
      month = {11},
      pages = {256-265},
      title = {AraVec: A set of Arabic Word Embedding Models for use in Arabic NLP},
      volume = {117},
      journal = {Procedia Computer Science},
      doi = {10.1016/j.procs.2017.10.117}
    }
  '''

  bt_citation = '''
    @InProceedings{TiedemannThottingal:EAMT2020,
      author = {J{\"o}rg Tiedemann and Santhosh Thottingal},
      title = {{OPUS-MT} — {B}uilding open translation services for the {W}orld},
      booktitle = {Proceedings of the 22nd Annual Conferenec of the European Association for Machine Translation (EAMT)},
      year = {2020},
      address = {Lisbon, Portugal}
    }
  '''
  st.code(aragpt2_citation)
  st.code(arabert_citation)
  st.code(w2v_citation)
  st.code(bt_citation)