import streamlit as st 
from model import aug_bert, aug_w2v, back_translate, random_sentence, spl, aug_m2m, aug_GPT
from model import load_bert, load_GPT, load_m2m, load_w2v
from citations import citations

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
    created synthetic data from existing data. Read more [here](https://en.wikipedia.org/wiki/Data_augmentation).

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
  # test_text = "RT @USER: رحمك الله يا صدام يا بطل ومقدام. URL	NOT_OFF	NOT_HS" # text to be used for testing purposes only

  text_input_container = st.empty()
  user_text_input = text_input_container.text_input("Enter your text here (AR):", "رحمك الله يا صدام يا بطل ومقدام") # TODO: make the default text to placeholder
  random_sentence_generator = st.checkbox('Use a Random Sentence (AR)?')
  if random_sentence_generator:
    user_text_input = random_sentence('./data/WikiNewsTruth.txt')
    text_input_container.empty()
    st.info(user_text_input)
  submit_button = st.button(label='Submit')

  if submit_button:
    bert_container = st.container()
    w2v_container = st.container()
    gpt2_container = st.container()
    text_to_text_container = st.container()
    back_translation_container = st.container()

    with bert_container:
      st.subheader("AraBERT Data Augmentation")
      sentences_bert = aug_bert('aubmindlab/bert-large-arabertv2', user_text_input)

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
    
    with gpt2_container:
          st.subheader("AraGPT2 Data Augmentation")
          sentences_gpt = aug_GPT('aubmindlab/aragpt2-medium', user_text_input)

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

    with w2v_container:
      st.subheader("W2V Data Augmentation")
      sentences_w2v = aug_w2v('./data/full_grams_cbow_100_twitter.mdl', user_text_input)
      
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

    with text_to_text_container:
      st.markdown("### Text-to-Text Augmentation")
      sentences_m2m = aug_m2m('facebook/mbart-large-50-many-to-many-mmt',user_text_input)

      output_m2m = ""

      for sent in sentences_m2m:
        output_m2m = f""" <p>
                            <span style="color:#ffffff">{sent}</span>
                            </p> """
      with st.expander("Open to see Text-to-Text Results"):
        st.markdown(output_m2m, unsafe_allow_html=True)

st.write("-------------------------------------------------")

# Citations for Models and References used
citations()