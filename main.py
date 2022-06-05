import streamlit as st 
from model import text_generation, fill_mask

# for now only doing arabert and aragpt2 models 
# TODO: aravec model

st.title("Web Interface with araGPT2")

# TODO: w2v function
# TODO: aravec models

user_text_input = st.text_input("Enter Text here (AR):")

if user_text_input:
    # Tell the user that the data is loading
    loading_state = st.text("Loading data...")

    # GPT2-based text generation models
    aragpt2 = text_generation('aubmindlab/aragpt2-medium',user_text_input) ## use the mega model

    # BERT-based fill mask models (4)
    # arabert = fill_mask('aubmindlab/bert-base-arabert', user_text_input)
    # arabertv2 = fill_mask('aubmindlab/bert-large-arabertv2',user_text_input)
    # arabertv02 = fill_mask('aubmindlab/bert-large-arabertv02',user_text_input)
    # arabertv01 = fill_mask('aubmindlab/bert-base-arabertv01',user_text_input)

    # Inform the user the data is loaded
    loading_state.text("Data Loaded ✅")

    output = "org: " + text + "\n"
    st.write(output)

    gpt = list(set(aragpt2))
    for aug in gpt:
        st.write(f"GPT2-based: {aug} \n")

    # bert = list(set(arabert) | set(arabertv2) | set(arabertv02) | set(arabertv01))
    # for aug in bert:
    #     st.write(f"BERT-based: {aug}\n ")