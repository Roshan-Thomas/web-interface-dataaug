import streamlit as st
from model import (aug_bert, aug_w2v, double_back_translate, random_sentence, aug_m2m, aug_GPT,
                   farasa_pos_output, display_similarity_table, similarity_checker)
from helper import (translate_user_text_input, models_data,
                    get_df_data, download_all_outputs)

## ----------------------------------------------- Page Config --------------------------------------------- ##

st.set_page_config(
    page_title="Data Augmentation",
    page_icon='📈'
)

# Read the models.json to see which all models to be run. Change the flags to run only certain models. (1 = ON; 0 = OFF)
data = models_data('./data/models.json')

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

  We are using twelve machine learning models to do data augmentation on Arabic text: 
      
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
      * Word-to-Vector (W2V) Augmentation
      * Back Translation
  """
)

## --------------------------------------- End of Introduction --------------------------------------------- ##


## ----------------------------------------------- Sidebar ------------------------------------------------- ##

with st.sidebar:
    # Display choices of data augmentation techniques in the sidebar
    st.write("Choose the data augmentation techniques below 👇")

    col1, col2 = st.columns(2)

    with col1:
        data['arabert'] = st.checkbox('AraBERT', value=True)
        data['qarib_bert'] = st.checkbox('QARiB')
        data['xlm-roberta-bert'] = st.checkbox('XLM-RoBERTa', value=True)
        data['arabart'] = st.checkbox('AraBART')
        data['camelbert'] = st.checkbox('CAMeLBERT', value=True)
        data['bert-large-arabic'] = st.checkbox('Arabic BERT (Large)')
        data['ubc-arbert'] = st.checkbox('ARBERT', value=True)

    with col2:
        data['ubc-marbertv2'] = st.checkbox('MARBERTv2', value=True)
        data['araelectra'] = st.checkbox('AraELECTRA', value=True)
        data['aragpt2'] = st.checkbox('AraGPT2')
        data['aravec'] = st.checkbox('Word-to-Vector')
        data['double-back-translation'] = st.checkbox(
            'Double Back Translation', value=True)
        # data['m2m'] = st.checkbox('Text-to-Text')

## -------------------------------------------- End of Sidebar --------------------------------------------- ##

## ---------------------------------------- 'Test the App' ------------------------------------------------- ##

test_app_container = st.container()

with test_app_container:
    st.markdown("# Test out our app here :blush::")
    st.markdown("Write a sentence you want to augment below (in the text field) and choose the augmentation techniques in the sidebar.")
    # test_text = "وبذلك تشتد المنافسة بين فايبر وبرنامج سكايب الذي يقدم خدمات مماثلة" # text to be used for testing purposes only

    text_input_container = st.empty()
    translated_input_container = st.empty()
    farasa_pos_container = st.empty()

    user_text_input = text_input_container.text_input("Enter your text here (AR):",
                                                      placeholder="وبذلك تشتد المنافسة بين فايبر وبرنامج سكايب الذي يقدم خدمات مماثلة")

    random_sentence_container = st.empty()
    random_sentence_checkbox = random_sentence_container.checkbox(
        "Use a Random Sentence (AR)?")

    if random_sentence_checkbox:
        # Radio buttons to allow the user to choose MSA or Dialectal Arabic
        msa_or_dialectal_radio_button = random_sentence_container.radio(
            "Choose the type of Arabic sentence used for the random sentence:",
            ('Modern Standard Arabic (MSA)', 'Dialectal Arabic'), horizontal=True,
        )

        if msa_or_dialectal_radio_button == 'Modern Standard Arabic (MSA)':
            # Run the code below if MSA Arabic is choosen
            st.markdown("""<span style="color:#b0b3b8">*We are using a dataset from WikiNewsTruth from 2013 and 2014 to give a random news title with Modern Standard Arabic for augmentation.*</span>""",
                        unsafe_allow_html=True)
            random_sentence_generator = st.checkbox(
                'Use a Random MSA Sentence?')
            if random_sentence_generator:
                text_input_container.empty()
                user_text_input = random_sentence('./data/WikiNewsTruth.txt')
                text_input_container.text_input(
                    "Enter your text here (AR):", value=user_text_input)
                st.markdown("""
                    <span style="color:#b0b3b8">*Note: If you want to generate a new sentence, STOP the running, uncheck and recheck the 'Use a Random Sentence (AR)?' checkbox.*</span>""",
                            unsafe_allow_html=True
                            )
        else:
            # Run the code below if Dialectal Arabic is choosen
            st.markdown("""<span style="color:#b0b3b8">*We are using the ArSAS Training dataset (16k tweets) to generate some random tweets with Dialectal Arabic for augmentation.*</span>""",
                        unsafe_allow_html=True)
            random_sentence_generator = st.checkbox(
                'Use a Random Dialectal Arabic Sentence?')
            if random_sentence_generator:
                text_input_container.empty()
                user_text_input = random_sentence(
                    './data/ArSAS-train-clean.txt')
                text_input_container.text_input(
                    "Enter your text here (AR):", value=user_text_input)
                st.markdown("""
                    <span style="color:#b0b3b8">*Note: If you want to generate a new sentence, STOP the running, uncheck and recheck the 'Use a Random Sentence (AR)?' checkbox.*</span>""",
                            unsafe_allow_html=True
                            )

    if user_text_input:

        # Farasa 'Parts of Speech tagger' output
        try:
            farasa_pos_container.markdown(f"""*<span style="color:#AAFF00">Parts of Speech:</span>* {farasa_pos_output(user_text_input)}""",
                                          unsafe_allow_html=True)
        except:
            # 'Except' case when Farasa API is not functional (and not returning any output)
            st.error(
                "We are facing issues with the Farasa API. Please try again later.")
            st.stop()

        # Translate the sentence from arabic to english for the user
        translated_input_container.markdown(f"""*<span style="color:#AAFF00">Translated sentence (EN):</span>* {translate_user_text_input("Helsinki-NLP/opus-mt-ar-en", user_text_input)}""",
                                            unsafe_allow_html=True)

        st.sidebar.write("--------------------------")
        st.sidebar.markdown(f"""*<span style="color:#AAFF00">Original Sentence:</span>* <br /> {user_text_input}""",
                            unsafe_allow_html=True)

        model_text_data = models_data('./data/models_data.json')

        # List of all dataframes of augmented text (for export to csv)
        list_of_dataframes = []

        ## ---------------------------- aubmindlab/bert-large-arabertv2 ----------------------- ##
        if data['arabert']:
            bert_container = st.container()
            with bert_container:
                # Details of Arabert for the user
                st.markdown(model_text_data["arabert"]["header"])
                st.markdown(model_text_data["arabert"]["text"])

                # Augment sentences with Arabert
                sentences_bert = aug_bert(model_text_data["arabert"]["url"],
                                          user_text_input,
                                          model_text_data["arabert"]["name"]
                                          )

                # Generate List of similarity score for each augmented sentence and average similarity scores
                similarity_list, average_similarity = similarity_checker(
                    sentences_bert, user_text_input)
                list_of_dataframes.append(get_df_data(
                    sentences_bert, similarity_list))

                # Show Arabert results to the user
                with st.expander(model_text_data["arabert"]["results"]):
                    st.markdown(
                        f"Average Similarity: {average_similarity:.6f}")
                    display_similarity_table(
                        sentences_bert, similarity_list, model_text_data["arabert"]["name"])
                    st.markdown(
                        model_text_data["common"]["word-info-expander"])

        ## -------------------------- qarib/bert-base-qarib ----------------------------------- ##
        if data['qarib-bert']:
            qarib_bert_container = st.container()
            with qarib_bert_container:
                # Details of Qarib for the user
                st.markdown(model_text_data["qarib-bert"]["header"])
                st.markdown(model_text_data["qarib-bert"]["text"])

                # Augment sentences with Qarib
                sentences_qarib_bert = aug_bert(model_text_data["qarib-bert"]["url"],
                                                user_text_input,
                                                model_text_data["qarib-bert"]["name"]
                                                )

                # Generate List of similarity score for each augmented sentence and average similarity scores
                similarity_list, average_similarity = similarity_checker(
                    sentences_qarib_bert, user_text_input)
                list_of_dataframes.append(get_df_data(
                    sentences_qarib_bert, similarity_list))

                # Show Qarib results to the user
                with st.expander(model_text_data["qarib-bert"]["results"]):
                    st.markdown(
                        f"Average Similarity: {average_similarity:.6f}")
                    display_similarity_table(
                        sentences_qarib_bert, similarity_list, model_text_data["qarib-bert"]["name"])
                    st.markdown(
                        model_text_data["common"]["word-info-expander"])

        ## ----------------------------- xlm-roberta-base ------------------------------------- ##
        if data['xlm-roberta-bert']:
            xlm_bert_container = st.container()
            with xlm_bert_container:
                # Details of XLM-RoBERTa for the user
                st.markdown(model_text_data["xlm-roberta-bert"]["header"])
                st.markdown(model_text_data["xlm-roberta-bert"]["text"])

                # Augment sentences with XLM-RoBERTa
                sentences_xlm_bert = aug_bert(model_text_data["xlm-roberta-bert"]["url"],
                                              user_text_input,
                                              model_text_data["xlm-roberta-bert"]["name"]
                                              )

                # Generate List of similarity score for each augmented sentence and average similarity scores
                similarity_list, average_similarity = similarity_checker(
                    sentences_xlm_bert, user_text_input)
                list_of_dataframes.append(get_df_data(
                    sentences_xlm_bert, similarity_list))

                # Show results of XLM-RoBERTa to the user
                with st.expander(model_text_data["xlm-roberta-bert"]["results"]):
                    st.markdown(
                        f"Average Similarity: {average_similarity:.6f}")
                    display_similarity_table(
                        sentences_xlm_bert, similarity_list, model_text_data["xlm-roberta-bert"]["name"])
                    st.markdown(
                        model_text_data["common"]["word-info-expander"])

        ## ----------------------------- moussaKam/AraBART ------------------------------------ ##
        if data['arabart']:
            arabart_bert_container = st.container()
            with arabart_bert_container:
                # Details about Arabart for the user
                st.markdown(model_text_data["arabart"]["header"])
                st.markdown(model_text_data["arabart"]["text"])

                # Augment sentences with Arabart
                sentences_arabart_bert = aug_bert(model_text_data["arabart"]["url"],
                                                  user_text_input,
                                                  model_text_data["arabart"]["name"]
                                                  )

                # Generate List of similarity score for each augmented sentence and average similarity scores
                similarity_list, average_similarity = similarity_checker(
                    sentences_arabart_bert, user_text_input)
                list_of_dataframes.append(get_df_data(
                    sentences_arabart_bert, similarity_list))

                # Display Arabart results for the user
                with st.expander(model_text_data["arabart"]["results"]):
                    st.markdown(
                        f"Average Similarity: {average_similarity:.6f}")
                    display_similarity_table(
                        sentences_arabart_bert, similarity_list, model_text_data["arabart"]["name"])
                    st.markdown(
                        model_text_data["common"]["word-info-expander"])

        ## ---------------------- CAMeL-Lab/bert-base-arabic-camelbert-mix -------------------- ##
        if data['camelbert']:
            camelbert_bert_container = st.container()
            with camelbert_bert_container:
                # Camelbert details for the user
                st.markdown(model_text_data["camelbert"]["header"])
                st.markdown(model_text_data["camelbert"]["text"])

                # Augment sentences with Camelbert
                sentences_camelbert_bert = aug_bert(model_text_data["camelbert"]["url"],
                                                    user_text_input,
                                                    model_text_data["camelbert"]["name"]
                                                    )

                # Generate List of similarity score for each augmented sentence and average similarity scores
                similarity_list, average_similarity = similarity_checker(
                    sentences_camelbert_bert, user_text_input)
                list_of_dataframes.append(get_df_data(
                    sentences_camelbert_bert, similarity_list))

                # Display Camelbert results for the user
                with st.expander(model_text_data["camelbert"]["results"]):
                    st.markdown(
                        f"Average Similarity: {average_similarity:.6f}")
                    display_similarity_table(
                        sentences_camelbert_bert, similarity_list, model_text_data["camelbert"]["name"])
                    st.markdown(
                        model_text_data["common"]["word-info-expander"])

        ## --------------------------- asafaya/bert-large-arabic ------------------------------ ##
        if data['bert-large-arabic']:
            large_arabic_bert_container = st.container()
            with large_arabic_bert_container:
                # Bert Large Arabic Details for the user.
                st.markdown(model_text_data["bert-large-arabic"]["header"])
                st.markdown(model_text_data["bert-large-arabic"]["text"])

                # Augment sentences with Bert Large Arabic
                sentences_large_arabic_bert = aug_bert(model_text_data["bert-large-arabic"]["url"],
                                                       user_text_input,
                                                       model_text_data["bert-large-arabic"]["name"]
                                                       )

                # Generate List of similarity score for each augmented sentence and average similarity scores
                similarity_list, average_similarity = similarity_checker(
                    sentences_large_arabic_bert, user_text_input)
                list_of_dataframes.append(get_df_data(
                    sentences_large_arabic_bert, similarity_list))

                # Display results of Bert Large Arabic to the user
                with st.expander(model_text_data["bert-large-arabic"]["results"]):
                    st.markdown(
                        f"Average Similarity: {average_similarity:.6f}")
                    display_similarity_table(
                        sentences_large_arabic_bert, similarity_list, model_text_data["bert-large-arabic"]["name"])
                    st.markdown(
                        model_text_data["common"]["word-info-expander"])

        ## --------------------------------- UBC-NLP/ARBERT ----------------------------------- ##
        if data['ubc-arbert']:
            ubc_arbert_bert_container = st.container()
            with ubc_arbert_bert_container:
                # UBC-Arbert details for the user.
                st.markdown(model_text_data["ubc-arbert"]["header"])
                st.markdown(model_text_data["ubc-arbert"]["text"])

                # Augment sentences with UBC Arbert
                sentences_ubc_arbert_bert = aug_bert(model_text_data["ubc-arbert"]["url"],
                                                     user_text_input,
                                                     model_text_data["ubc-arbert"]["name"]
                                                     )

                # Generate List of similarity score for each augmented sentence and average similarity scores
                similarity_list, average_similarity = similarity_checker(
                    sentences_ubc_arbert_bert, user_text_input)
                list_of_dataframes.append(get_df_data(
                    sentences_ubc_arbert_bert, similarity_list))

                # Display results of UBC-Arbert to the user.
                with st.expander(model_text_data["ubc-arbert"]["results"]):
                    st.markdown(
                        f"Average Similarity: {average_similarity:.6f}")
                    display_similarity_table(
                        sentences_ubc_arbert_bert, similarity_list, model_text_data["ubc-arbert"]["name"])
                    st.markdown(
                        model_text_data["common"]["word-info-expander"])

        ## --------------------------------- UBC-NLP/MARBERTv2 -------------------------------- ##
        if data['ubc-marbertv2']:
            ubc_marbertv2_bert_container = st.container()
            with ubc_marbertv2_bert_container:
                # Show details of UBC-Marbertv2 to the user
                st.markdown(model_text_data["ubc-marbertv2"]["header"])
                st.markdown(model_text_data["ubc-marbertv2"]["text"])

                # Augment sentences with UBC-MarbertV2
                sentences_ubc_marbertv2_bert = aug_bert(model_text_data["ubc-marbertv2"]["url"],
                                                        user_text_input,
                                                        model_text_data["ubc-marbertv2"]["name"]
                                                        )

                # Generate List of similarity score for each augmented sentence and average similarity scores
                similarity_list, average_similarity = similarity_checker(
                    sentences_ubc_marbertv2_bert, user_text_input)
                list_of_dataframes.append(get_df_data(
                    sentences_ubc_marbertv2_bert, similarity_list))

                # Display results of UBC-MarbertV2 to the user
                with st.expander(model_text_data["ubc-marbertv2"]["results"]):
                    st.markdown(
                        f"Average Similarity: {average_similarity:.6f}")
                    display_similarity_table(
                        sentences_ubc_marbertv2_bert, similarity_list, model_text_data["ubc-marbertv2"]["name"])
                    st.markdown(
                        model_text_data["common"]["word-info-expander"])

        ## ------------------------ aubmindlab/araelectra-base-generator ---------------------- ##
        if data['araelectra']:
            araelectra_bert_container = st.container()
            with araelectra_bert_container:
                # Show Araelectra details to the user
                st.markdown(model_text_data["araelectra"]["header"])
                st.markdown(model_text_data["araelectra"]["text"])

                # Augment sentences with Araelectra
                sentences_araelectra_bert = aug_bert(model_text_data["araelectra"]["url"],
                                                     user_text_input,
                                                     model_text_data["araelectra"]["name"]
                                                     )

                # Generate List of similarity score for each augmented sentence and average similarity scores
                similarity_list, average_similarity = similarity_checker(
                    sentences_araelectra_bert, user_text_input)
                list_of_dataframes.append(get_df_data(
                    sentences_araelectra_bert, similarity_list))

                # Display Araelectra results to the user
                with st.expander(model_text_data["araelectra"]["results"]):
                    st.markdown(
                        f"Average Similarity: {average_similarity:.6f}")
                    display_similarity_table(
                        sentences_araelectra_bert, similarity_list, model_text_data["araelectra"]["name"])
                    st.markdown(
                        model_text_data["common"]["word-info-expander"])

        ## ------------------------------------- araGPT2 -------------------------------------- ##
        if data['aragpt2']:
            gpt2_container = st.container()
            with gpt2_container:
                # Show AraGPT2 details to the user
                st.markdown(model_text_data["aragpt2"]["header"])
                st.markdown(model_text_data["aragpt2"]["text"])

                # Augment sentences with AraGPT2
                sentences_gpt = aug_GPT(
                    model_text_data["aragpt2"]["url"], user_text_input)

                # Generate List of similarity score for each augmented sentence and average similarity scores
                similarity_list, average_similarity = similarity_checker(
                    sentences_gpt, user_text_input)
                list_of_dataframes.append(
                    get_df_data(sentences_gpt, similarity_list))

                # Display results of AraGPT2 to the user
                with st.expander(model_text_data["aragpt2"]["results"]):
                    st.markdown(
                        f"Average Similarity: {average_similarity:.6f}")
                    display_similarity_table(
                        sentences_gpt, similarity_list, model_text_data["aragpt2"]["name"])
                    st.markdown(
                        model_text_data["common"]["word-info-expander"])

        ## ------------------------------------- AraVec --------------------------------------- ##
        if data['aravec']:  # model not function currently
            w2v_container = st.container()
            with w2v_container:
                # Show details of Aravec model to the user
                st.markdown(model_text_data["aravec"]["header"])
                st.markdown(model_text_data["aravec"]["text"])

                # Augment sentences with aravec using different models
                sentences_w2v = aug_w2v(
                    './data/full_grams_cbow_100_twitter.mdl', 'glove-twitter-25', user_text_input, "Aravec")

                # Generate List of similarity score for each augmented sentence and average similarity scores
                similarity_list, average_similarity = similarity_checker(
                    sentences_w2v, user_text_input)

                # Display results of Aravec to the user
                with st.expander(model_text_data["aravec"]["results"]):
                    st.markdown(
                        f"Average Similarity: {average_similarity:.6f}")
                    display_similarity_table(
                        sentences_w2v, similarity_list, model_text_data["aravec"]["name"])
                    st.markdown(
                        model_text_data["common"]["word-info-expander"])

        ## ------------------------------------- Back- Translation ---------------------------- ##
        if data['double-back-translation']:
            back_translation_container = st.container()
            with back_translation_container:
                # Show details of Back translation to the user
                st.markdown(
                    model_text_data["double-back-translation"]["header"])
                available_languages = ['ar-en', 'ar-fr', 'ar-tr', 'ar-ru',
                                       'ar-pl', 'ar-it', 'ar-es', 'ar-el', 'ar-de', 'ar-he']
                back_translated_sentences = []
                st.markdown(model_text_data["double-back-translation"]["text"])
                st.markdown(
                    model_text_data["double-back-translation"]["text-2"])

                # Augment sentences with back translation
                back_translated_sentences = double_back_translate(
                    user_text_input)

                # Generate List of similarity score for each augmented sentence and average similarity scores
                similarity_list, average_similarity = similarity_checker(
                    back_translated_sentences, user_text_input)
                list_of_dataframes.append(get_df_data(
                    back_translated_sentences, similarity_list))

                # Display results of Double Back translation to the user
                with st.expander(model_text_data["double-back-translation"]["results"]):
                    st.markdown(
                        f"Average Similarity: {average_similarity:.6f}")
                    display_similarity_table(
                        back_translated_sentences, similarity_list, model_text_data["double-back-translation"]["name"])
                    st.markdown(
                        model_text_data["double-back-translation"]["results-info"])

        ## ----------------------- Download All Outputs to CSV -------------------------------- ##
        if len(list_of_dataframes) > 0:
            # Download Button for exporting outputs to csv file
            st.write("----------------------------")
            st.markdown("### Download all outputs as *one* CSV File?")
            download_all_outputs(list_of_dataframes)

## ---------------------------------------- End of 'Test the App' ------------------------------------------ ##
