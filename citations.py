import streamlit as st

def citations():
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