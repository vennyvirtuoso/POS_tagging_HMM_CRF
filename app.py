import streamlit as st
import dill
import os
from dotenv import load_dotenv
from collections import defaultdict
import numpy as np
from nltk.stem import PorterStemmer

POS_TAGGER_MODEL_FILE = "hmm-pos-tagger.pkl"

with open(POS_TAGGER_MODEL_FILE, "rb") as f:
    hmm_model = dill.load(f)

from collections import defaultdict
def hmm_pos_tagger(sentence):    
    return [tag for word, tag in hmm_model.predict(sentence)]

# Load CRF model
CRF_POS_TAGGER_MODEL_FILE = "crf-pos-tagger.pkl"
with open(CRF_POS_TAGGER_MODEL_FILE, "rb") as f:
    crf_model = dill.load(f)

def crf_pos_tagger(sentence):
    prediction = crf_model.predict(sentence)
    print(prediction)
    return [tag for _, tag in prediction[0]]

def create_comparison_table(sentence_words, crf_tags, hmm_tags):
    table = "<table style='width:100%; border: 1px solid black; border-collapse: collapse;'>"

    table += "<tr><th style='border: 1px solid black; padding: 8px; text-align: left;'>Sentence</th>"
    for word in sentence_words:
        table += f"<td style='border: 1px solid black; padding: 8px;'>{word}</td>"
    table += "</tr>"

    table += "<tr><th style='border: 1px solid black; padding: 8px; text-align: left;'>CRF POS Tagger</th>"
    for tag in crf_tags:
        table += f"<td style='border: 1px solid black; padding: 8px; color: orange;'>{tag}</td>"
    table += "</tr>"

    table += "<tr><th style='border: 1px solid black; padding: 8px; text-align: left;'>HMM POS Tagger</th>"
    ind = 0
    for tag in hmm_tags:
        if ind < len(crf_tags):
            color = "green" if tag.strip() == crf_tags[ind].strip() else "red"
        else:
            color = "red"
        table += f"<td style='border: 1px solid black; padding: 8px;'><span style=\"color:{color}\">{tag}</span></td>"
        ind += 1
    table += "</tr>"

    table += "</table>"
    
    return table

st.set_page_config(page_title="POS Tagger Demo", page_icon="📊", layout="wide")
st.markdown("<h1 style='text-align: center;'>POS Tagger Comparison: HMM vs CRF</h1>", unsafe_allow_html=True)

sentence = st.text_area("Sentence Input", height=50, placeholder="Type your sentence here...", label_visibility="collapsed")

if st.button("Submit", use_container_width=True):
    if sentence.strip():
        sentence_words = hmm_model.tokenizer(sentence)
        crf_tags = crf_pos_tagger(sentence)
        hmm_tags = hmm_pos_tagger(sentence)
        print(crf_tags)
        print(hmm_tags)
        comparison_table = create_comparison_table(sentence_words, crf_tags, hmm_tags)
    
        st.write("### POS Tagging Comparison")
        st.markdown(comparison_table, unsafe_allow_html=True)
    else:
        st.warning("Please enter a sentence.")
