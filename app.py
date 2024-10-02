import streamlit as st
import dill
import os
from dotenv import load_dotenv
from collections import defaultdict
import numpy as np
from nltk.stem import PorterStemmer

import openai

# Load the models
POS_TAGGER_MODEL_FILE = "hmm-pos-tagger.pkl"
with open(POS_TAGGER_MODEL_FILE, "rb") as f:
    hmm_model = dill.load(f)

# Load CRF model
CRF_POS_TAGGER_MODEL_FILE = "crf-pos-tagger.pkl"
with open(CRF_POS_TAGGER_MODEL_FILE, "rb") as f:
    crf_model = dill.load(f)

# Set OpenAI API key
load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Function for HMM POS tagging
def hmm_pos_tagger(sentence):    
    return [tag for word, tag in hmm_model.predict(sentence)]

# Function for CRF POS tagging
def crf_pos_tagger(sentence):
    prediction = crf_model.predict(sentence)
    return [tag for _, tag in prediction[0]]

# Function for GPT-4o POS tagging
def gpt4_pos_tagger(sentence):
    allowed_tags = "{'X', 'ADV', 'PRT', 'CONJ', 'ADP', 'VERB', 'PRON', 'ADJ', 'NOUN', '.', 'NUM', 'DET'}"
    prompt = (
        f"Tag the parts of speech for the following sentence using only these tags {allowed_tags}. "
        f"These tags are the same as the tags from the universal tag set of the brown corpus."
        f"If a word does not fit any category, use 'X'.\n\n"
        f"Sentence: '{sentence}'\n\n"
        "Output only the tags in the same order as the words, separated by spaces."
    )
    
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a part-of-speech (POS) tagger."},
            {"role": "user", "content": prompt}
        ]
    )
    
    tags = response.choices[0].message['content'].strip().split()
    return tags

# Function to create the comparison table
def create_comparison_table(sentence_words, gpt4_tags, crf_tags, hmm_tags):
    table = "<table style='width:100%; border: 1px solid black; border-collapse: collapse;'>"

    # Sentence words row
    table += "<tr><th style='border: 1px solid black; padding: 8px; text-align: left;'>Sentence</th>"
    for word in sentence_words:
        table += f"<td style='border: 1px solid black; padding: 8px;'>{word}</td>"
    table += "</tr>"

    # GPT-4 tags row
    table += "<tr><th style='border: 1px solid black; padding: 8px; text-align: left;'>GPT-4</th>"
    for tag in gpt4_tags:
        table += f"<td style='border: 1px solid black; padding: 8px; color: orange;'>{tag}</td>"
    table += "</tr>"

    # CRF tags row
    table += "<tr><th style='border: 1px solid black; padding: 8px; text-align: left;'>CRF POS Tagger</th>"
    for i, tag in enumerate(crf_tags):
        color = "green" if tag.strip() == gpt4_tags[i].strip() else "red"
        table += f"<td style='border: 1px solid black; padding: 8px;'><span style=\"color:{color}\">{tag}</span></td>"
    table += "</tr>"

    # HMM tags row
    table += "<tr><th style='border: 1px solid black; padding: 8px; text-align: left;'>HMM POS Tagger</th>"
    for i, tag in enumerate(hmm_tags):
        color = "green" if tag.strip() == gpt4_tags[i].strip() else "red"
        table += f"<td style='border: 1px solid black; padding: 8px;'><span style=\"color:{color}\">{tag}</span></td>"
    table += "</tr>"

    table += "</table>"
    
    return table

# Streamlit app configuration
st.set_page_config(page_title="POS Tagger Demo", page_icon="📊", layout="wide")
st.markdown("<h1 style='text-align: center;'>POS Tagger Comparison: HMM vs CRF vs GPT-4</h1>", unsafe_allow_html=True)

# Sentence input
sentence = st.text_area("Sentence Input", height=50, placeholder="Type your sentence here...", label_visibility="collapsed")

if st.button("Submit", use_container_width=True):
    if sentence.strip():
        sentence_words = hmm_model.tokenizer(sentence)  # Assuming the tokenizer is part of the HMM model
        
        gpt4_tags = gpt4_pos_tagger(sentence)
        crf_tags = crf_pos_tagger(sentence)
        hmm_tags = hmm_pos_tagger(sentence)

        # Adjust last word tagging if needed
        if sentence.strip():
            last_word = sentence.split()[-1]
            if last_word[-1] == '.'or'?'or'!'or '!!':
                crf_tags[-1] = '.'
                hmm_tags[-1] = '.'

        comparison_table = create_comparison_table(sentence_words, gpt4_tags, crf_tags, hmm_tags)
    
        st.write("### POS Tagging Comparison")
        st.markdown(comparison_table, unsafe_allow_html=True)
    else:
        st.warning("Please enter a sentence.")
