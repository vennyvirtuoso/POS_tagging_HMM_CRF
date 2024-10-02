import streamlit as st
import dill
import pandas as pd
import os
from collections import defaultdict

# Load POS Tagger Models
POS_TAGGER_MODEL_FILE = "hmm-pos-tagger.pkl"
with open(POS_TAGGER_MODEL_FILE, "rb") as f:
    hmm_model = dill.load(f)

CRF_POS_TAGGER_MODEL_FILE = "crf-pos-tagger.pkl"
with open(CRF_POS_TAGGER_MODEL_FILE, "rb") as f:
    crf_model = dill.load(f)

def hmm_pos_tagger(sentence):
    return [tag for word, tag in hmm_model.predict(sentence)]

def crf_pos_tagger(sentence):
    return [tag for tag in crf_model.predict(sentence)]

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
    for i, tag in enumerate(hmm_tags):
        color = "green" if tag.strip() == crf_tags[i].strip() else "red"
        table += f"<td style='border: 1px solid black; padding: 8px;'><span style=\"color:{color}\">{tag}</span></td>"
    table += "</tr>"

    table += "</table>"
    return table

def generate_report(sentence, crf_tags, hmm_tags):
    report = f"### POS Tagging Report\n\n"
    report += f"**Input Sentence:** {sentence}\n\n"
    report += "#### CRF POS Tags:\n"
    report += ", ".join(crf_tags) + "\n\n"
    report += "#### HMM POS Tags:\n"
    report += ", ".join(hmm_tags) + "\n\n"
    return report

# Streamlit UI
st.set_page_config(page_title="POS Tagger Demo", page_icon="📊", layout="wide")
st.markdown("<h1 style='text-align: center;'>POS Tagger Comparison: HMM vs CRF</h1>", unsafe_allow_html=True)

# File Upload

# Use StringIO to simulate a file
df = pd.read_csv('testing.csv')

# Keep only the 'Sentence' column
df = df[['Sentence']]

# Print the output
uploaded_file = st.file_uploader("Upload a CSV file with sentences", type=["csv"])
if uploaded_file:
    df = df[['Sentence']]
    st.write(df)

sentence_input = st.text_area("Sentence Input", height=50, placeholder="Type your sentence here...", label_visibility="collapsed")

if st.button("Submit", use_container_width=True):
    if sentence_input.strip():
        sentence_words = hmm_model.tokenizer(sentence_input)
        crf_tags = crf_pos_tagger(sentence_input)
        hmm_tags = hmm_pos_tagger(sentence_input)

        # Create comparison table
        comparison_table = create_comparison_table(sentence_words, crf_tags, hmm_tags)
    
        st.write("### POS Tagging Comparison")
        st.markdown(comparison_table, unsafe_allow_html=True)

        # Generate report
        report = generate_report(sentence_input, crf_tags, hmm_tags)
        st.write(report)
    else:
        st.warning("Please enter a sentence.")
