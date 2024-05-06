import re
import numpy as np
import streamlit as st 
import nltk
nltk.download('stopwords')
nltk.download('punkt')

import networkx as nx
import base64
import pdfplumber
import PyPDF2

from newspaper import Article
from bs4 import BeautifulSoup
from tqdm import tqdm
from io import BytesIO
from pdfminer.high_level import extract_text
from transformers import AutoTokenizer, LongT5ForConditionalGeneration
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
from PyPDF2 import PdfWriter
from rouge_score import rouge_scorer
from pdfminer.high_level import extract_text


st.set_page_config(layout="wide")

tokenizer = AutoTokenizer.from_pretrained("pszemraj/long-t5-tglobal-base-16384-book-summary", padding_side="left", do_basic_tokenize=False, is_split_into_words=True)
model = LongT5ForConditionalGeneration.from_pretrained("pszemraj/long-t5-tglobal-base-16384-book-summary")

@st.cache_data(show_spinner=False)
def extract_word_embeddings(file_path):
    word_embeddings = {}
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            word_embeddings[word] = coefs
    return word_embeddings

word_embeddings = extract_word_embeddings('glove.6B.100d.txt')

@st.cache_data(show_spinner=False)
def save_uploadedfile(uploadedfile):
    file_path = uploadedfile.name
    with open(file_path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return file_path

@st.cache_data(show_spinner="Extracting text from PDF")
def extract_text_from_pdf(pdf_path):
    text = ''
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)

        for page_num in range(num_pages):
            with pdfplumber.open(file) as pdf:
                page = pdf.pages[page_num]
                text += page.extract_text()
                text += '\n'
    return text

@st.cache_data(show_spinner=False)
def extract_and_return_text(url):
    article = Article(url)

    # Download, parse, and perform natural language processing
    article.download()
    article.parse()
    article.nlp()

    # Return the extracted text
    return article.text
  
@st.cache_data(show_spinner=False)
def num_of_words(text):
    words = text.split()
    num_words = len(words)
    return num_words 

@st.cache_data(show_spinner=False)
def preprocessing(sentences):
    clean_sentences = []
    for i in range(len(sentences)):
        sen = re.sub('[^a-zA-Z0-9\s]', " ", sentences[i])
        sen = sen.lower()
        sen = sen.split()
        sen = ' '.join([i for i in sen if i not in stopwords.words('english')])
        clean_sentences.append(sen)
    return clean_sentences

@st.cache_data(show_spinner=False)
def sentence_to_vectors(clean_sentences, word_embeddings):
    sentence_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i]) / (len(i) + 0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)
    return sentence_vectors

@st.cache_data(show_spinner=False)
def get_similarity_matrix(sentences,sentence_vectors):
    sim_mat = np.zeros([len(sentences), len(sentences)])

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1, 100), sentence_vectors[j].reshape(1, 100))[0,0]
    return sim_mat

@st.cache_data(show_spinner=False)
def textrank_summary(sim_mat, sentences, ratio):
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    total_sentences = len(ranked_sentences)
    num_summary_sentences = int(total_sentences * ratio)
    textrank_summary = [sentence[1] for sentence in ranked_sentences[:num_summary_sentences]]
    textrank_summary = ' '.join(textrank_summary)
    return textrank_summary

@st.cache_data(show_spinner=False)
def generate_t5_summary(textrank_summary, _tokenizer, _model):
    inputs = tokenizer(textrank_summary, return_tensors="pt", padding='longest')
    input_ids = inputs.input_ids
    input_mask = inputs.attention_mask

    summary_ids = model.generate(input_ids, max_length=16384, length_penalty=5.0)
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary_text

@st.cache_data(show_spinner=False)
def summarize_text_dynamically(input_text, ratio):
    # Create the progress text
    progress_text = st.empty()
    progress_text.text("Summarisation in progress. Please wait.")
    # Create the progress bar
    progress_bar = st.progress(0)
    sentences = sent_tokenize(input_text)
    progress_bar.progress(10)  # 10% progress
    clean_sentences = preprocessing(sentences)
    progress_bar.progress(20)  # 20% progress
    progress_bar.progress(30)  # 30% progress
    sentence_vectors = sentence_to_vectors(clean_sentences, word_embeddings)
    progress_bar.progress(40)  # 40% progress
    similarity_matrix = get_similarity_matrix(clean_sentences, sentence_vectors)
    progress_bar.progress(60)  # 60% progress
    text_rank_summary = textrank_summary(similarity_matrix, sentences, ratio)
    progress_bar.progress(80)  # 80% progress
    with st.expander("Extractive Summary (Key Sentences: TextRank) "):
        st.markdown(text_rank_summary)
        words = num_of_words(text_rank_summary)
        st.markdown(f'<div style="text-align: right;"> {words} words </div>', unsafe_allow_html=True)
    t5_summary = generate_t5_summary(text_rank_summary, tokenizer, model)
    progress_bar.progress(100)  # 100% progress
    progress_text.text("Summarisation completed.") 
    return t5_summary

def main():
    choice = st.sidebar.selectbox("Select your choice", ["Text", "URL", "Upload File"])
    
    if choice == "Text":
        st.subheader("Text Summariser on Text")
        input_text = st.text_area("Paste your input text", height=300)
        words = num_of_words(input_text)
        st.markdown(f'<div style="text-align: right;"> {words} words </div>', unsafe_allow_html=True)
        if input_text is not None:
            ratio = st.slider("Adjust Key Sentences Summary Ratio", 0.1, 1.0, 0.4, 0.05)
            if st.button("Get Summary！"):
                t5_summary = summarize_text_dynamically(input_text, ratio)
                st.markdown("Abstractive Summary Generated (Long T5 Model)")
                st.success(t5_summary)
                words = num_of_words(t5_summary)
                st.markdown(f'<div style="text-align: right;"> {words} words </div>', unsafe_allow_html=True)
 
    elif choice == "Upload File":
        st.subheader("Text Summariser on Uploaded File :file_folder:")
        input_file = st.file_uploader("Upload your document here", type=['pdf', 'txt'])
        if input_file is not None:
            ratio = st.slider("Adjust Key Sentences Summary Ratio", 0.1, 1.0, 0.4, 0.05)
            if st.button("Get Summary！"):
                # Rows for layout
                col1, col2 = st.columns([2,1])

                with col1:
                    # Save the uploaded file and get the file path
                    file_path = save_uploadedfile(input_file)
                    with st.spinner("Extracting text... This may take a moment."):
                        # Display PDF viewer or Text content
                        st.markdown("**Document Viewer:**")
                        if file_path.endswith('.pdf'):
                            with open(file_path, "rb") as f:
                                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                                pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="750" height="1000" type="application/pdf"></iframe>'
                                st.markdown(pdf_display, unsafe_allow_html=True)
                                pdf_text = extract_text_from_pdf(file_path)
                                # st.write(pdf_text)
                                # Display word count
                                words = num_of_words(pdf_text)
                                st.markdown(f'<div style="text-align: left;"> {words} words </div>', unsafe_allow_html=True)
                        else:  # Assume it's a text file
                            with open(file_path, 'r') as f:
                                text_content = f.read()
                                st.text_area("Text Content", text_content, height=400)
                                # Display word count
                                words = num_of_words(text_content)
                                st.markdown(f'<div style="text-align: left;"> {words} words </div>', unsafe_allow_html=True)
                                
                with col2:
                    if file_path.endswith('.pdf'):
                        pdf_text = extract_text_from_pdf(file_path)
                        
                    else:  # Assume it's a text file
                        pdf_text = text_content 

                    t5_summary = summarize_text_dynamically(pdf_text, ratio)
                    st.markdown("Abstractive Summary Generated (Long T5 Model)")
                    st.success(t5_summary)
                    words = num_of_words(t5_summary)
                    st.markdown(f'<div style="text-align: right;"> {words} words </div>', unsafe_allow_html=True)

                        
    elif choice == "URL":
            st.subheader("Text Summariser on Website")
            url = st.text_input("Enter the URL of the website:")
            if url is not None:
                ratio = st.slider("Adjust Key Sentences Summary Ratio", 0.1, 1.0, 0.4, 0.05)
                if st.button("Get Summary!"):
                    if url:
                        try:
                            with st.spinner("Extracting text... This may take a moment."):
                                extracted_text = extract_and_return_text(url)
                            st.success("Text extraction successful.")
                            words = num_of_words(extracted_text)
                            st.markdown(f'<div style="text-align: right;"> {words} words </div>', unsafe_allow_html=True)
                            # st.markdown(extracted_text)
                            t5_summary = summarize_text_dynamically(extracted_text, ratio)
                            st.markdown("Abstractive Summary Generated (Long T5 Model)")
                            st.success(t5_summary)
                            words = num_of_words(t5_summary)
                            st.markdown(f'<div style="text-align: right;"> {words} words </div>', unsafe_allow_html=True)
                        except Exception as e:
                            st.error("An error occurred during text extraction.")
                            st.error(str(e))
                    else:
                        st.warning("Please enter a URL.")

if __name__ == '__main__':
	main()
