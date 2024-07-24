import logging
import os
import re
import shutil
import tempfile
import warnings
from random import randint, random
from typing import List, Dict, Any, Tuple, Optional, Union

import nltk
import numpy as np
import pandas as pd
import spacy
import streamlit as st
import torch
from PIL import Image
from PyPDF2 import PdfReader
from docx import Document

from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords, wordnet
from pytesseract import image_to_string
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    RobertaTokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
    pipeline
)
from transformers.models.roberta import RobertaForQuestionAnswering
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from contractions import expand_contractions

DEFAULT_SIMILARITY_THRESHOLD = 0.2
LOG_FILE_PATH = "app.log"
UPLOAD_FOLDER = "uploads"
TEMP_FOLDER = "temp"
MODEL_NAMES = {
    "QA": "deepset/roberta-base-squad2",
    "SUMMARIZER": "google/pegasus-xsum"
}
# Create the upload folder if it does not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Setup warnings and logging
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint .* were not used.*")
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


# Custom exception classes
class FileProcessingError(Exception):
    pass


class ModelLoadingError(Exception):
    pass


class AnalysisError(Exception):
    pass


class UnsupportedFileTypeError(Exception):
    pass


# Load models
try:
    nlp = spacy.load("en_core_web_sm")
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise ModelLoadingError("An error occurred while loading the models.")

# Initialize sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()


# Model initialization functions
@st.cache_resource
def load_qa_model(model_name: str):
    try:
        if model_name == 'RoBERTa':
            return RobertaForQuestionAnswering.from_pretrained(
                'deepset/roberta-base-squad2'), RobertaTokenizer.from_pretrained('deepset/roberta-base-squad2')
        else:
            raise ModelLoadingError(f"Unsupported model name: {model_name}")
    except Exception as e:
        logger.error(f"Error loading QA model: {e}")
        raise ModelLoadingError(f"Error loading QA model: {e}")


@st.cache_resource
def load_summarizer_model():
    try:
        return BartForConditionalGeneration.from_pretrained('facebook/bart-large-xsum'), BartTokenizer.from_pretrained(
            'facebook/bart-large-xsum')
    except Exception as e:
        logger.error(f"Error loading summarizer model: {e}")
        raise ModelLoadingError("An error occurred while loading the summarizer model.")


qa_model, qa_tokenizer = load_qa_model('RoBERTa')
summarizer_model, summarizer_tokenizer = load_summarizer_model()


# Centralized exception handling
def handle_exception(e, message):
    logger.error(f"{message}: {str(e)}")
    st.error(f"{message}. Please try again.")


# Text preprocessing functions
def preprocess_text(text: str) -> List[str]:
    try:
        text = expand_contractions(text)
        text = re.sub(r'[^\w\s]', '', text.lower())
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    except Exception as e:
        logger.error(f"Error during text preprocessing: {e}")
        raise


# Text Augmentation class
class TextAugmentation:
    def __init__(self, aug_prob: float = 0.1):
        self.aug_prob = aug_prob

    def augment_text(self, text: str) -> str:
        try:
            aug_methods = [self.synonym_replacement, self.random_insertion, self.random_swap, self.random_deletion]
            for method in aug_methods:
                if random() < self.aug_prob:
                    text = method(text)
            return text
        except Exception as e:
            logger.error(f"Error during text augmentation: {e}")
            raise

    def synonym_replacement(self, text: str) -> str:
        try:
            words = text.split()
            new_words = self.replace_synonyms(words)
            return ' '.join(new_words)
        except Exception as e:
            logger.error(f"Error during synonym replacement: {e}")
            raise

    def replace_synonyms(self, words: List[str]) -> List[str]:
        try:
            new_words = words.copy()
            num_replacements = max(1, int(len(words) * self.aug_prob))
            for _ in range(num_replacements):
                word_to_replace = words[randint(0, len(words) - 1)]
                synonyms = self.get_synonyms(word_to_replace)
                if synonyms:
                    synonym = synonyms[randint(0, len(synonyms) - 1)]
                    new_words = [synonym if word == word_to_replace else word for word in new_words]
            return new_words
        except Exception as e:
            logger.error(f"Error replacing synonyms: {e}")
            raise

    def random_insertion(self, text: str) -> str:
        try:
            words = text.split()
            new_words = words.copy()
            num_insertions = max(1, int(len(words) * self.aug_prob))
            for _ in range(num_insertions):
                new_words.insert(randint(0, len(new_words) - 1), self.get_random_word())
            return ' '.join(new_words)
        except Exception as e:
            logger.error(f"Error during random insertion: {e}")
            raise

    def random_swap(self, text: str) -> str:
        try:
            words = text.split()
            new_words = self.swap_random_words(words)
            return ' '.join(new_words)
        except Exception as e:
            logger.error(f"Error during random swap: {e}")
            raise

    def swap_random_words(self, words: List[str]) -> List[str]:
        try:
            new_words = words.copy()
            num_swaps = max(1, int(len(words) * self.aug_prob))
            for _ in range(num_swaps):
                idx1, idx2 = randint(0, len(new_words) - 1), randint(0, len(new_words) - 1)
                new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
            return new_words
        except Exception as e:
            logger.error(f"Error during swapping words: {e}")
            raise

    def random_deletion(self, text: str) -> str:
        try:
            words = text.split()
            new_words = [word for word in words if random() > self.aug_prob]
            return ' '.join(new_words)
        except Exception as e:
            logger.error(f"Error during random deletion: {e}")
            raise

    def get_synonyms(self, word: str) -> List[str]:
        try:
            synonyms = set()
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonyms.add(lemma.name())
            return list(synonyms)
        except Exception as e:
            logger.error(f"Error getting synonyms: {e}")
            raise

    def get_random_word(self) -> str:
        try:
            words = list(wordnet.words())
            return words[randint(0, len(words) - 1)]
        except Exception as e:
            logger.error(f"Error getting random word: {e}")
            raise


# Dataset class
class TextDataset:
    def __init__(self, dataframe: pd.DataFrame, text_column: str, target_column: Optional[str] = None,
                 augment: bool = False, aug_prob: float = 0.1,
                 similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
        self.dataframe = dataframe
        self.text_column = text_column
        self.target_column = target_column
        self.augment = augment
        self.augmenter = TextAugmentation(aug_prob)
        self.similarity_threshold = similarity_threshold

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        try:
            text = preprocess_text(self.dataframe.iloc[idx][self.text_column])
            if self.augment:
                text = self.augmenter.augment_text(" ".join(text))
            target = self.dataframe.iloc[idx][self.target_column] if self.target_column else None
            return {'text': text, 'target': target}
        except Exception as e:
            logger.error(f"Error getting item from dataset: {e}")
            raise

    def answer_question(self, question: str) -> str:
        try:
            processed_question = preprocess_text(question)
            best_match = None
            highest_similarity = 0
            for idx in range(len(self.dataframe)):
                text_entry = preprocess_text(self.dataframe.iloc[idx][self.text_column])
                similarity = calculate_similarity(" ".join(processed_question), " ".join(text_entry))
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = text_entry
            return " ".join(best_match) if best_match else "No relevant answer found."
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            raise


# Document analysis functions
def analyze_file(file_content_or_path: Union[bytes, str], file_name: Optional[str] = None,
                 question: Optional[str] = None, progress: Optional[st.delta_generator.DeltaGenerator] = None) -> Dict[
    str, Any]:
    try:
        logger.info("Analyzing file: %s", file_name)
        if progress:
            progress.progress(10)
        file_path = save_temp_file(file_content_or_path) if isinstance(file_content_or_path,
                                                                       bytes) else file_content_or_path
        if not os.path.exists(file_path):
            raise FileProcessingError(f"File does not exist: {file_path}")
        if progress:
            progress.progress(30)
        document_info = analyze_document(file_path, file_name, question=question)
        logger.info("File analysis completed: %s", file_name)
        if progress:
            progress.progress(70)
        os.remove(file_path)
        return document_info
    except Exception as e:
        handle_exception(e, "Unexpected error during file analysis")
        raise AnalysisError("An unexpected error occurred during file analysis.")


def analyze_document(file_path: str, file_name: str, question: Optional[str] = None) -> Dict[str, Any]:
    try:
        text_content = read_file_content(file_path, file_name)
        if question:
            answer = answer_question(question, text_content)
        else:
            answer = None
        preprocessed_text, named_entities, keywords, sentiment = process_text(text_content)
        return {
            'text_content': text_content,
            'preprocessed_text': preprocessed_text,
            'named_entities': named_entities,
            'keywords': keywords,
            'answer': answer,
            'sentiment': sentiment
        }
    except Exception as e:
        handle_exception(e, "Unexpected error during document analysis")
        raise AnalysisError("An unexpected error occurred during document analysis.")


def read_file_content(file_path: str, file_name: str) -> str:
    try:
        logger.info("Reading file content: %s", file_name)
        if file_name.lower().endswith('.txt'):
            return read_text_file(file_path)
        elif file_name.lower().endswith('.pdf'):
            return parse_text_from_pdf(file_path)
        elif file_name.lower().endswith('.docx'):
            return parse_text_from_docx(file_path)
        elif file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            return parse_text_from_image(file_path)
        else:
            raise UnsupportedFileTypeError(f"Unsupported file format: {file_name}")
    except Exception as e:
        handle_exception(e, "Unexpected error while reading file content")
        raise FileProcessingError("An error occurred while reading the file content.")


def read_text_file(file_path: str) -> str:
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            if not content.strip():
                raise FileProcessingError("The text file is empty.")
            return content
    except Exception as e:
        handle_exception(e, "Unexpected error while reading text file")
        raise FileProcessingError("An error occurred while reading the text file.")


def parse_text_from_pdf(file_path: str) -> str:
    try:
        logger.info("Parsing text from PDF: %s", file_path)
        text_content = ""
        reader = PdfReader(file_path)
        for page in reader.pages:
            text_content += page.extract_text() or ""
        if not text_content.strip():
            raise FileProcessingError("The PDF file is empty or does not contain readable text.")
        return text_content
    except Exception as e:
        handle_exception(e, "Unexpected error while parsing PDF file")
        raise FileProcessingError("An error occurred while parsing the PDF file.")


def parse_text_from_docx(file_path: str) -> str:
    try:
        logger.info("Parsing text from DOCX: %s", file_path)
        doc = Document(file_path)
        content = "\n".join([para.text for para in doc.paragraphs])
        if not content.strip():
            raise FileProcessingError("The DOCX file is empty or does not contain readable text.")
        return content
    except Exception as e:
        handle_exception(e, "Unexpected error while parsing DOCX file")
        raise FileProcessingError("An error occurred while parsing the DOCX file.")


def parse_text_from_image(file_path: str) -> str:
    try:
        logger.info("Parsing text from image: %s", file_path)
        image = Image.open(file_path)
        text_content = image_to_string(image)
        if not text_content.strip():
            raise FileProcessingError("The image file does not contain readable text.")
        return text_content
    except Exception as e:
        handle_exception(e, "Unexpected error while parsing image file")
        raise FileProcessingError("An error occurred while parsing the image file.")


def extract_named_entities(text: str) -> List[str]:
    try:
        logger.info("Extracting named entities...")
        ner_results = ner_pipeline(text)
        named_entities_list = []
        for entity in ner_results:
            entity_text = entity['word']
            if entity['entity'].startswith("I-") and named_entities_list:
                named_entities_list[-1] += f" {entity_text}"
            else:
                named_entities_list.append(entity_text)
        return list(set(named_entities_list))
    except Exception as e:
        handle_exception(e, "Unexpected error while extracting named entities")
        raise AnalysisError("An error occurred while extracting named entities.")


def extract_keywords(text: str, num_keywords: int = 5) -> List[str]:
    try:
        logger.info("Extracting keywords...")
        tokens = preprocess_text(text)
        processed_text = ' '.join(tokens)
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform([processed_text])
        feature_names = tfidf.get_feature_names_out()
        top_indices = tfidf_matrix.toarray().argsort()[0][::-1][:num_keywords]
        return [feature_names[idx] for idx in top_indices]
    except Exception as e:
        handle_exception(e, "Unexpected error while extracting keywords")
        raise AnalysisError("An error occurred while extracting keywords.")


# File saving functions
def save_uploaded_file(file_content: bytes, folder: str = "uploads", filename: str = "uploaded_file") -> str:
    try:
        logger.info("Saving uploaded file: %s", filename)
        os.makedirs(folder, exist_ok=True)
        file_path = os.path.join(folder, filename)
        with open(file_path, "wb") as f:
            f.write(file_content)
        logger.info("File saved successfully: %s", file_path)
        return file_path
    except Exception as e:
        handle_exception(e, "Unexpected error while saving uploaded file")
        raise FileProcessingError("An error occurred while saving the uploaded file.")


def save_temp_file(file_content: bytes) -> str:
    try:
        logger.info("Creating temporary file...")
        os.makedirs(TEMP_FOLDER, exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, dir=TEMP_FOLDER) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        logger.info("Temporary file created: %s", temp_file_path)
        return temp_file_path
    except Exception as e:
        handle_exception(e, "Unexpected error while creating temporary file")
        raise FileProcessingError("An error occurred while creating a temporary file.")


def clean_temp_files():
    try:
        logger.info("Cleaning temporary files...")
        if os.path.exists(TEMP_FOLDER):
            shutil.rmtree(TEMP_FOLDER)
        os.makedirs(TEMP_FOLDER, exist_ok=True)
    except Exception as e:
        handle_exception(e, "Unexpected error while cleaning temporary files")
        raise FileProcessingError("An error occurred while cleaning temporary files.")


# Similarity calculation function
def calculate_similarity(text1: str, text2: str) -> float:
    try:
        embeddings1 = sentence_model.encode(text1, convert_to_tensor=True)
        embeddings2 = sentence_model.encode(text2, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embeddings1, embeddings2)
        return similarity.item()
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        raise


# Text processing functions
def process_text(text: str) -> Tuple[List[str], List[str], List[str], Dict[str, float]]:
    try:
        preprocessed_text = preprocess_text(text)
        named_entities = extract_named_entities(" ".join(preprocessed_text))
        keywords = extract_keywords(" ".join(preprocessed_text))
        sentiment = analyze_sentiment(" ".join(preprocessed_text))
        return preprocessed_text, named_entities, keywords, sentiment
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        raise


# Sentiment analysis function
def analyze_sentiment(text: str) -> Dict[str, float]:
    try:
        sentiment_scores = sentiment_analyzer.polarity_scores(text)
        return sentiment_scores
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        raise AnalysisError("An error occurred while analyzing sentiment.")


# Question answering functions
def is_question_related(question: str, text_content: str, threshold: float) -> bool:
    try:
        logger.info("Checking question relevance...")
        sentences = nltk.sent_tokenize(text_content)
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform([question] + sentences)
        similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        mean_score = np.mean(similarity_scores)
        std_dev = np.std(similarity_scores)
        adaptive_threshold = mean_score + std_dev
        return any(score >= threshold for score in similarity_scores)
    except Exception as e:
        handle_exception(e, "Unexpected error while checking question relevance")
        raise AnalysisError("An error occurred while checking question relevance.")


def answer_question(question: str, context: str, model_name: str = 'RoBERTa') -> str:
    try:
        logger.info("Generating answer to the question...")
        if not context:
            raise ValueError("No context provided for question answering.")

        max_length = 512
        stride = 128
        context_tokens = qa_tokenizer.tokenize(context)
        question_tokens = qa_tokenizer.tokenize(question)

        chunked_inputs = []
        for i in range(0, len(context_tokens), max_length - len(question_tokens) - 3 - stride):
            chunk = context_tokens[i:i + (max_length - len(question_tokens) - 3)]
            inputs = qa_tokenizer.encode_plus(question_tokens, chunk, return_tensors='pt', max_length=max_length,
                                              truncation=True)
            chunked_inputs.append(inputs)

        answers = []
        for inputs in chunked_inputs:
            outputs = qa_model(**inputs)
            answer_start = torch.argmax(outputs[0])
            answer_end = torch.argmax(outputs[1]) + 1
            answer = qa_tokenizer.convert_tokens_to_string(
                qa_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
            answers.append(answer)

        return max(answers, key=len) if answers else "No relevant answer found."
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        raise AnalysisError("An error occurred while generating an answer.")


# Streamlit interface
def main():
    st.title("Multi-File Upload and Analysis")
    uploaded_files = st.sidebar.file_uploader("Upload files",
                                              type=["txt", "pdf", "docx", "png", "jpg", "jpeg", "tiff", "bmp"],
                                              accept_multiple_files=True)
    model_options = ["RoBERTa"]
    model_name = st.sidebar.selectbox("Choose QA Model", model_options)
    aug_prob = st.sidebar.slider("Text Augmentation Probability", 0.0, 1.0, 0.1, 0.05)
    similarity_threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, DEFAULT_SIMILARITY_THRESHOLD, 0.05)

    if uploaded_files:
        handle_file_upload(uploaded_files, model_name, aug_prob, similarity_threshold)

    question = st.text_input("Enter your question (optional):")
    ask_button = st.button("Ask")

    if question and uploaded_files and ask_button:
        handle_question_answering(question, uploaded_files)


def handle_file_upload(uploaded_files, model_name, aug_prob, similarity_threshold):
    clean_temp_files()  # Clean temporary files before processing new ones
    for uploaded_file in uploaded_files:
        file_content = uploaded_file.getvalue()
        file_name = uploaded_file.name
        file_path = save_uploaded_file(file_content, UPLOAD_FOLDER, file_name)
        display_file_metadata(file_content, file_name)
        progress = st.progress(0)
        try:
            analysis_results = analyze_file(file_path, file_name, progress=progress)
            progress.progress(80)
            display_response_answers(analysis_results)
            progress.progress(100)
        except FileProcessingError as e:
            st.error(f"File processing error: {str(e)}")
        except AnalysisError as e:
            st.error(f"Analysis error: {str(e)}")
        except Exception as e:
            st.error("An unexpected error occurred. Please try again.")
            logger.error(f"Unexpected error: {str(e)}")


def handle_question_answering(question, uploaded_files):
    clean_temp_files()  # Clean temporary files before processing new ones
    for uploaded_file in uploaded_files:
        progress = st.progress(0)
        file_content = uploaded_file.getvalue()
        file_name = uploaded_file.name
        file_path = save_uploaded_file(file_content, UPLOAD_FOLDER, file_name)
        try:
            analysis_results = analyze_file(file_path, file_name, question=question, progress=progress)
            progress.progress(80)
            if analysis_results.get('answer'):
                st.write(f"Question: {question}")
                st.write(f"Answer: {analysis_results['answer']}")
                st.write("----------------------------------------")
                summarized_content = summarize_answer(analysis_results['text_content'], analysis_results['answer'])
                st.write("Summarized Answer:")
                st.write(summarized_content)
            else:
                st.write("Answer not found.")
            progress.progress(100)
        except AnalysisError as e:
            st.error(f"Analysis error: {str(e)}")
        except Exception as e:
            st.error("An unexpected error occurred. Please try again.")
            logger.error(f"Unexpected error: {str(e)}")


def display_file_metadata(file_content: bytes, file_name: str):
    st.write(f"Uploaded file: {file_name}")
    st.write(f"File size: {len(file_content)} bytes")


def display_response_answers(analysis_results: Dict[str, Any]):
    st.write("Named Entities:")
    st.write(", ".join(analysis_results['named_entities']))
    st.write("Keywords:")
    st.write(", ".join(analysis_results['keywords']))
    st.write("Sentiment Scores:")
    st.write(analysis_results['sentiment'])
    if analysis_results['answer']:
        st.write("Answer:")
        st.write(analysis_results['answer'])


def summarize_answer(context: str, answer: str) -> str:
    inputs = summarizer_tokenizer.encode("summarize: " + answer, return_tensors='pt', max_length=512, truncation=True)
    summary_ids = summarizer_model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4,
                                            early_stopping=True)
    return summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)


if __name__ == '__main__':
    main()
