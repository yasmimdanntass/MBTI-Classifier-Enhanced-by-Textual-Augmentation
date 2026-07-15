import pandas as pd
import string
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from pipeline.utils.emoticons_and_expressions import EMOTICONS_EMO, CHAT_WORDS_STR

nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

class Preprocessor:

    def __init__(self, texts_column):
        self.texts_column = texts_column
        
        self.mbti_types = {
            'infj', 'infp', 'intj', 'intp', 'isfj', 'isfp', 'istj', 'istp',
            'enfj', 'enfp', 'entj', 'entp', 'esfj', 'esfp', 'estj', 'estp'
        }

    def lowercase(self, df):
        df[self.texts_column] = df[self.texts_column].apply(lambda x: str(x).lower())
        return df

    def expand_contractions(self, df):
        contraction_mapping = {
            "ain't": "is not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have",
            "'cause": "because", "could've": "could have", "couldn't": "could not",
            "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not",
            "don't": "do not", "hadn't": "had not", "hadn't've": "had not have",
            "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'd've": "he would have",
            "he'll": "he will", "he'll've": "he will have", "he's": "he is", "how'd": "how did",
            "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "i'd": "i would",
            "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have", "i'm": "i am",
            "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have",
            "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us",
            "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not",
            "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
            "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
            "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
            "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
            "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
            "she'll've": "she will have", "she's": "she is", "should've": "should have",
            "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
            "so's": "so as", "that'd": "that would", "that'd've": "that would have",
            "that's": "that is", "there'd": "there would", "there'd've": "there would have",
            "there's": "there is", "they'd": "they would", "they'd've": "they would have",
            "they'll": "they will", "they'll've": "they will have", "they're": "they are",
            "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would",
            "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
            "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
            "what'll've": "what will have", "what're": "what are", "what's": "what is",
            "what've": "what have", "when's": "when is", "when've": "when have",
            "where'd": "where did", "where's": "where is", "where've": "where have",
            "who'll": "who will", "who'll've": "who will have", "who's": "who is",
            "who've": "who have", "why's": "why is", "why've": "why have",
            "will've": "will have", "won't": "will not", "won't've": "will not have",
            "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
            "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have",
            "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would",
            "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
            "you're": "you are", "you've": "you have"
        }
        
        sorted_keys = sorted(contraction_mapping.keys(), key=len, reverse=True)
        contraction_pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in sorted_keys) + r')\b')
        
        def expand_match(match):
            return contraction_mapping.get(match.group(0))

        def expand_single(text):
            return contraction_pattern.sub(expand_match, str(text))

        df[self.texts_column] = df[self.texts_column].apply(expand_single)
        return df

    def remove_links_mentions_urls(self, df):
        def remove_single(text):
            pattern = re.compile(r'https?://[^\s|]+|www\.[^\s|]+|@[^\s|]+|#[^\s|]+')
            return pattern.sub(r'', str(text))
        df[self.texts_column] = df[self.texts_column].apply(remove_single)
        return df

    def retain_alphabetic(self, df):
        def retain_alpha_single(text):
            text = re.sub(r'[^a-zA-Z\s]', ' ', str(text))
            return ' '.join(text.split())
        df[self.texts_column] = df[self.texts_column].apply(retain_alpha_single)
        return df

    def remove_punctuation(self, df):
        def remove_punctuation_single(text):
            text = str(text).replace('_', ' ')
            return ' '.join(text.translate(str.maketrans('', '', string.punctuation)).split())
        
        df[self.texts_column] = df[self.texts_column].apply(remove_punctuation_single)
        return df
    
    def remove_whitespaces(self, df):
        df[self.texts_column] = df[self.texts_column].apply(lambda x: str(x).strip())
        return df
    
    def remove_stopwords(self, df):
        stop_words = set(stopwords.words('english'))

        excecoes = {'i', 'me', 'you'}
        stop_words = stop_words - excecoes

        def remove_stopwords_single(text):
            return ' '.join([word for word in str(text).split() if word not in stop_words])
        
        df[self.texts_column] = df[self.texts_column].apply(remove_stopwords_single)
        return df
    
    def lemmatize(self, df):
        lemmatizer = WordNetLemmatizer()

        def lemmatize_words(text):
            return " ".join([lemmatizer.lemmatize(word) for word in text.split()])
        
        df[self.texts_column] = df[self.texts_column].apply(lemmatize_words)
        return df
    
    def stem(self, df):
        stemmer = PorterStemmer()

        def stem_words(text):
            return " ".join([stemmer.stem(word) for word in text.split()])
        
        df[self.texts_column] = df[self.texts_column].apply(stem_words)   
        return df
    
    def remove_urls(self, df):
        def remove_urls_single(text):
            url_pattern = re.compile(r'https?://[^\s|]+|www\.[^\s|]+')
            return url_pattern.sub(r'', str(text))
        df[self.texts_column] = df[self.texts_column].apply(remove_urls_single)   
        return df
    
    def convert_emoticons(self, df):
        def convert_emoticons_single(text):
            for emot in EMOTICONS_EMO:
                pattern = re.escape(emot) 
                replacement = "_".join(
                    EMOTICONS_EMO[emot].replace(",", "").split()
                )
                text = re.sub(pattern, replacement, str(text))
            return text
        df[self.texts_column] = df[self.texts_column].apply(convert_emoticons_single)   
        return df
    
    def remove_html(self, df):
        def remove_html_single(text):
            html_pattern = re.compile(r'<[a-zA-Z\/][^>]*>')
            return html_pattern.sub(r'', str(text))
        
        df[self.texts_column] = df[self.texts_column].apply(remove_html_single)   
        return df
    
    def convert_chat_words(self, df):
        chat_words_map_dict = {}
        chat_words_list = []
        for line in CHAT_WORDS_STR.split("\n"):
            if line != "":
                cw = line.split("=")[0]
                cw_expanded = line.split("=")[1]
                chat_words_list.append(cw)
                chat_words_map_dict[cw] = cw_expanded
        chat_words_list = set(chat_words_list)

        def chat_words_conversion(text):
            new_text = []
            for w in text.split():
                if w.upper() in chat_words_list:
                    new_text.append(chat_words_map_dict[w.upper()])
                else:
                    new_text.append(w)
            return " ".join(new_text)

        df[self.texts_column] = df[self.texts_column].apply(chat_words_conversion)  
        return df
    
    def remove_mbti_words(self, df): 
        #Vamos remover pra evitar viés.
        def remove_mbti_single(text):
            return ' '.join([word for word in text.split() if word not in self.mbti_types])
        
        df[self.texts_column] = df[self.texts_column].apply(remove_mbti_single)
        return df
    
    def preprocess_simple(self, df):
        df[f"{self.texts_column}_clean"] = df[self.texts_column].copy()
        self.texts_column = f"{self.texts_column}_clean"

        df = self.lowercase(df)
        df = self.remove_whitespaces(df)
        df = self.remove_punctuation(df)
        return df

    def preprocess_complete(self, df): #Entirely based on the paper
        df[f"{self.texts_column}_clean"] = df[self.texts_column].copy()
        self.texts_column = f"{self.texts_column}_clean"

        # 1. Lowercasing
        df = self.lowercase(df)
        
        # 2. Expand contractions
        df = self.expand_contractions(df)
        
        # 3. Remove irrelevant links, mentions (@, #), and URLs
        df = self.remove_links_mentions_urls(df)
        
        # 4. Retain only alphabetic characters (removes non-alphabetic and handles multiple spaces)
        df = self.retain_alphabetic(df)
        
        # 5. Remove MBTI words
        df = self.remove_mbti_words(df)
        
        # 6. Lemmatization
        df = self.lemmatize(df)
        
        # 7. Remove stopwords 
        df = self.remove_stopwords(df)

        return df