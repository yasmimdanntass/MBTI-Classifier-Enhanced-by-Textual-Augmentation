import pandas as pd
import string
import nltk
import re
import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from pipeline.utils.emoticons_and_expressions import EMOTICONS_EMO, CHAT_WORDS_STR

class Preprocessor:

    def __init__(self, texts_column):
        self.texts_column = texts_column

    def lowercase(self, df):
        df[self.texts_column] = df[self.texts_column].apply(lambda x: x.lower())
        return df
    
    def remove_punctuation(self, df):
        PUNCT_TO_REMOVE = string.punctuation

        def remove_punctuation_single(text):
            return ' '.join(text.translate(str.maketrans('', '', PUNCT_TO_REMOVE)).split())
        
        df[self.texts_column] = df[self.texts_column].apply(remove_punctuation_single)
        return df
    
    def remove_whitespaces(self, df):
        df[self.texts_column] = df[self.texts_column].apply(lambda x: x.strip())
        return df
    
    def remove_stopwords(self, df):
        STOPWORDS = set(stopwords.words('english'))

        def remove_stopwords(text):
            return ' '.join([word for word in str(text).split() if word not in STOPWORDS])
        
        df[self.texts_column] = df[self.texts_column].apply(remove_stopwords)
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
        def remove_urls(text):
            url_pattern = re.compile(r'https?://\S+|www\.\S+')
            return url_pattern.sub(r'', text)
        df[self.texts_column] = df[self.texts_column].apply(remove_urls)   
        return df
    
    def convert_emoticons(self, df):

        def convert_emoticons(text):
            for emot in EMOTICONS_EMO:
                pattern = re.escape(emot) 
                replacement = "_".join(
                    EMOTICONS_EMO[emot].replace(",", "").split()
                )
                text = re.sub(pattern, replacement, text)
            return text
        df[self.texts_column] = df[self.texts_column].apply(convert_emoticons)   
        return df
    
    def remove_html(self, df):
        def remove_html(text):
            html_pattern = re.compile('<.*?>')
            return html_pattern.sub(r'', text)
        
        df[self.texts_column] = df[self.texts_column].apply(remove_html)   
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

        df[self.texts_column] = df[self.texts_column].apply(chat_words_conversion )  
        return df
    
    def remove_stopwords(self, df):
        nltk.download('stopwords')

        STOPWORDS = set(stopwords.words('english')) 

        def remove_stopwords(text):
            return ' '.join([word for word in text.split() if word not in STOPWORDS])

        df[self.texts_column] = df[self.texts_column].apply(remove_stopwords)

        return df
    
    def preprocess_simple(self, df):
        df = self.lowercase(df)
        df = self.remove_whitespaces(df)
        df = self.remove_punctuation(df)

        return df

    def preprocess_complete(self, df):
        df = self.convert_chat_words(df)
        df = self.convert_emoticons(df)
        df = self.lowercase(df)
        df = self.remove_urls(df)
        df = self.remove_html(df)
        df = self.remove_punctuation(df)
        df = self.remove_whitespaces(df)
        df = self.lemmatize(df)
        df = self.remove_stopwords(df)

        return df 

        
