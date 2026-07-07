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
        # Como no futuro vamos remover essas siglas aparecendo pra evitar viés, vamos deixar aqui.
        self.mbti_types = {
            'infj', 'infp', 'intj', 'intp', 'isfj', 'isfp', 'istj', 'istp',
            'enfj', 'enfp', 'entj', 'entp', 'esfj', 'esfp', 'estj', 'estp'
        }

    def lowercase(self, df):
        df[self.texts_column] = df[self.texts_column].apply(lambda x: str(x).lower())
        return df
    
    def remove_punctuation(self, df):
        punct_to_remove = string.punctuation.replace('_', '')

        def remove_punctuation_single(text):
            return ' '.join(text.translate(str.maketrans('', '', punct_to_remove)).split())
        
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
            # Alterado de \S+ para [^\s|]+ para evitar que engula o delimitador ||| do dataset.
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
            # Alterado para casar apenas com tags HTML prováveis (começando com letra ou /)
            # Evita deletar coisas como "<3" ou equações "< 5"
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
        df = self.lowercase(df)
        df = self.remove_whitespaces(df)
        df = self.remove_punctuation(df)
        return df

    def preprocess_complete(self, df):
        # 1. Limpezas estruturais de texto 
        df = self.remove_urls(df)
        df = self.remove_html(df)
        df = self.convert_chat_words(df)
        df = self.convert_emoticons(df)
        
        # 2. Padronização de upper e lowercase 
        df = self.lowercase(df)
        
        # 3. Remoção de pontuação e formatação
        df = self.remove_punctuation(df)
        df = self.remove_whitespaces(df)
        
        # 4. Remoção das siglas de MBTI
        df = self.remove_mbti_words(df)
        
        # 5. Lematização pra reduzir os tokens.
        df = self.lemmatize(df)
        df = self.remove_stopwords(df)

        return df