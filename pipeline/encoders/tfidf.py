from sklearn.feature_extraction.text import TfidfVectorizer

# Coincide com as equações 4 e 5 do artigo
class TFIDFTokenizer:
    # Valores adaptados da Tabela 7
    def __init__(
        self,
        max_features: int = 10000,
        sublinear_tf: bool = True,
        smooth_idf: bool = True,
        norm: str = "l2",
        min_df: float = 2,
        max_df: float = 0.8,
        ngram_range: tuple = (1, 2),
        stop_words: str = "english",
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            sublinear_tf=sublinear_tf,
            smooth_idf=smooth_idf,
            norm=norm,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            stop_words=stop_words,
        )

    def fit(self, texts):
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    @property
    def feature_names(self):
        return self.vectorizer.get_feature_names_out()