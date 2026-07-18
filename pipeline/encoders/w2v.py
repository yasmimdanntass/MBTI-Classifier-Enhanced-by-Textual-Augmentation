from gensim.models import Word2Vec
import multiprocessing

class Word2VecTokenizer:
    # Valores da Tabela 8 do artigo
    def __init__(
        self,
        vector_size: int = 500,
        window: int = 5,
        min_count: int = 1,
        workers: int = multiprocessing.cpu_count(),
        epochs: int = 10,
        alpha: float = 0.025,
        min_alpha: float = 0.0001,
        sg: int = 0,  # 0 para CBOW (equação 6), 1 para Skip-Gram (equação 7)
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.sg = sg
        self.model = None

    def tokenize_sentence(self, text):
        return text.lower().split()

    # Converte uma lista de sentenças em uma lista de palavras tokenizadas
    def prepare_data(self, texts):
        return [self.tokenize_sentence(t) for t in texts]

    def fit(self, texts):
        tokenized_data = self.prepare_data(texts)
        self.model = Word2Vec(
            sentences=tokenized_data,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
            alpha=self.alpha,
            min_alpha=self.min_alpha,
            sg=self.sg,
        )
        return self

    def get_word_vector(self, word):
        if self.model and word in self.model.wv:
            return self.model.wv[word]
        raise ValueError(
            f"Palavra '{word}' não está no vocabulário. Verifique se o fit foi realizado."
        )

    # Cria um embedding a nível de documento calculando a média dos vetores do documento
    def get_document_embedding(self, text):
        tokens = self.tokenize_sentence(text)
        vectors = [self.model.wv[w] for w in tokens if w in self.model.wv]
        if not vectors:
            import numpy as np
            return np.zeros(self.vector_size)
        import numpy as np
        return np.mean(vectors, axis=0)