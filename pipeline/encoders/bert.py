import torch
from transformers import BertTokenizer, BertModel
import numpy as np

class BertVectorizer:
    def __init__(
        self, 
        model_name: str = 'bert-base-uncased',
        batch_size: int = 16,
        max_length: int = 512
    ):
        """
        Inicializa o vetorizador do BERT com parâmetros configuráveis.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Carregamento do tokenizador e modelo
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name)
        
        # Configuração de dispositivo (GPU se disponível, senão CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval() # Modo de inferência (sem atualização de pesos)

    def fit(self, texts=None):
        """
        Para o BERT pré-treinado, não há treinamento ou aprendizado de vocabulário
        nesta etapa. Mantido para compatibilidade com a interface do projeto.
        """
        return self

    def transform(self, texts):
        """
        Converte uma lista de textos nos embeddings do token [CLS] gerados pelo BERT.
        """
        all_embeddings = []
        
        # Processamento em lotes para otimização de memória
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenização e formatação para o BERT
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            input_ids = encoded_input['input_ids'].to(self.device)
            attention_mask = encoded_input['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                
            # Extrai o embedding da última camada oculta correspondente ao token [CLS] (índice 0)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.extend(cls_embeddings.cpu().numpy())
            
        return np.array(all_embeddings)

    def fit_transform(self, texts):
        """
        Executa o fit (pass-through) e o transform em sequência.
        """
        return self.fit(texts).transform(texts) 