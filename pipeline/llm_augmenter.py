import ollama
import pandas as pd

class LlamaAugmenter:
    def __init__(self, texts_column, label_column, model_name='llama3'):
        self.texts_column = texts_column
        self.label_column = label_column
        self.model_name = model_name

    def _paraphrase_with_ollama(self, original_text):
        prompt = f"""You are a linguistics expert. 
        Rewrite the text below maintaining the exact same meaning, tone of voice, and MBTI personality characteristics of the original author. 
        Use different words, but do not add new information.
        Respond ONLY with the paraphrased text, without any introductions or conversational filler.
        
        Text: {original_text}"""

        try:
            response = ollama.chat(model=self.model_name, messages=[
                {'role': 'user', 'content': prompt}
            ],
            options={
                'temperature': 0.6, # Fiz um ajuste pessoal pra reduzir a temperatura que estava no artigo, porque o llama tende a ser mais criativo.
                'top_p': 0.95,   
                'top_k': 50    # Aqui mantive igual.
            })
            return response['message']['content']
        except Exception as e:
            print(f"Error communicating with Ollama: {e}")
            return None
        
    def augment_minority_classes(self, df, sample_size=None):
        print(f"Iniciando augmentation local com {self.model_name}...")
        
        mask = df[self.label_column].str.contains('E|S', na=False)
        df_minoritario = df[mask]
        
        if sample_size:
            df_minoritario = df_minoritario.head(sample_size)

        novos_dados = []

        for index, row in df_minoritario.iterrows():
            tipo_mbti = row[self.label_column]
            texto_base = str(row[self.texts_column])
            
            # 1. Quebra o texto usando o delimitador do Kaggle
            frases = texto_base.split('|||')
            
            # 2. Agrupa de 5 em 5 frases
            chunks = ['|||'.join(frases[i:i+5]) for i in range(0, len(frases), 5)]
            
            textos_gerados = []
            
            # 3. Parafraseia cada bloco
            for chunk in chunks:
                if chunk.strip():
                    gerado = self._paraphrase_with_ollama(chunk)
                    if gerado:
                        textos_gerados.append(gerado)
            
            # 4. Junta tudo de novo usando o mesmo delimitador
            if textos_gerados:
                texto_final = '|||'.join(textos_gerados)
                
                nova_linha = row.copy()
                nova_linha[self.texts_column] = texto_final
                nova_linha['is_synthetic'] = True 
                novos_dados.append(nova_linha)
                print(f"Gerado sucesso: {tipo_mbti}")

        if novos_dados:
            df_gerado = pd.DataFrame(novos_dados)
            if 'is_synthetic' not in df.columns:
                df['is_synthetic'] = False
            return pd.concat([df, df_gerado], ignore_index=True)
        return df