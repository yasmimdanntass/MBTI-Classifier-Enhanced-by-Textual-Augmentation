import ollama
import pandas as pd
import re
import os

class SLMAugmenter:
    def __init__(self, texts_column, label_column, model_name='qwen2.5'):
        self.texts_column = texts_column
        self.label_column = label_column
        self.model_name = model_name
        
        try:
            print(f"Downloading '{self.model_name}'... ")
            ollama.pull(self.model_name)
        except Exception as e:
            print(f"Erro ao tentar baixar o modelo automaticamente: {e}")

    def _paraphrase_with_ollama(self, original_text):
        prompt = f"""You are a linguistics expert. 
        Rewrite the text below maintaining the exact same meaning, tone of voice, and MBTI personality characteristics of the original author. 
        Use different words, but do not add new information.
        Respond ONLY with the paraphrased text. Do NOT include any introductions, explanations, or conversational filler like "Here is the text".
        
        Text: {original_text}"""

        try:
            response = ollama.chat(model=self.model_name, messages=[
                {'role': 'user', 'content': prompt}
            ],
            options={
                'temperature': 0.6, 
                'top_p': 0.95,   
                'top_k': 50    
            })

            texto_bruto = response['message']['content'].strip()
            
            pattern = re.compile(r'^(here is|here\'s|sure|i have rewritten|below is|here are|certainly).*?(\n\n|\n|:)', re.IGNORECASE)
            
            while pattern.match(texto_bruto):
                texto_bruto = pattern.sub('', texto_bruto, count=1).strip()
            
            return texto_bruto
        
        except Exception as e:
            print(f"Error communicating with Ollama: {e}")
            return None
        
    def augment_minority_classes(self, df, sample_size=None, checkpoint_file='data/augmentation_checkpoint.csv'):
        print(f"Starting local augmentation using {self.model_name}...")
        
        if 'augmented_posts' not in df.columns:
            df['augmented_posts'] = pd.NA
            
        mask = df[self.label_column].str.contains('E|S', na=False)
        df_minoritario = df[mask]
        
        if sample_size:
            df_minoritario = df_minoritario.head(sample_size)

        dados_checkpoint = []
        indices_ja_processados = set()
        if checkpoint_file and os.path.exists(checkpoint_file):
            print(f"Loading checkpoint on '{checkpoint_file}'...")
            df_checkpoint = pd.read_csv(checkpoint_file)
            
            if 'original_index' in df_checkpoint.columns:
                for _, row_cp in df_checkpoint.iterrows():
                    idx = row_cp['original_index']
                    texto_salvo = row_cp['augmented_posts']
                    df.at[idx, 'augmented_posts'] = texto_salvo
                    
                    dados_checkpoint.append({'original_index': idx, 'augmented_posts': texto_salvo})
                    indices_ja_processados.add(idx)
                    
            print(f"{len(indices_ja_processados)} textos já processados encontrados. Retomando de onde parou...")

        for index, row in df_minoritario.iterrows():
            if index in indices_ja_processados:
                continue

            tipo_mbti = row[self.label_column]
            texto_base = str(row[self.texts_column])
            
            frases = texto_base.split('|||')
            chunks = ['|||'.join(frases[i:i+5]) for i in range(0, len(frases), 5)]
            
            textos_gerados = []
            
            for chunk in chunks:
                if chunk.strip():
                    gerado = self._paraphrase_with_ollama(chunk)
                    if gerado:
                        textos_gerados.append(gerado)
            
            if textos_gerados:
                texto_final = '|||'.join(textos_gerados)
                
                df.at[index, 'augmented_posts'] = texto_final
                
                dados_checkpoint.append({'original_index': index, 'augmented_posts': texto_final})
                print(f"Generated with success: {tipo_mbti} (index original: {index})")

                if checkpoint_file:
                    pd.DataFrame(dados_checkpoint).to_csv(checkpoint_file, index=False)

        print("Augmentation finalizado com sucesso!")
        return df