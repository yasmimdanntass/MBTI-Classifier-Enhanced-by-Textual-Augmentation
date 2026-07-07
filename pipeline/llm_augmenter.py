import ollama
import pandas as pd

class QwenAugmenter:
    def __init__(self, texts_column, label_column, model_name='qwen2.5'):
        self.texts_column = texts_column
        self.label_column = label_column
        self.model_name = model_name
        
        # Download automático do modelo caso ele não exista na máquina
        try:
            print(f"Verificando/Baixando o modelo '{self.model_name}'... Isso pode demorar na 1ª vez se você ainda não baixou.")
            ollama.pull(self.model_name)
        except Exception as e:
            print(f"Erro ao tentar baixar o modelo automaticamente: {e}")

    def _paraphrase_with_ollama(self, original_text):
        import re
        
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
                'temperature': 0.6, # Fiz um ajuste pessoal pra reduzir a temperatura que estava no artigo, porque o llama tende a ser mais criativo.
                'top_p': 0.95,   
                'top_k': 50    # Aqui mantive igual.
            })

            texto_bruto = response['message']['content'].strip()
            
            # Usa regex para remover introduções indesejadas (case-insensitive)
            # Ex: "Here is the rewritten text:", "Sure! Here is...", "Below is..."
            pattern = re.compile(r'^(here is|here\'s|sure|i have rewritten|below is|here are|certainly).*?(\n\n|\n|:)', re.IGNORECASE)
            
            # Continua removendo o padrão enquanto ele existir no início do texto
            while pattern.match(texto_bruto):
                texto_bruto = pattern.sub('', texto_bruto, count=1).strip()
            
            return texto_bruto
        
        except Exception as e:
            print(f"Error communicating with Ollama: {e}")
            return None
        
    def augment_minority_classes(self, df, sample_size=None, checkpoint_file='data/augmentation_checkpoint.csv'):
        import os
        print(f"Iniciando augmentation local com {self.model_name}...")
        
        mask = df[self.label_column].str.contains('E|S', na=False)
        df_minoritario = df[mask]
        
        if sample_size:
            df_minoritario = df_minoritario.head(sample_size)

        novos_dados = []
        indices_ja_processados = set()

        if checkpoint_file and os.path.exists(checkpoint_file):
            print(f"Carregando checkpoint salvo em '{checkpoint_file}'...")
            df_checkpoint = pd.read_csv(checkpoint_file)
            if 'original_index' in df_checkpoint.columns:
                indices_ja_processados = set(df_checkpoint['original_index'].values)
            novos_dados = df_checkpoint.to_dict('records')
            print(f"{len(indices_ja_processados)} textos já processados encontrados. Retomando de onde parou...")

        for index, row in df_minoritario.iterrows():
            if index in indices_ja_processados:
                continue

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
                nova_linha['original_index'] = index
                novos_dados.append(nova_linha)
                print(f"Gerado sucesso: {tipo_mbti} (index original: {index})")

                # Salva o checkpoint no disco a cada texto gerado com sucesso
                if checkpoint_file:
                    pd.DataFrame(novos_dados).to_csv(checkpoint_file, index=False)

        if novos_dados:
            df_gerado = pd.DataFrame(novos_dados)
            # Remove a coluna auxiliar antes de juntar ao dataset original
            if 'original_index' in df_gerado.columns:
                df_gerado = df_gerado.drop(columns=['original_index'])
                
            if 'is_synthetic' not in df.columns:
                df['is_synthetic'] = False
            return pd.concat([df, df_gerado], ignore_index=True)
        return df