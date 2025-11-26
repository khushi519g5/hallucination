!pip install deep-translator swifter tqdm

import pandas as pd
from deep_translator import GoogleTranslator
import swifter
import time
import os
from tqdm import tqdm
import re

# 1️⃣ Load dataset
df = pd.read_csv('/kaggle/input/medical/medquad_qna_with_weights.csv')

# 2️⃣ Preprocessing function
def preprocess_text(text):
    text = str(text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters except basic punctuation
    text = re.sub(r'[^A-Za-z0-9\s,.?!]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Lowercase (optional)
    text = text.lower()
    return text

# Apply preprocessing to questions and answers
df['question_clean'] = df['question'].swifter.apply(preprocess_text)
df['answer_clean'] = df['answer'].swifter.apply(preprocess_text)

# 3️⃣ Progress file
progress_file = '/kaggle/working/translated_progress.parquet'

# 4️⃣ Resume if exists
if os.path.exists(progress_file):  
    df_translated = pd.read_parquet(progress_file)
    print("✅ Resuming from saved progress...")
else:
    df_translated = df.copy()
    df_translated['question_hi'] = ''
    df_translated['answer_hi'] = ''

# 5️⃣ Translation function with retry
def translate_text(text, target='hi'):
    text = str(text)
    for _ in range(3):
        try:
            return GoogleTranslator(source='auto', target=target).translate(text)
        except:
            time.sleep(0.5)
    return text

# 6️⃣ Batch size for safe saving
batch_size = 500

# Use tqdm for progress bar
for start in tqdm(range(0, len(df_translated), batch_size), desc="Translating Batches"):
    end = min(start + batch_size, len(df_translated))
    subset = df_translated.loc[start:end]

    # Translate only empty rows
    if subset['question_hi'].eq('').all():
        df_translated.loc[start:end, 'question_hi'] = subset['question_clean'].swifter.apply(lambda x: translate_text(x, 'hi'))
        df_translated.loc[start:end, 'answer_hi'] = subset['answer_clean'].swifter.apply(lambda x: translate_text(x, 'hi'))

        # Save progress after each batch
        df_translated.to_parquet(progress_file, index=False)
        time.sleep(1)  # short cooldown to avoid being blocked

# 7️⃣ Final save
df_translated.to_parquet('/kaggle/working/translated_dataset.parquet', index=False)
print("✅ Preprocessing + Translation completed and saved!")
