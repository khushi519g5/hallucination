# STEP 1 — Install dependencies
!pip install -q transformers accelerate torch sentencepiece pandas tqdm

# STEP 2 — Login to Hugging Face
from huggingface_hub import login
login("#REPLACE WITH YOUR OWN TOKEN")  # 🔑 replace with your token

# STEP 3 — Imports & setup
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
from transformers import pipeline
import torch
from tqdm import tqdm

# STEP 4 — Device setup
device = 0 if torch.cuda.is_available() else -1
print(f"✅ Using {'GPU' if device == 0 else 'CPU'} for inference")

# STEP 5 — Load BLOOM model
pipe = pipeline(
    "text-generation",
    model="bigscience/bloom-560m",
    device=device
)

# STEP 6 — Load dataset
file_path = "/kaggle/input/finalkind/translated_dataset_fully_translated (1).csv"
df = pd.read_csv(file_path)

# Ensure required columns exist
if "question_clean" not in df.columns or "question_hi" not in df.columns:
    raise ValueError("CSV must contain 'question_clean' and 'question_hi' columns!")

# ✅ Add new columns for LLM answers
df["llm_answer_en"] = df.get("llm_answer_en", "")
df["llm_answer_hi"] = df.get("llm_answer_hi", "")

# STEP 7 — Generate LLM answers (without overwriting trusted ones)
save_interval = 20
output_path = "/kaggle/working/generated_llm_answers.csv"

for i in tqdm(range(len(df))):
    # English answer
    if pd.isna(df.loc[i, "llm_answer_en"]) or df.loc[i, "llm_answer_en"] == "":
        q_en = df.loc[i, "question_clean"]
        if isinstance(q_en, str) and len(q_en.strip()) > 0:
            prompt = f"Q: {q_en}\nA:"
            ans_en = pipe(prompt, max_new_tokens=80, do_sample=True, temperature=0.7, top_p=0.9)
            df.loc[i, "llm_answer_en"] = ans_en[0]["generated_text"].split("A:")[-1].strip()

    # Hindi answer
    if pd.isna(df.loc[i, "llm_answer_hi"]) or df.loc[i, "llm_answer_hi"] == "":
        q_hi = df.loc[i, "question_hi"]
        if isinstance(q_hi, str) and len(q_hi.strip()) > 0:
            prompt_hi = f"प्रश्न: {q_hi}\nउत्तर:"
            ans_hi = pipe(prompt_hi, max_new_tokens=80, do_sample=True, temperature=0.7, top_p=0.9)
            df.loc[i, "llm_answer_hi"] = ans_hi[0]["generated_text"].split("उत्तर:")[-1].strip()

    # Save progress every few rows
    if i % save_interval == 0:
        df.to_csv(output_path, index=False)

# Final save
df.to_csv(output_path, index=False)
print(f"✅ LLM answers generated and saved to: {output_path}")
