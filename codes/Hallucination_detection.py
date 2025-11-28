import numpy as np

# Set thresholds (you can tune them based on your dataset)
BLEU_THRESHOLD = 0.2
ROUGE_THRESHOLD = 0.2
COSINE_THRESHOLD = 0.5
FINAL_THRESHOLD = 0.25

def detect_hallucination(row, lang="EN"):
    bleu = row[f"BLEU_{lang}"]
    rouge = row[f"ROUGE-L_{lang}"]
    cosine = row[f"Cosine_Similarity_{lang}"]
    final = row[f"Final_Score_{lang}"]

    # Check hallucination levels
    if final < FINAL_THRESHOLD or cosine < COSINE_THRESHOLD:
        return " Hallucinated"
    elif final < (FINAL_THRESHOLD + 0.1):
        return " Partially Accurate"
    else:
        return "Accurate"

# Apply detection for English and Hindi
df["Hallucination_EN"] = df.apply(lambda r: detect_hallucination(r, "EN"), axis=1)
df["Hallucination_HI"] = df.apply(lambda r: detect_hallucination(r, "HI"), axis=1)

# Count results
print("\n🧮 English Answer Distribution:")
print(df["Hallucination_EN"].value_counts())

print("\n🧮 Hindi Answer Distribution:")
print(df["Hallucination_HI"].value_counts())

# Save results
df.to_csv("/kaggle/working/hallucination_detected.csv", index=False)
print("\n📁 Results saved to: /kaggle/working/hallucination_detected.csv")
