# STEP 1 — Install required libraries
!pip install evaluate transformers sentence-transformers rouge_score -q

# STEP 2 — Imports
import pandas as pd
import evaluate
from sentence_transformers import SentenceTransformer, util

# STEP 3 — Load dataset
df = pd.read_csv("/kaggle/input/finally/generatednew_llm_answers.csv")

# STEP 4 — Initialize metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

# ✅ STEP 5 — Safe compute function with auto-save progress
def compute_text_metrics(ref_list, pred_list, lang_tag):
    bleu_scores, rouge_scores = [], []
    for i, (ref, pred) in enumerate(zip(ref_list, pred_list)):
        ref, pred = str(ref), str(pred)
        try:
            bleu_result = bleu.compute(predictions=[pred], references=[ref])
            rouge_result = rouge.compute(predictions=[pred], references=[ref])
            bleu_scores.append(bleu_result["bleu"])
            rouge_scores.append(rouge_result["rougeL"])
        except Exception as e:
            bleu_scores.append(0)
            rouge_scores.append(0)

        # 🔹 Auto-save progress every 20 rows
        if i % 20 == 0 and i > 0:
            df[f"BLEU_{lang_tag}"] = bleu_scores + [None]*(len(df)-len(bleu_scores))
            df[f"ROUGE-L_{lang_tag}"] = rouge_scores + [None]*(len(df)-len(rouge_scores))
            df.to_csv("/kaggle/working/metrics_partial_save.csv", index=False)
            print(f"✅ Progress saved till row {i} for {lang_tag}")
    return bleu_scores, rouge_scores

# STEP 6 — Compute BLEU and ROUGE (auto-save every 20 rows)
df["BLEU_EN"], df["ROUGE-L_EN"] = compute_text_metrics(df["answer_clean"], df["llm_answer_en"], "EN")
df["BLEU_HI"], df["ROUGE-L_HI"] = compute_text_metrics(df["answer_hi"], df["llm_answer_hi"], "HI")

# STEP 7 — Cosine Similarity with periodic save
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def compute_cosine_similarity(refs, preds, lang_tag):
    scores = []
    for i in range(0, len(refs), 20):  # process in chunks
        ref_chunk = refs[i:i+20]
        pred_chunk = preds[i:i+20]
        emb_ref = model.encode(ref_chunk, convert_to_tensor=True)
        emb_pred = model.encode(pred_chunk, convert_to_tensor=True)
        chunk_scores = util.cos_sim(emb_ref, emb_pred).diagonal().cpu().tolist()
        scores.extend(chunk_scores)

        # 🔹 Auto-save every chunk
        df[f"Cosine_Similarity_{lang_tag}"] = scores + [None]*(len(df)-len(scores))
        df.to_csv("/kaggle/working/metrics_partial_save.csv", index=False)
        print(f"💾 Cosine progress saved till row {i+len(chunk_scores)} for {lang_tag}")
    return scores

df["Cosine_Similarity_EN"] = compute_cosine_similarity(df["answer_clean"].astype(str).tolist(),
                                                      df["llm_answer_en"].astype(str).tolist(), "EN")
df["Cosine_Similarity_HI"] = compute_cosine_similarity(df["answer_hi"].astype(str).tolist(),
                                                      df["llm_answer_hi"].astype(str).tolist(), "HI")

# STEP 8 — Combine metrics into final score
w_bleu, w_rouge, w_cosine = 0.3, 0.3, 0.4

df["Final_Score_EN"] = (
    w_bleu * df["BLEU_EN"] +
    w_rouge * df["ROUGE-L_EN"] +
    w_cosine * df["Cosine_Similarity_EN"]
)

df["Final_Score_HI"] = (
    w_bleu * df["BLEU_HI"] +
    w_rouge * df["ROUGE-L_HI"] +
    w_cosine * df["Cosine_Similarity_HI"]
)

# STEP 9 — Print averages
print("✅ English Metrics")
print("Average BLEU:", df["BLEU_EN"].mean())
print("Average ROUGE-L:", df["ROUGE-L_EN"].mean())
print("Average Cosine:", df["Cosine_Similarity_EN"].mean())
print("Average Final Score:", df["Final_Score_EN"].mean())

print("\n✅ Hindi Metrics")
print("Average BLEU:", df["BLEU_HI"].mean())
print("Average ROUGE-L:", df["ROUGE-L_HI"].mean())
print("Average Cosine:", df["Cosine_Similarity_HI"].mean())
print("Average Final Score:", df["Final_Score_HI"].mean())

# STEP 10 — Final save
df.to_csv("/kaggle/working/metrics_results_bilingual_final.csv", index=False)
print("\n📁 Results saved to: /kaggle/working/metrics_results_bilingual_final.csv") 
