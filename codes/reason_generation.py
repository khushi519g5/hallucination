# ============================================================
# Combined hallucination-reason detector (robust, bilingual)
# + Hugging Face login + auto-save + progress bar
# ============================================================

# ---------- CONFIG ----------
HF_TOKEN = "   # <<< REPLACE with your token
INPUT_PATH = "/kaggle/input/reaons/hallucination_reasons_combined.csv"
OUT_PATH = "hallucination_reasonsnew_combined.csv"

# ---------- Try installing compatible packages ----------
import os, sys, subprocess
def safe_run(cmd):
    print("RUN:", cmd)
    res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print("ERR:", res.stderr[:800])
    else:
        print(res.stdout[:300])
    return res.returncode == 0

safe_run('pip install -q "transformers==4.45.2" "sentence-transformers==3.1.1" "huggingface_hub==0.28.1" "accelerate==0.31.0" tqdm')

# ---------- Imports ----------
import pandas as pd, numpy as np, re, traceback
from tqdm import tqdm

try:
    from huggingface_hub import login
    login(token=HF_TOKEN)
    print("✅ Hugging Face login ok")
except Exception as e:
    print("Could not login:", e)

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ---------- Transformers + models ----------
have_hf = have_nli = have_ner = have_st = False
try:
    from transformers import pipeline
    have_hf = True
    print("✅ transformers ok")
except Exception as e:
    print("Transformers import failed:", e)

try:
    from sentence_transformers import SentenceTransformer, util
    have_st = True
    print("✅ sentence-transformers ok")
except Exception as e:
    print("sentence-transformers not available:", e)

ner_en = nli = embed_en = embed_hi = None

if have_hf:
    try:
        ner_en = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
        have_ner = True
        print("✅ English NER loaded")
    except Exception as e:
        print("NER load failed:", e)

    try:
        nli = pipeline("text-classification", model="facebook/bart-large-mnli")
        have_nli = True
        print("✅ NLI (MNLI) loaded")
    except Exception as e:
        print("NLI load failed:", e)

if have_st:
    try:
        embed_en = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embed_hi = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        print("✅ Embedding models loaded")
    except Exception as e:
        print("Embedding load failed:", e)
        have_st = False

# ---------- Helpers ----------
HINDI_STOPWORDS = {"है","हैं","था","थे","थी","और","को","के","का","पर","से","में","कि","यह","ये","वह","जो","एक","यहाँ","लेकिन","भी"}

def normalize_text(s):
    if pd.isna(s): return ""
    return re.sub(r"\s+", " ", str(s)).strip()

def tokens_en(t): 
    t = normalize_text(t).lower()
    return [w for w in re.findall(r"\w+", t) if w not in ENGLISH_STOP_WORDS]

def tokens_hi(t):
    t = normalize_text(t)
    devan = re.findall(r"[\u0900-\u097F]+", t)
    parts = devan if devan else re.findall(r"\w+", t)
    return [w for w in parts if w not in HINDI_STOPWORDS]

def entities_from_text_en(text):
    if not text.strip(): return set()
    if have_ner and ner_en:
        try:
            ents = ner_en(text)
            return {e.get("word","").lower() for e in ents if e.get("word")}
        except: pass
    caps = set([m.group(0).lower() for m in re.finditer(r"\b[A-Z][a-z]+\b", text)])
    return caps.union(set(tokens_en(text)))

def entities_from_text_hi(text):
    dev = set(re.findall(r"[\u0900-\u097F]{2,}", text))
    return dev.union(set(tokens_hi(text)))

def number_mismatch(a,b):
    an, bn = set(re.findall(r"\d+", str(a))), set(re.findall(r"\d+", str(b)))
    if not an and not bn: return False, None
    if an == bn: return False, None
    return True, {"numbers_in_true": list(bn), "numbers_in_llm": list(an)}

def token_overlap(a,b, lang='en'):
    ta = set(tokens_en(a)) if lang=='en' else set(tokens_hi(a))
    tb = set(tokens_en(b)) if lang=='en' else set(tokens_hi(b))
    if not ta and not tb: return 0
    return len(ta & tb)/len(ta | tb)

def semantic_similarity(a,b, lang='en'):
    if not have_st: return None
    try:
        m = embed_en if lang=='en' else embed_hi
        ea, eb = m.encode(a, convert_to_tensor=True), m.encode(b, convert_to_tensor=True)
        return float(util.cos_sim(ea, eb))
    except: return None

def generate_reason(row, lang='en'):
    llm = str(row.get(f"llm_answer_{lang}", "") or "")
    gold = str(row.get("answer" if lang=='en' else f"answer_{lang}", "") or "")
    bleu = row.get(f"bleu_{lang}", None)
    rouge = row.get(f"rouge-l_{lang}", None)
    cos = row.get(f"cosine_similarity_{lang}", None)
    hall = str(row.get(f"hallucination_{lang}", "")).lower()

    # entity + numeric mismatch
    ents_fn = entities_from_text_en if lang=='en' else entities_from_text_hi
    extra = ents_fn(llm) - ents_fn(gold)
    missing = ents_fn(gold) - ents_fn(llm)
    num_mis, num_info = number_mismatch(gold, llm)

    # semantic similarity
    sim = None
    try: sim = float(cos)
    except: sim = semantic_similarity(gold, llm, lang)

    # NLI contradiction check (truncate to avoid warning)
    contradiction = None
    if have_nli:
        try:
            p, h = gold[:4000], llm[:4000]
            out = nli(f"{p} </s></s> {h}")
            if isinstance(out, list) and "label" in out[0]:
                contradiction = out[0]["label"].lower()
        except: pass

    reasons = []
    if contradiction:
        if "contradict" in contradiction:
            reasons.append("Contradicts ground truth")
        elif "neutral" in contradiction:
            reasons.append("No clear entailment")
        elif "entailment" in contradiction:
            reasons.append("Entails ground truth")

    if extra and missing:
        reasons.append(f"Wrong entities — added {list(extra)[:3]} missed {list(missing)[:3]}")
    elif extra:
        reasons.append(f"Extra entities {list(extra)[:3]}")
    elif missing:
        reasons.append(f"Missed entities {list(missing)[:3]}")
    if num_mis:
        reasons.append(f"Numeric mismatch {num_info}")
    if sim is not None:
        if sim < 0.45: reasons.append(f"Low semantic sim ({sim:.2f})")
        elif sim < 0.65: reasons.append(f"Medium sim ({sim:.2f})")
        else: reasons.append(f"High sim ({sim:.2f})")
    if bleu is not None:
        try:
            b = float(bleu)
            if b < 0.25: reasons.append(f"Low BLEU ({b:.2f})")
            elif b < 0.5: reasons.append(f"Moderate BLEU ({b:.2f})")
            else: reasons.append(f"High BLEU ({b:.2f})")
        except: pass
    if rouge is not None:
        try:
            r = float(rouge)
            if r < 0.25: reasons.append(f"Low ROUGE ({r:.2f})")
            elif r < 0.5: reasons.append(f"Moderate ROUGE ({r:.2f})")
            else: reasons.append(f"High ROUGE ({r:.2f})")
        except: pass
    if hall in {"1","yes","true","y","hallucinated"}:
        reasons.append("Human label: hallucination flagged")

    return " ; ".join(reasons[:8])

# ---------- Load + Process with Progress ----------
df = pd.read_csv(INPUT_PATH)
df.columns = [c.lower().strip() for c in df.columns]
for col in ["llm_answer_en","answer","llm_answer_hi","answer_hi"]:
    if col not in df.columns: df[col] = ""

print(f"Processing {len(df)} rows...\n")

# Progress bar + autosave
reasons_en, reasons_hi = [], []
for i, row in tqdm(df.iterrows(), total=len(df)):
    r_en = generate_reason(row, 'en')
    r_hi = generate_reason(row, 'hi')
    reasons_en.append(r_en)
    reasons_hi.append(r_hi)

    if i % 100 == 0 and i > 0:
        df.loc[:i, "hallucination_reason_en"] = reasons_en
        df.loc[:i, "hallucination_reason_hi"] = reasons_hi
        df.to_csv(OUT_PATH, index=False)
        print(f"💾 Auto-saved progress at row {i}")

df["hallucination_reason_en"] = reasons_en
df["hallucination_reason_hi"] = reasons_hi
df.to_csv(OUT_PATH, index=False)
print("✅ Final saved:", OUT_PATH)


















