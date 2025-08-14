import pandas as pd
import warnings
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
import re

# -------------------- Step 1: KB Loader --------------------
def load_kb(path):
    try:
        df = pd.read_csv(
            path,
            quotechar='"',
            on_bad_lines="skip",  # skip malformed rows
            engine="python"       # forgiving parser
        )

        # Normalize column names
        df.columns = [c.strip().lower() for c in df.columns]

        # Ensure required columns exist
        required_cols = {"name", "type", "description"}
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(f"CSV must have columns: {required_cols}")

        # Remove duplicate header rows and empty entries
        df = df.dropna(subset=["name", "type", "description"])
        df = df[df["name"].str.lower() != "name"]
        df.drop_duplicates(subset=["name", "type", "description"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df
    except Exception as e:
        raise RuntimeError(f"Error loading KB: {e}")

# -------------------- Step 2: Paragraph Loader --------------------
def load_paragraph(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    sentences = [s.strip() for s in re.split(r'[.?!]', text) if s.strip()]
    return text, sentences

# -------------------- Step 3: Unsupervised Extraction --------------------
def unsupervised_extraction(text):
    keywords = set()
    # Very simple noun phrase matcher (could use spaCy for better)
    for match in re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', text):
        if len(match) > 3:
            keywords.add(match)
    return list(keywords)

# -------------------- Step 4: Semantic KB Matching --------------------
def semantic_match(sentences, kb_df, threshold=0.65):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    kb_embeddings = model.encode(kb_df['description'].tolist(), convert_to_tensor=True)

    cosine_scores = util.cos_sim(sentence_embeddings, kb_embeddings)

    detected = []
    for i, sentence in enumerate(sentences):
        for j, kb_entry in kb_df.iterrows():
            score = cosine_scores[i][j].item()
            if score >= threshold:
                detected.append({
                    "name": kb_entry['name'],
                    "type": kb_entry['type'],
                    "score": score
                })
    return detected

# -------------------- Step 5: Aggregate Results --------------------
def aggregate_results(detected, min_results=3):
    results = defaultdict(list)
    for item in sorted(detected, key=lambda x: x['score'], reverse=True):
        if item['name'] not in results[item['type']]:
            results[item['type']].append(item['name'])

    all_detected = []
    for t in results:
        all_detected.extend(results[t])

    # Ensure minimum number of outputs
    if len(all_detected) < min_results:
        all_detected = (all_detected + ["<None>"] * min_results)[:min_results]

    return results, all_detected

# -------------------- Step 6: Main --------------------
if __name__ == "__main__":
    # Load KB
    kb_df = load_kb("onto.csv")

    # Load paragraph
    paragraph, sentences = load_paragraph("para.txt")

    # KB-based matching
    detected_kb = semantic_match(sentences, kb_df)

    # Unsupervised keyword extraction
    extracted_keywords = unsupervised_extraction(paragraph)

    # Aggregate results
    results, all_detected = aggregate_results(detected_kb, min_results=3)

    # -------------------- Output --------------------
    print("Detected Methodologies/Tools/Domains (top 3 per type):")
    for t in ['tool', 'technique', 'domain']:
        if t in results and results[t]:
            print(f"{t.capitalize()}: {', '.join(results[t][:3])}")

    print("\nTop aggregated entries (minimum 3):")
    for entry in all_detected:
        print(f"- {entry}")

    print("\n[Unsupervised Extraction] Possible Keywords/Concepts:")
    print(", ".join(extracted_keywords))

