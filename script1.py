import pandas as pd
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
import csv

# --- Step 1: Load paragraph ---
with open("para.txt", "r", encoding="utf-8") as f:
    paragraph = f.read().strip()

# --- Step 2: Load Knowledge Base ---
kb_file = "onto.csv"
kb_entries = []

with open(kb_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, quotechar='"')
    for i, row in enumerate(reader, start=1):
        if "name" in row and "type" in row and "description" in row:
            kb_entries.append(row)
        else:
            print(f"[Warning] Skipping malformed line {i}: {row}")

if not kb_entries:
    raise ValueError("Knowledge base is empty or malformed!")

kb_df = pd.DataFrame(kb_entries)

# --- Step 3: Load Model ---
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Step 4: Encode paragraph and KB entries (name + description) ---
paragraph_embedding = model.encode([paragraph], convert_to_tensor=True)
kb_df['text'] = kb_df['name'] + ". " + kb_df['description']
kb_embeddings = model.encode(kb_df['text'].tolist(), convert_to_tensor=True)

# --- Step 5: Compute similarity per type ---
threshold = 0.6
results = defaultdict(list)

for t in ['tool', 'technique', 'domain']:
    # Filter KB by type
    kb_filtered = kb_df[kb_df['type'] == t]
    if kb_filtered.empty:
        continue

    # Compute similarity
    filtered_embeddings = kb_embeddings[kb_filtered.index.tolist()]
    scores = util.cos_sim(paragraph_embedding, filtered_embeddings)[0]

    # Sort by score descending
    scored_entries = sorted(zip(kb_filtered['name'], scores), key=lambda x: x[1], reverse=True)

    # Pick top 3 per type with score above threshold
    for name, score in scored_entries:
        if score.item() >= threshold and name not in results[t]:
            results[t].append(name)
        if len(results[t]) >= 3:
            break

# --- Step 6: Ensure minimum 3 entries overall ---
all_detected = results.get('tool', []) + results.get('technique', []) + results.get('domain', [])
if len(all_detected) < 3:
    for _, row in kb_df.iterrows():
        if row['name'] not in all_detected:
            all_detected.append(row['name'])
        if len(all_detected) >= 3:
            break

# --- Step 7: Print Results ---
print("Detected Methodologies/Tools/Domains (top 3 per type):")
for t in ['tool', 'technique', 'domain']:
    if results.get(t):
        print(f"{t.capitalize()}: {', '.join(results[t][:3])}")

print("\nTop aggregated entries (minimum 3):")
for entry in all_detected[:3]:
    print(f"- {entry}")

