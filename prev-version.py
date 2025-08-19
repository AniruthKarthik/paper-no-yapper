#!/usr/bin/env python3
import os
import re
import csv
import math
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, Counter
from difflib import SequenceMatcher

try:
    from fuzzywuzzy import fuzz
except ImportError:
    fuzz = None

try:
    import nltk
    from nltk.corpus import wordnet as wn
    nltk_available = True
    try:
        wn.ensure_loaded()
    except:
        nltk.download('wordnet')
        wn.ensure_loaded()
except ImportError:
    wn = None
    nltk_available = False

# OUTPUT DIR
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# LOG FILE
skipped_log_path = os.path.join(output_dir, "skipped_rows.log")
skipped_log = open(skipped_log_path, "w", encoding="utf-8")

def detect_header(filepath, sample_bytes=2048):
    import csv as csvmod
    with open(filepath, newline='', encoding="utf-8") as csvfile:
        sample = csvfile.read(sample_bytes)
        csvfile.seek(0)
        return csvmod.Sniffer().has_header(sample)

def load_papers(filepath):
    """Load papers.csv, fix misaligned rows by padding/truncating to match header length."""
    print("Loading papers from:", filepath)
    has_header = detect_header(filepath)
    print(f" - Header detected: {has_header}")

    cleaned_rows = []
    with open(filepath, newline='', encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
        if not rows:
            raise ValueError("papers.csv is empty!")
        
        if has_header:
            header = [h.strip() for h in rows[0]]
            data_rows = rows[1:]
        else:
            header = [f"col{i}" for i in range(1, len(rows[0]) + 1)]
            data_rows = rows
        
        expected_len = len(header)
        for idx, row in enumerate(data_rows, start=1 if has_header else 0):
            if len(row) < expected_len:
                row += [""] * (expected_len - len(row))
                skipped_log.write(f"Padded row {idx}: {row}\n")
            elif len(row) > expected_len:
                skipped_log.write(f"Truncated row {idx}: {row}\n")
                row = row[:expected_len]
            cleaned_rows.append(row)
    
    df = pd.DataFrame(cleaned_rows, columns=header)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def map_columns(df):
    print("Mapping columns...")
    col_map = {}
    expected = {
        'title': ['title', 'paper title', 'article title'],
        'abstract': ['abstract', 'summary'],
        'year': ['year', 'pubyear', 'publication year', 'date'],
        'authors': ['author', 'authors'],
        'keywords': ['keyword', 'keywords', 'key words'],
        'link': ['link', 'url', 'doi']
    }
    columns = list(df.columns)
    for field, keywords in expected.items():
        for col in columns:
            name = col.lower()
            if any(key in name for key in keywords):
                col_map[field] = col
                break
        if field not in col_map:
            col_map[field] = None
            print(f"Warning: No match for {field}")
    return col_map

def load_ontology(filepath):
    print("Loading ontology from:", filepath)
    onto_df = pd.read_csv(filepath, quotechar='"', on_bad_lines='skip', engine="python")
    onto_df.columns = [c.strip().lower() for c in onto_df.columns]
    term_col = [c for c in onto_df.columns if 'term' in c or 'name' in c][0]
    cat_col = [c for c in onto_df.columns if 'category' in c or 'type' in c][0]
    ontology = defaultdict(list)
    for _, row in onto_df.iterrows():
        term = str(row[term_col]).strip()
        cat = str(row[cat_col]).strip().capitalize()
        if pd.isna(term) or pd.isna(cat):
            continue
        ontology[cat].append(term)
    for cat in ontology:
        ontology[cat] = list(set(ontology[cat]))
    return ontology

def get_synonyms(term):
    syns = set()
    if wn:
        for syn in wn.synsets(term):
            for lemma in syn.lemmas():
                syn_term = lemma.name().replace('_', ' ')
                syns.add(syn_term.lower())
    return syns

def match_terms(text, terms):
    matches = Counter()
    lower_text = text.lower()
    for term in terms:
        term_lower = term.lower()
        if re.search(r'\b' + re.escape(term_lower) + r'\b', lower_text):
            matches[term] += 1
            continue
        if fuzz and fuzz.partial_ratio(term_lower, lower_text) >= 90:
            matches[term] += 1
            continue
        elif not fuzz:
            if SequenceMatcher(None, term_lower, lower_text).ratio() > 0.9:
                matches[term] += 1
                continue
        if wn:
            for syn in get_synonyms(term_lower):
                if re.search(r'\b' + re.escape(syn) + r'\b', lower_text):
                    matches[term] += 1
                    break
    return matches

def analyze_papers(df, col_map, ontology):
    stats = {cat: defaultdict(lambda: {
        'count': 0, 'first_year': math.inf, 'first_title': '', 'first_link': '',
        'year_trend': Counter(), 'authors': Counter(), 'co_other': Counter(),
        'max_paper_count': 0, 'top_paper': ''
    }) for cat in ['Domain', 'Methodology', 'Technique']}
    paper_results = []
    total = len(df)
    for idx, row in df.iterrows():
        try:
            title = str(row[col_map['title']]) if col_map['title'] else ''
            abstract = str(row[col_map['abstract']]) if col_map['abstract'] else ''
            keywords = str(row[col_map['keywords']]) if col_map['keywords'] else ''
            year = int(row[col_map['year']]) if col_map['year'] and str(row[col_map['year']]).isdigit() else None
            authors = []
            if col_map['authors']:
                authors = [a.strip() for a in str(row[col_map['authors']]).replace("\t"," ").split(",") if a.strip()]
            authors_str = "; ".join(authors)
            link = str(row[col_map['link']]) if col_map['link'] else ''
        except Exception as e:
            skipped_log.write(f"Row {idx} error: {e}\n")
            continue

        full_text = " ".join([title, abstract, keywords])
        domain_matches = match_terms(full_text, ontology.get('Domain', []))
        meth_matches   = match_terms(full_text, ontology.get('Methodology', []))
        tech_matches   = match_terms(full_text, ontology.get('Technique', []))
        top_domains = [t for t, _ in domain_matches.most_common(3)]
        top_methods = [t for t, _ in meth_matches.most_common(3)]
        top_techs   = [t for t, _ in tech_matches.most_common(3)]

        paper_results.append({
            'Title': title, 'Authors': authors_str, 'Year': year, 'Link': link,
            'TopDomains': "; ".join(top_domains),
            'TopMethodologies': "; ".join(top_methods),
            'TopTechniques': "; ".join(top_techs)
        })

        for term in top_domains:
            s = stats['Domain'][term]
            s['count'] += 1
            if year:
                s['year_trend'][year] += 1
                if year < s['first_year']:
                    s['first_year'] = year
                    s['first_title'] = title
                    s['first_link'] = link
            for a in authors:
                s['authors'][a] += 1
        for term in top_methods:
            s = stats['Methodology'][term]
            s['count'] += 1
            if year:
                s['year_trend'][year] += 1
                if year < s['first_year']:
                    s['first_year'] = year
                    s['first_title'] = title
                    s['first_link'] = link
            for a in authors:
                s['authors'][a] += 1
        for term in top_techs:
            s = stats['Technique'][term]
            s['count'] += 1
            if year:
                s['year_trend'][year] += 1
                if year < s['first_year']:
                    s['first_year'] = year
                    s['first_title'] = title
                    s['first_link'] = link
            for a in authors:
                s['authors'][a] += 1
        print(f"Processing paper {idx+1}/{total}: {title}")
    return paper_results, stats

def write_paper_csv(paper_results, filepath):
    pd.DataFrame(paper_results).to_csv(filepath, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
    print(f"Wrote paper analysis to {filepath}")

def write_insights_csv(stats, category, filepath):
    rows = []
    for term, data in stats[category].items():
        if data['count'] == 0:
            continue
        first_year = data['first_year'] if data['first_year'] != math.inf else ''
        trend = "; ".join(f"{yr}:{cnt}" for yr, cnt in sorted(data['year_trend'].items()))
        top_auths = "; ".join(f"{a}({c})" for a, c in data['authors'].most_common(3))
        rows.append([term, first_year, data['first_title'], data['first_link'], data['count'],
                     trend, top_auths, data['top_paper']])
    out_df = pd.DataFrame(rows, columns=[category, "FirstYear", "FirstPaperTitle", "FirstPaperLink",
                                         "TotalUsage", "YearTrend", "TopAuthors", "MostInfluentialPaper"])
    out_df.to_csv(filepath, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
    print(f"Wrote {category.lower()} insights to {filepath}")

def main():
    papers_df = load_papers("papers.csv")
    col_map = map_columns(papers_df)
    ontology = load_ontology("onto.csv")
    paper_results, stats = analyze_papers(papers_df, col_map, ontology)
    write_paper_csv(paper_results, os.path.join(output_dir, "paper_analysis.csv"))
    write_insights_csv(stats, 'Domain', os.path.join(output_dir, "domain_insights.csv"))
    write_insights_csv(stats, 'Methodology', os.path.join(output_dir, "methodology_insights.csv"))
    write_insights_csv(stats, 'Technique', os.path.join(output_dir, "technique_insights.csv"))
    print("All outputs saved in 'output/'.")

if __name__ == "__main__":
    main()

