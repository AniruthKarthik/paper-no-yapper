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
    """Load papers.csv with the new Scopus headers."""
    print("Loading papers from:", filepath)
    has_header = detect_header(filepath)
    print(f" - Header detected: {has_header}")

    # Read CSV with pandas for better handling
    try:
        df = pd.read_csv(filepath, encoding='utf-8', on_bad_lines='skip', quotechar='"')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='latin1', on_bad_lines='skip', quotechar='"')
    
    # Clean column names
    df.columns = [str(c).strip() for c in df.columns]
    print(f"Loaded {len(df)} papers with columns: {list(df.columns)}")
    return df

def map_columns(df):
    """Map the new Scopus columns to our expected fields."""
    print("Mapping columns...")
    col_map = {}
    
    # Updated mapping for Scopus export format
    column_mapping = {
        'title': ['Title'],
        'abstract': ['Abstract'],
        'year': ['Year'],
        'authors': ['Authors', 'Author full names'],
        'keywords': ['Author Keywords', 'Index Keywords'],
        'link': ['Link', 'DOI'],
        'cited_by': ['Cited by'],
        'source': ['Source title'],
        'document_type': ['Document Type'],
        'conference_name': ['Conference name'],
        'conference_date': ['Conference date'],
        'conference_location': ['Conference location']
    }
    
    columns = list(df.columns)
    for field, possible_cols in column_mapping.items():
        for col in columns:
            if col in possible_cols:
                col_map[field] = col
                print(f"Mapped {field} -> {col}")
                break
        if field not in col_map:
            col_map[field] = None
            if field in ['title', 'abstract', 'year']:  # Critical fields
                print(f"Warning: No match for critical field {field}")
    
    return col_map

def load_ontology(filepath):
    """Load ontology from CSV file."""
    print("Loading ontology from:", filepath)
    try:
        onto_df = pd.read_csv(filepath, quotechar='"', on_bad_lines='skip', engine="python")
    except Exception as e:
        print(f"Error loading ontology: {e}")
        return defaultdict(list)
    
    onto_df.columns = [c.strip().lower() for c in onto_df.columns]
    
    # Find term and category columns
    term_col = None
    cat_col = None
    
    for col in onto_df.columns:
        if 'term' in col or 'name' in col:
            term_col = col
            break
    
    for col in onto_df.columns:
        if 'category' in col or 'type' in col or 'class' in col:
            cat_col = col
            break
    
    if not term_col or not cat_col:
        print("Error: Could not find term and category columns in ontology")
        return defaultdict(list)
    
    ontology = defaultdict(list)
    for _, row in onto_df.iterrows():
        term = str(row[term_col]).strip()
        cat = str(row[cat_col]).strip().capitalize()
        if pd.isna(term) or pd.isna(cat) or term == 'nan' or cat == 'nan':
            continue
        ontology[cat].append(term)
    
    # Remove duplicates
    for cat in ontology:
        ontology[cat] = list(set(ontology[cat]))
    
    print(f"Loaded ontology with categories: {list(ontology.keys())}")
    return ontology

def get_synonyms(term):
    """Get synonyms for a term using WordNet."""
    syns = set()
    if wn:
        for syn in wn.synsets(term):
            for lemma in syn.lemmas():
                syn_term = lemma.name().replace('_', ' ')
                syns.add(syn_term.lower())
    return syns

def match_terms(text, terms):
    """Match terms in text using exact match, fuzzy matching, and synonyms."""
    matches = Counter()
    if not text or pd.isna(text):
        return matches
    
    lower_text = str(text).lower()
    
    for term in terms:
        if pd.isna(term):
            continue
        term_lower = str(term).lower().strip()
        if not term_lower:
            continue
            
        # Exact match
        if re.search(r'\b' + re.escape(term_lower) + r'\b', lower_text):
            matches[term] += 1
            continue
        
        # Fuzzy match
        if fuzz and fuzz.partial_ratio(term_lower, lower_text) >= 85:
            matches[term] += 1
            continue
        elif not fuzz:
            if SequenceMatcher(None, term_lower, lower_text).ratio() > 0.85:
                matches[term] += 1
                continue
        
        # Synonym match
        if wn:
            for syn in get_synonyms(term_lower):
                if re.search(r'\b' + re.escape(syn) + r'\b', lower_text):
                    matches[term] += 1
                    break
    
    return matches

def safe_int(value, default=0):
    """Safely convert value to int."""
    try:
        if pd.isna(value):
            return default
        return int(float(str(value)))
    except (ValueError, TypeError):
        return default

def analyze_papers(df, col_map, ontology):
    """Analyze papers and extract domain, methodology, and technique information."""
    # Initialize statistics
    stats = {}
    for category in ['Domain', 'Methodology', 'Technique']:
        stats[category] = defaultdict(lambda: {
            'count': 0,
            'first_year': math.inf,
            'first_paper': {},
            'most_cited_paper': {},
            'max_citations': 0,
            'papers': []
        })
    
    paper_results = []
    total = len(df)
    
    print(f"Analyzing {total} papers...")
    
    for idx, row in df.iterrows():
        try:
            # Extract paper information
            title = str(row[col_map['title']]) if col_map['title'] and not pd.isna(row[col_map['title']]) else ''
            abstract = str(row[col_map['abstract']]) if col_map['abstract'] and not pd.isna(row[col_map['abstract']]) else ''
            
            # Handle keywords - might be in multiple columns
            keywords = ''
            if col_map['keywords']:
                if isinstance(col_map['keywords'], list):
                    kw_parts = []
                    for kw_col in col_map['keywords']:
                        if kw_col in df.columns and not pd.isna(row[kw_col]):
                            kw_parts.append(str(row[kw_col]))
                    keywords = '; '.join(kw_parts)
                else:
                    keywords = str(row[col_map['keywords']]) if not pd.isna(row[col_map['keywords']]) else ''
            
            year = safe_int(row[col_map['year']]) if col_map['year'] else None
            cited_by = safe_int(row[col_map['cited_by']]) if col_map['cited_by'] else 0
            
            # Handle authors
            authors = []
            if col_map['authors'] and not pd.isna(row[col_map['authors']]):
                author_str = str(row[col_map['authors']])
                authors = [a.strip() for a in author_str.replace(";", ",").split(",") if a.strip()]
            authors_str = "; ".join(authors)
            
            link = str(row[col_map['link']]) if col_map['link'] and not pd.isna(row[col_map['link']]) else ''
            
            # Create paper info dictionary
            paper_info = {
                'title': title,
                'authors': authors_str,
                'year': year,
                'abstract': abstract,
                'link': link,
                'cited_by': cited_by
            }
            
        except Exception as e:
            skipped_log.write(f"Row {idx} error extracting basic info: {e}\n")
            continue
        
        # Skip if no title
        if not title.strip():
            continue
        
        # Combine text for analysis
        full_text = " ".join([title, abstract, keywords])
        
        # Match terms for each category
        domain_matches = match_terms(full_text, ontology.get('Domain', []))
        methodology_matches = match_terms(full_text, ontology.get('Methodology', []))
        technique_matches = match_terms(full_text, ontology.get('Technique', []))
        
        # Get top matches
        top_domains = [term for term, _ in domain_matches.most_common(5)]
        top_methodologies = [term for term, _ in methodology_matches.most_common(5)]
        top_techniques = [term for term, _ in technique_matches.most_common(5)]
        
        # Store paper results
        paper_results.append({
            'Title': title,
            'Authors': authors_str,
            'Year': year,
            'Abstract': abstract[:500] + "..." if len(abstract) > 500 else abstract,
            'Link': link,
            'Cited_By': cited_by,
            'Top_Domains': "; ".join(top_domains),
            'Top_Methodologies': "; ".join(top_methodologies),
            'Top_Techniques': "; ".join(top_techniques)
        })
        
        # Update statistics for each category
        categories = {
            'Domain': top_domains,
            'Methodology': top_methodologies,
            'Technique': top_techniques
        }
        
        for category, terms in categories.items():
            for term in terms:
                s = stats[category][term]
                s['count'] += 1
                s['papers'].append(paper_info)
                
                # Update first paper
                if year and year < s['first_year']:
                    s['first_year'] = year
                    s['first_paper'] = paper_info.copy()
                
                # Update most cited paper
                if cited_by > s['max_citations']:
                    s['max_citations'] = cited_by
                    s['most_cited_paper'] = paper_info.copy()
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{total} papers...")
    
    print(f"Analysis complete. Processed {len(paper_results)} papers.")
    return paper_results, stats

def write_paper_csv(paper_results, filepath):
    """Write paper analysis results to CSV."""
    df = pd.DataFrame(paper_results)
    df.to_csv(filepath, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
    print(f"Wrote paper analysis to {filepath}")

def write_first_papers_csv(stats, filepath):
    """Write first papers for each domain, methodology, and technique."""
    rows = []
    
    for category in ['Domain', 'Methodology', 'Technique']:
        for term, data in stats[category].items():
            if data['count'] == 0 or not data['first_paper']:
                continue
            
            fp = data['first_paper']
            rows.append([
                category,
                term,
                fp.get('title', ''),
                fp.get('link', ''),
                fp.get('year', ''),
                fp.get('abstract', ''),
                fp.get('authors', ''),
                fp.get('cited_by', 0)
            ])
    
    df = pd.DataFrame(rows, columns=[
        'Category', 'Term', 'First_Paper_Title', 'First_Paper_Link', 
        'First_Paper_Year', 'First_Paper_Abstract', 'First_Paper_Authors', 'First_Paper_Citations'
    ])
    df = df.sort_values(['Category', 'First_Paper_Year'])
    df.to_csv(filepath, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
    print(f"Wrote first papers analysis to {filepath}")

def write_most_cited_csv(stats, filepath):
    """Write most cited papers for each domain, methodology, and technique."""
    rows = []
    
    for category in ['Domain', 'Methodology', 'Technique']:
        for term, data in stats[category].items():
            if data['count'] == 0 or not data['most_cited_paper']:
                continue
            
            mcp = data['most_cited_paper']
            rows.append([
                category,
                term,
                mcp.get('title', ''),
                mcp.get('link', ''),
                mcp.get('year', ''),
                mcp.get('abstract', ''),
                mcp.get('authors', ''),
                mcp.get('cited_by', 0)
            ])
    
    df = pd.DataFrame(rows, columns=[
        'Category', 'Term', 'Most_Cited_Paper_Title', 'Most_Cited_Paper_Link',
        'Most_Cited_Paper_Year', 'Most_Cited_Paper_Abstract', 'Most_Cited_Paper_Authors', 'Citations'
    ])
    df = df.sort_values(['Category', 'Citations'], ascending=[True, False])
    df.to_csv(filepath, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
    print(f"Wrote most cited papers analysis to {filepath}")

def print_results(stats):
    """Print detailed results to console."""
    print("\n" + "="*80)
    print("RESEARCH PAPER ANALYSIS RESULTS")
    print("="*80)
    
    for category in ['Domain', 'Methodology', 'Technique']:
        print(f"\n{category.upper()}S:")
        print("-" * 50)
        
        # Sort by count (popularity)
        sorted_terms = sorted(stats[category].items(), key=lambda x: x[1]['count'], reverse=True)
        
        for term, data in sorted_terms[:10]:  # Show top 10
            if data['count'] == 0:
                continue
                
            print(f"\n{term} (Found in {data['count']} papers):")
            
            # First paper
            if data['first_paper']:
                fp = data['first_paper']
                print(f"  First Paper ({fp.get('year', 'Unknown')}):")
                print(f"    Title: {fp.get('title', '')[:100]}...")
                print(f"    Authors: {fp.get('authors', '')[:80]}...")
                if fp.get('link'):
                    print(f"    Link: {fp.get('link')}")
            
            # Most cited paper
            if data['most_cited_paper'] and data['max_citations'] > 0:
                mcp = data['most_cited_paper']
                print(f"  Most Cited Paper ({mcp.get('cited_by', 0)} citations):")
                print(f"    Title: {mcp.get('title', '')[:100]}...")
                print(f"    Year: {mcp.get('year', 'Unknown')}")
                print(f"    Authors: {mcp.get('authors', '')[:80]}...")
        
        if len(sorted_terms) > 10:
            print(f"\n... and {len(sorted_terms) - 10} more {category.lower()}s")

def main():
    """Main function to run the analysis."""
    try:
        # Load data
        papers_df = load_papers("papers.csv")
        col_map = map_columns(papers_df)
        ontology = load_ontology("onto.csv")
        
        if not ontology:
            print("Error: Could not load ontology. Please check onto.csv file.")
            return
        
        # Analyze papers
        paper_results, stats = analyze_papers(papers_df, col_map, ontology)
        
        # Write outputs
        write_paper_csv(paper_results, os.path.join(output_dir, "paper_analysis.csv"))
        write_first_papers_csv(stats, os.path.join(output_dir, "first_papers_analysis.csv"))
        write_most_cited_csv(stats, os.path.join(output_dir, "most_cited_papers_analysis.csv"))
        
        # Print results to console
        print_results(stats)
        
        print(f"\nAll outputs saved in '{output_dir}/' directory:")
        print("- paper_analysis.csv: Complete paper analysis with classifications")
        print("- first_papers_analysis.csv: First papers for each domain/methodology/technique")
        print("- most_cited_papers_analysis.csv: Most cited papers for each category")
        print("- skipped_rows.log: Log of any processing issues")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find required file - {e}")
        print("Please ensure both 'papers.csv' and 'onto.csv' files are in the current directory.")
    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        import traceback
        traceback.print_exc()
    finally:
        skipped_log.close()

if __name__ == "__main__":
    main()