#!/usr/bin/env python3
import os
import re
import csv
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from collections import defaultdict, Counter
from difflib import SequenceMatcher
import numpy as np

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

def create_visualizations(paper_results, stats):
    """Create comprehensive visualizations based on identified domains, methodologies, and techniques."""
    print("Creating visualizations...")
    
    # Set style for better-looking plots
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Domain Distribution Bar Chart
    create_category_distribution_plot(stats, 'Domain', plots_dir)
    create_category_distribution_plot(stats, 'Methodology', plots_dir)
    create_category_distribution_plot(stats, 'Technique', plots_dir)
    
    # 2. Timeline Evolution Plots
    create_timeline_evolution_plot(paper_results, stats, plots_dir)
    
    # 3. Citation Analysis Plots
    create_citation_analysis_plot(stats, plots_dir)
    
    # 4. Co-occurrence Network Analysis
    create_cooccurrence_network(paper_results, plots_dir)
    
    # 5. Research Impact Heatmap
    create_impact_heatmap(stats, plots_dir)
    
    # 6. Temporal Trend Analysis
    create_temporal_trends(paper_results, plots_dir)
    
    # 7. Category Comparison Plots
    create_category_comparison_plots(stats, plots_dir)
    
    print(f"All visualizations saved in '{plots_dir}/' directory")

def create_category_distribution_plot(stats, category, plots_dir):
    """Create distribution plots for each category."""
    category_data = stats[category]
    if not category_data:
        return
    
    # Get top 15 terms by count
    sorted_terms = sorted(category_data.items(), key=lambda x: x[1]['count'], reverse=True)[:15]
    
    if not sorted_terms:
        return
    
    terms = [item[0] for item in sorted_terms]
    counts = [item[1]['count'] for item in sorted_terms]
    citations = [item[1]['max_citations'] for item in sorted_terms]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Plot 1: Paper count distribution
    bars1 = ax1.barh(terms, counts, color=sns.color_palette("viridis", len(terms)))
    ax1.set_xlabel('Number of Papers', fontsize=12)
    ax1.set_ylabel(f'{category} Terms', fontsize=12)
    ax1.set_title(f'Top {category}s by Paper Count', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars1, counts)):
        ax1.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height()/2, 
                str(count), ha='left', va='center', fontweight='bold')
    
    # Plot 2: Maximum citations per term
    bars2 = ax2.barh(terms, citations, color=sns.color_palette("plasma", len(terms)))
    ax2.set_xlabel('Maximum Citations', fontsize=12)
    ax2.set_ylabel(f'{category} Terms', fontsize=12)
    ax2.set_title(f'Top {category}s by Maximum Citations', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, cit) in enumerate(zip(bars2, citations)):
        if cit > 0:
            ax2.text(bar.get_width() + max(citations) * 0.01, bar.get_y() + bar.get_height()/2, 
                    str(cit), ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{category.lower()}_distribution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_timeline_evolution_plot(paper_results, stats, plots_dir):
    """Create timeline showing evolution of domains, methodologies, and techniques."""
    # Convert to DataFrame for easier processing
    df = pd.DataFrame(paper_results)
    df = df[df['Year'].notna() & (df['Year'] > 1990)]  # Filter valid years
    
    if df.empty:
        return
    
    # Create evolution timeline
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    
    categories = [
        ('Top_Domains', 'Domain', axes[0]),
        ('Top_Methodologies', 'Methodology', axes[1]), 
        ('Top_Techniques', 'Technique', axes[2])
    ]
    
    for col_name, category, ax in categories:
        # Count papers per year for top terms
        year_term_counts = defaultdict(lambda: defaultdict(int))
        
        for _, row in df.iterrows():
            year = row['Year']
            terms = str(row[col_name]).split(';')
            for term in terms:
                term = term.strip()
                if term and term != 'nan':
                    year_term_counts[year][term] += 1
        
        # Get top 8 terms overall
        term_totals = defaultdict(int)
        for year_data in year_term_counts.values():
            for term, count in year_data.items():
                term_totals[term] += count
        
        top_terms = [term for term, _ in sorted(term_totals.items(), key=lambda x: x[1], reverse=True)[:8]]
        
        # Plot evolution for top terms
        years = sorted(year_term_counts.keys())
        for i, term in enumerate(top_terms):
            term_counts_by_year = [year_term_counts[year][term] for year in years]
            ax.plot(years, term_counts_by_year, marker='o', linewidth=2, 
                   label=term[:30] + '...' if len(term) > 30 else term)
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Number of Papers', fontsize=12)
        ax.set_title(f'{category} Evolution Over Time', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'temporal_evolution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_citation_analysis_plot(stats, plots_dir):
    """Create citation impact analysis plots."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    categories = ['Domain', 'Methodology', 'Technique']
    
    for i, category in enumerate(categories):
        category_data = stats[category]
        if not category_data:
            continue
        
        # Get data for scatter plot
        terms = []
        paper_counts = []
        max_citations = []
        avg_citations = []
        
        for term, data in category_data.items():
            if data['count'] > 0:
                terms.append(term[:20] + '...' if len(term) > 20 else term)
                paper_counts.append(data['count'])
                max_citations.append(data['max_citations'])
                
                # Calculate average citations for papers containing this term
                total_cites = sum(paper['cited_by'] for paper in data['papers'])
                avg_cites = total_cites / len(data['papers']) if data['papers'] else 0
                avg_citations.append(avg_cites)
        
        # Create scatter plot: Paper count vs Average citations
        scatter = axes[i].scatter(paper_counts, avg_citations, 
                                 s=[c/5 + 50 for c in max_citations],  # Size by max citations
                                 alpha=0.6, c=range(len(terms)), cmap='viridis')
        
        # Add labels for top points
        for j, (x, y, term) in enumerate(zip(paper_counts, avg_citations, terms)):
            if y > np.percentile(avg_citations, 75) or x > np.percentile(paper_counts, 75):
                axes[i].annotate(term, (x, y), xytext=(5, 5), textcoords='offset points',
                               fontsize=8, ha='left')
        
        axes[i].set_xlabel('Number of Papers', fontsize=12)
        axes[i].set_ylabel('Average Citations per Paper', fontsize=12)
        axes[i].set_title(f'{category} Impact Analysis', fontsize=14, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'citation_impact_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_cooccurrence_network(paper_results, plots_dir):
    """Create co-occurrence network of domains, methodologies, and techniques."""
    df = pd.DataFrame(paper_results)
    
    # Build co-occurrence matrices
    categories = {
        'domains': 'Top_Domains',
        'methodologies': 'Top_Methodologies', 
        'techniques': 'Top_Techniques'
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    for idx, (cat_name, col_name) in enumerate(categories.items()):
        # Build co-occurrence matrix
        cooccurrence = defaultdict(lambda: defaultdict(int))
        term_counts = defaultdict(int)
        
        for _, row in df.iterrows():
            terms = [t.strip() for t in str(row[col_name]).split(';') if t.strip() and t.strip() != 'nan']
            
            # Count individual terms
            for term in terms:
                term_counts[term] += 1
            
            # Count co-occurrences
            for i, term1 in enumerate(terms):
                for term2 in terms[i+1:]:
                    cooccurrence[term1][term2] += 1
                    cooccurrence[term2][term1] += 1
        
        # Get top terms
        top_terms = [term for term, count in sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[:15]]
        
        if len(top_terms) < 2:
            continue
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes
        for term in top_terms:
            G.add_node(term, size=term_counts[term])
        
        # Add edges (co-occurrences)
        for term1 in top_terms:
            for term2 in top_terms:
                if term1 != term2 and cooccurrence[term1][term2] > 0:
                    G.add_edge(term1, term2, weight=cooccurrence[term1][term2])
        
        # Draw network
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Node sizes proportional to frequency
        node_sizes = [G.nodes[node]['size'] * 50 for node in G.nodes()]
        
        # Edge widths proportional to co-occurrence
        edge_widths = [G[u][v]['weight'] * 0.5 for u, v in G.edges()]
        
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.7, 
                             node_color=range(len(G.nodes())), cmap='Set3', ax=axes[idx])
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, ax=axes[idx])
        
        # Add labels
        labels = {node: node[:15] + '...' if len(node) > 15 else node for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=axes[idx])
        
        axes[idx].set_title(f'{cat_name.title()} Co-occurrence Network', 
                           fontsize=14, fontweight='bold')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'cooccurrence_networks.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_impact_heatmap(stats, plots_dir):
    """Create heatmap showing impact metrics across categories."""
    categories = ['Domain', 'Methodology', 'Technique']
    
    # Collect data for heatmap
    heatmap_data = []
    terms_list = []
    
    for category in categories:
        category_data = stats[category]
        sorted_terms = sorted(category_data.items(), key=lambda x: x[1]['count'], reverse=True)[:10]
        
        for term, data in sorted_terms:
            if data['count'] > 0:
                avg_citations = sum(p['cited_by'] for p in data['papers']) / len(data['papers']) if data['papers'] else 0
                first_year = data['first_year'] if data['first_year'] != math.inf else 2024
                maturity = 2024 - first_year  # Years since first appearance
                
                heatmap_data.append([
                    data['count'],  # Paper count
                    data['max_citations'],  # Max citations
                    avg_citations,  # Average citations
                    maturity  # Maturity (years)
                ])
                terms_list.append(f"{category[:4]}: {term[:25]}")
    
    if not heatmap_data:
        return
    
    # Create heatmap
    heatmap_df = pd.DataFrame(heatmap_data, 
                             columns=['Paper Count', 'Max Citations', 'Avg Citations', 'Maturity (Years)'],
                             index=terms_list)
    
    # Normalize data for better visualization
    heatmap_normalized = heatmap_df.div(heatmap_df.max())
    
    plt.figure(figsize=(12, max(8, len(terms_list) * 0.4)))
    sns.heatmap(heatmap_normalized, annot=True, cmap='YlOrRd', 
                cbar_kws={'label': 'Normalized Score'}, fmt='.2f')
    plt.title('Research Impact Heatmap\n(Normalized Metrics)', fontsize=16, fontweight='bold')
    plt.xlabel('Impact Metrics', fontsize=12)
    plt.ylabel('Terms by Category', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'impact_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_temporal_trends(paper_results, plots_dir):
    """Create temporal trend analysis."""
    df = pd.DataFrame(paper_results)
    df = df[df['Year'].notna() & (df['Year'] > 1990)]
    
    if df.empty:
        return
    
    # Annual publication trends
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Overall publication trend
    yearly_counts = df['Year'].value_counts().sort_index()
    ax1.plot(yearly_counts.index, yearly_counts.values, marker='o', linewidth=2)
    ax1.set_title('Annual Publication Trends', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Number of Papers')
    ax1.grid(True, alpha=0.3)
    
    # 2. Citation trends over time
    citation_by_year = df.groupby('Year')['Cited_By'].agg(['mean', 'max']).fillna(0)
    ax2.plot(citation_by_year.index, citation_by_year['mean'], 
             label='Average Citations', marker='o', linewidth=2)
    ax2.plot(citation_by_year.index, citation_by_year['max'], 
             label='Maximum Citations', marker='s', linewidth=2)
    ax2.set_title('Citation Trends Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Citations')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Research diversity over time (unique terms per year)
    diversity_data = []
    for year in sorted(df['Year'].unique()):
        year_papers = df[df['Year'] == year]
        unique_domains = set()
        unique_methods = set()
        unique_techniques = set()
        
        for _, row in year_papers.iterrows():
            domains = [t.strip() for t in str(row['Top_Domains']).split(';') if t.strip() != 'nan']
            methods = [t.strip() for t in str(row['Top_Methodologies']).split(';') if t.strip() != 'nan']
            techniques = [t.strip() for t in str(row['Top_Techniques']).split(';') if t.strip() != 'nan']
            
            unique_domains.update(domains)
            unique_methods.update(methods)
            unique_techniques.update(techniques)
        
        diversity_data.append({
            'year': year,
            'domains': len(unique_domains),
            'methodologies': len(unique_methods),
            'techniques': len(unique_techniques)
        })
    
    diversity_df = pd.DataFrame(diversity_data)
    ax3.plot(diversity_df['year'], diversity_df['domains'], 
             label='Unique Domains', marker='o', linewidth=2)
    ax3.plot(diversity_df['year'], diversity_df['methodologies'], 
             label='Unique Methodologies', marker='s', linewidth=2)
    ax3.plot(diversity_df['year'], diversity_df['techniques'], 
             label='Unique Techniques', marker='^', linewidth=2)
    ax3.set_title('Research Diversity Over Time', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Number of Unique Terms')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Research maturity index (cumulative unique terms)
    cumulative_terms = []
    all_domains = set()
    all_methods = set()
    all_techniques = set()
    
    for year in sorted(df['Year'].unique()):
        year_papers = df[df['Year'] <= year]  # Cumulative
        
        for _, row in year_papers.iterrows():
            domains = [t.strip() for t in str(row['Top_Domains']).split(';') if t.strip() != 'nan']
            methods = [t.strip() for t in str(row['Top_Methodologies']).split(';') if t.strip() != 'nan']
            techniques = [t.strip() for t in str(row['Top_Techniques']).split(';') if t.strip() != 'nan']
            
            all_domains.update(domains)
            all_methods.update(methods)
            all_techniques.update(techniques)
        
        total_terms = len(all_domains) + len(all_methods) + len(all_techniques)
        cumulative_terms.append({'year': year, 'total_terms': total_terms})
    
    cum_df = pd.DataFrame(cumulative_terms)
    ax4.plot(cum_df['year'], cum_df['total_terms'], 
             marker='o', linewidth=3, color='purple')
    ax4.set_title('Research Field Maturity Index', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Cumulative Unique Terms')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'temporal_trends.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_category_comparison_plots(stats, plots_dir):
    """Create comparison plots between categories."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    categories = ['Domain', 'Methodology', 'Technique']
    
    # 1. Category sizes comparison
    category_sizes = []
    category_names = []
    for cat in categories:
        active_terms = sum(1 for data in stats[cat].values() if data['count'] > 0)
        category_sizes.append(active_terms)
        category_names.append(cat)
    
    axes[0, 0].bar(category_names, category_sizes, color=['skyblue', 'lightgreen', 'salmon'])
    axes[0, 0].set_title('Active Terms per Category', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Number of Active Terms')
    for i, v in enumerate(category_sizes):
        axes[0, 0].text(i, v + max(category_sizes) * 0.01, str(v), 
                        ha='center', va='bottom', fontweight='bold')
    
    # 2. Total papers per category
    total_papers = []
    for cat in categories:
        total = sum(data['count'] for data in stats[cat].values())
        total_papers.append(total)
    
    axes[0, 1].bar(category_names, total_papers, color=['lightcoral', 'gold', 'lightsteelblue'])
    axes[0, 1].set_title('Total Paper Classifications per Category', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Total Classifications')
    for i, v in enumerate(total_papers):
        axes[0, 1].text(i, v + max(total_papers) * 0.01, str(v), 
                        ha='center', va='bottom', fontweight='bold')
    
    # 3. Average citations per category
    avg_citations = []
    for cat in categories:
        all_citations = []
        for data in stats[cat].values():
            for paper in data['papers']:
                all_citations.append(paper['cited_by'])
        avg_cit = np.mean(all_citations) if all_citations else 0
        avg_citations.append(avg_cit)
    
    axes[1, 0].bar(category_names, avg_citations, color=['mediumseagreen', 'orange', 'mediumpurple'])
    axes[1, 0].set_title('Average Citations per Category', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Average Citations')
    for i, v in enumerate(avg_citations):
        axes[1, 0].text(i, v + max(avg_citations) * 0.01, f'{v:.1f}', 
                        ha='center', va='bottom', fontweight='bold')
    
    # 4. Research maturity comparison (average years since first paper)
    avg_maturity = []
    current_year = 2024
    for cat in categories:
        maturities = []
        for data in stats[cat].values():
            if data['first_year'] != math.inf:
                maturity = current_year - data['first_year']
                maturities.append(maturity)
        avg_mat = np.mean(maturities) if maturities else 0
        avg_maturity.append(avg_mat)
    
    axes[1, 1].bar(category_names, avg_maturity, color=['teal', 'crimson', 'darkgoldenrod'])
    axes[1, 1].set_title('Average Research Maturity per Category', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Average Years Since First Paper')
    for i, v in enumerate(avg_maturity):
        axes[1, 1].text(i, v + max(avg_maturity) * 0.01, f'{v:.1f}', 
                        ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'category_comparisons.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to run the analysis."""
    try:
        # Load data
        papers_df = load_papers("scopus.csv")
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
        
        # Create visualizations
        create_visualizations(paper_results, stats)
        
        # Print results to console
        print_results(stats)
        
        print(f"\nAll outputs saved in '{output_dir}/' directory:")
        print("- paper_analysis.csv: Complete paper analysis with classifications")
        print("- first_papers_analysis.csv: First papers for each domain/methodology/technique")
        print("- most_cited_papers_analysis.csv: Most cited papers for each category")
        print("- plots/: Directory containing all visualization graphs")
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
