#  Paper No Yapper
### ( Work in Progress for the Unsupervised version in Master branch )

**Version:** v1.0.0

Paper No Yapper is a comprehensive **research paper analysis system**.
It automatically extracts **Domains, Methodologies, and Techniques** from research papers, using a combination of:

*  **Ontology-based term matching** (`onto.csv`)
*  **Exact, fuzzy, and synonym detection** (WordNet + FuzzyWuzzy)
*  **Automated statistics & visualizations**

Outputs include: first papers, most cited papers, research impact analysis, citation timelines, and co-occurrence networks.

---

## ✨ Features (v1.0.0)

*  Reads **Scopus CSV exports** (`scopus.csv`).
*  Maps Scopus columns automatically (Title, Abstract, Authors, Keywords, etc.).
*  Matches text against ontology terms (Domains / Methodologies / Techniques).
*  Generates multiple CSV reports:

  * **paper\_analysis.csv** → Paper-level classification.
  * **first\_papers\_analysis.csv** → Earliest paper per category.
  * **most\_cited\_papers\_analysis.csv** → Highest impact papers.
*  Creates **visualizations**:

  * Term distributions
  * Citation impact analysis
  * Temporal evolution & trends
  * Co-occurrence networks
  * Heatmaps & maturity index
  * Category comparisons
*  Logs skipped/problematic rows into `skipped_rows.log`.

---

##  Input Files

1. **scopus.csv** → Exported from Scopus (with abstracts, titles, authors, year, citations, etc.).
2. **onto.csv** → Ontology file with categories (Domain / Methodology / Technique).

Format example:

```csv
term,category
Deep Learning,Methodology
Static Analysis,Technique
Cybersecurity,Domain
```

---

##  Requirements

Install dependencies:

```bash
pip install pandas matplotlib seaborn networkx fuzzywuzzy nltk numpy
```

Optional: for WordNet synonyms (first run will download corpus):

```python
import nltk
nltk.download('wordnet')
```

---

##  Usage

```bash
python script.py
```

This will:

* Load `scopus.csv` + `onto.csv`
* Run analysis and classifications
* Save results to the `output/` folder:

```
output/
 ├─ paper_analysis.csv
 ├─ first_papers_analysis.csv
 ├─ most_cited_papers_analysis.csv
 ├─ skipped_rows.log
 └─ plots/
      ├─ domain_distribution.png
      ├─ methodology_distribution.png
      ├─ technique_distribution.png
      ├─ citation_impact_analysis.png
      ├─ temporal_evolution.png
      ├─ cooccurrence_networks.png
      ├─ impact_heatmap.png
      ├─ temporal_trends.png
      └─ category_comparisons.png
```

---

##  Version History

* [v0.0.0](https://github.com/AniruthKarthik/paper-no-yapper/releases/tag/v0.0.0) — Released before v1.0.0
* **\[v1.0.0]** – Major release with Scopus CSV support, ontology-driven classification, fuzzy matching, synonym expansion, citation analysis, and visualization suite.
* **\[v0.0.0]** – Initial config with KB matching & unsupervised extraction.

---

##  Related Repos

* **[Paper Grabber](https://github.com/AniruthKarthik/project-sandbox/tree/master/paper-grabber)** – Bookmarklet to save your favourite papers into an excel sheet.

---

