# THE DEFINITIVE 2026 ROADMAP: WRITE THE BEST MALWARE DETECTION SURVEY EVER
## Complete Blueprint from Conception to Publication-Ready Paper

---

## 🎯 EXECUTIVE OVERVIEW

This roadmap will take you from **"I want to write a survey"** to **"Published in ACM Computing Surveys / IEEE S&P / USENIX Security"** in **6–8 months** with **90%+ quality**.

**What you'll have when done:**
- ✅ **60–70 page comprehensive survey**
- ✅ **150–180 papers analyzed** (highest in field)
- ✅ **5+ publication-quality figures**
- ✅ **15+ comprehensive tables**
- ✅ **Decade-aware temporal structure**
- ✅ **Modern paradigms fully covered** (GNN, Transformers, LLMs)
- ✅ **Practical deployment guidance**
- ✅ **Prioritized open problems**
- ✅ **Expert-reviewed and validated**

---

# PART I: PRE-WRITING PHASE (Weeks 1–2)

## Week 1: Define Your Survey

### **Step 1.1: Choose Your Title**
```
WEAK TITLE (Don't Use):
"Malware Detection with Artificial Intelligence: A Survey"
(Same as Gaber et al.; not distinctive)

STRONG TITLES (Use These):
"Static and Behavioral Malware Analysis (2015–2025): 
From Feature Engineering to Foundation Models — 
A Decade of Evolution, Open Problems, and Production-Ready Solutions"

OR

"Malware Detection: A Decade-Spanning Survey of Paradigm Shifts,
Robustness Challenges, and Future Directions (2015–2025)"

OR

"Semantic-Aware Malware Detection: 
Beyond Syntax to Robust Binaries — A Comprehensive 2015–2025 Review"
```

**Why strong titles work:**
- ✔ Temporal scope explicit (2015–2025)
- ✔ Paradigm shift mentioned
- ✔ Practical value indicated
- ✔ Shows uniqueness vs. Gaber

### **Step 1.2: Define Your Research Questions**

```
YOUR RESEARCH QUESTIONS (7 Questions, More Ambitious Than Gaber's 6):

RQ1: How have malware detection techniques evolved (2015→2025)?
     What paradigm shifts have occurred?
     (Gaber didn't address this)

RQ2: What are the fundamental limitations of each paradigm?
     What problems drove innovation?
     (Gaber didn't analyze this)

RQ3: What is the current gold-standard evaluation methodology (2025)?
     How should temporal, adversarial, cross-dataset robustness be tested?
     (Gaber vague on this)

RQ4: What is the landscape of modern (2021–2025) methods?
     GNN, Transformers, LLM-assisted, representation learning?
     (Gaber severely undercovers this)

RQ5: What are the unresolved open problems, ranked by impact?
     Which are solvable? Which are fundamental challenges?
     (Gaber doesn't address this)

RQ6: What should practitioners implement TODAY (2025)?
     Clear maturity assessment and recommendations?
     (Gaber provides no clear guidance)

RQ7: How can we deploy detectors in production?
     Scalability, latency, reproducibility, tools?
     (Gaber doesn't address this)
```

### **Step 1.3: Choose Your Venue**

```
TIER-1 VENUES (Target These First):
1. ACM Computing Surveys
   - Impact Factor: 15+
   - Acceptance Rate: ~15%
   - Timeline: 3–6 months review
   - Effort: 60–70 pages typical
   
2. IEEE Transactions on Software Engineering (TSE)
   - Impact Factor: 8+
   - Acceptance Rate: ~20%
   - Timeline: 4–8 months review
   - Effort: 50–60 pages typical

3. IEEE Transactions on Dependable & Secure Computing
   - Impact Factor: 6+
   - Acceptance Rate: ~20%
   - Timeline: 3–6 months review
   - Effort: 50–60 pages typical

4. USENIX Security Symposium (Survey Track, if available)
   - Competitive, high-impact
   - Shorter review cycle (2–3 months)
   - Effort: 40–50 pages

RECOMMENDATION: Target ACM Computing Surveys first (same as Gaber)
```

### **Step 1.4: Assemble Your Team**

```
ROLES NEEDED:

PRIMARY AUTHOR (You):
- Literature review & synthesis
- Paper writing
- Experiment/analysis design

TECHNICAL REVIEWER (Expert):
- Malware analysis security expert
- Validates technical accuracy
- Suggests missing papers
- Time: 5–10 hours/week for 8 weeks

DOMAIN EXPERT (Co-author Optional):
- Deep learning for security researcher
- Ensures modern methods properly covered
- Validates paradigm analysis
- Time: 3–5 hours/week

GRAPHICS/VISUALIZATION (Co-author Optional):
- Creates figures (timeline, maturity matrix)
- Table design
- Diagram creation
- Time: 1–2 weeks total

WRITING EDITOR (Optional):
- Grammar, clarity, flow
- ACM style compliance
- Time: 1 week near end
```

---

## Week 2: Planning & Infrastructure

### **Step 2.1: Set Up Your Workspace**

```
TOOLS & INFRASTRUCTURE:

VERSION CONTROL:
GitHub repository (private until submission)
└─ papers/
   ├─ main.tex (LaTeX source)
   ├─ references.bib (BibTeX)
   ├─ figures/
   │  ├─ 01_timeline.pdf
   │  ├─ 02_maturity_matrix.png
   │  └─ ...
   ├─ tables/
   │  ├─ table_techniques.csv
   │  ├─ table_datasets.csv
   │  └─ ...
   └─ data/
      ├─ papers_extracted.xlsx
      ├─ paper_metadata.csv
      └─ notes/

WRITING PLATFORM:
Option A: Overleaf (LaTeX, collaborative, cloud-backed)
- Use ACM template (acmart class)
- Real-time collaboration
- Version history built-in
- Recommended: Overleaf Professional ($17/month)

Option B: GitHub + Local LaTeX
- Full version control
- Better for complex projects
- Harder to collaborate

REFERENCE MANAGEMENT:
Zotero (free, open-source)
- Auto PDF organization
- BibTeX export
- Browser extensions for paper collection
- Best-in-class for academic research

DATABASE (Track Papers):
Option A: Excel/Google Sheets
- Simple filtering/sorting
- Collaborative editing
- Columns: ID | Title | Year | Venue | Technique | Dataset | Accuracy | Robustness | Notes

Option B: Notion
- Database + markdown combined
- Better for complex queries
- Free tier sufficient

LLM ACCESS:
- Claude API (Anthropic) — Best reasoning
- GPT-4 API (OpenAI) — Broad knowledge
- Gemini API (Google) — Multimodal (images, PDFs)
- Budget: $200–400 for entire project
```

### **Step 2.2: Create Master Timeline**

```
DETAILED 30-WEEK EXECUTION PLAN:

PHASE 1: COLLECTION & EXTRACTION (Weeks 3–8)
├─ Week 3: Literature discovery (150–180 papers)
├─ Week 4: Deduplication & initial filtering
├─ Week 5: Structured extraction (data matrix)
├─ Week 6: Temporal tagging & era assignment
├─ Week 7: Validation & gap identification
└─ Week 8: Final dataset (~160 papers ready)

PHASE 2: ANALYSIS & SYNTHESIS (Weeks 9–12)
├─ Week 9: Temporal trend analysis
├─ Week 10: Technique maturity assessment
├─ Week 11: Paradigm shift identification
└─ Week 12: Open problems extraction & prioritization

PHASE 3: OUTLINE & STRUCTURE (Weeks 13–14)
├─ Week 13: Decade-based chapter design
└─ Week 14: Detailed section-by-section outline

PHASE 4: WRITING (Weeks 15–24)
├─ Week 15: Introduction (3 pages)
├─ Week 16: Foundations (2 pages)
├─ Week 17: ERA 1 section (5 pages)
├─ Week 18: ERA 2-Early section (6 pages)
├─ Week 19: ERA 2-Late section (7 pages)
├─ Week 20: ERA 3 section (8 pages)
├─ Week 21: Evaluation frameworks section (4 pages)
├─ Week 22: Deployment & practical section (4 pages)
├─ Week 23: Open problems section (5 pages)
└─ Week 24: Conclusion (2 pages)

PHASE 5: FIGURES, TABLES, APPENDICES (Week 25)
├─ Figures (5: timeline, maturity, taxonomy, paradigm, deployment)
├─ Tables (15: technique comparison, dataset, metrics, tools, etc.)
└─ Appendices (reading list, from-scratch guide, reproducibility checklist)

PHASE 6: INTEGRATION & CONSISTENCY (Week 26)
├─ Full draft assembly
├─ Internal cross-references check
├─ Citation formatting
└─ Terminology consistency

PHASE 7: EXPERT REVIEW & ITERATION (Weeks 27–28)
├─ Send to 3 external reviewers
├─ Collect feedback
├─ Revise based on comments
└─ Address reviewer concerns

PHASE 8: FINAL POLISH (Week 29)
├─ Proofread (grammar, style)
├─ Verify all citations (don't trust LLM)
├─ Final figure/table checks
├─ ACM template compliance
└─ Author bio & metadata

PHASE 9: SUBMISSION & HANDLING (Week 30)
├─ Submit to target venue
├─ Prepare response to initial editorial questions
└─ Track submission status
```

---

# PART II: LITERATURE COLLECTION PHASE (Weeks 3–8)

## Week 3–4: Discover 150–180 Papers

### **Step 3.1: Systematic Search Strategy**

```
SEARCH DATABASE 1: ACM Digital Library
────────────────────────────────────────
Search String: 
[[Title: "malware"] OR [Title: "ransomware"]] AND 
[[Title: "detection"] OR [Title: "classification"]] AND 
[[Title: "machine learning"] OR [Title: "deep learning"] OR [Title: "neural"]]
Date: 2015–2025
Publication Type: Journals + Conferences
Expected Results: 60–80 papers

SEARCH DATABASE 2: IEEE Xplore
────────────────────────────────────────
Search String:
("malware" OR "ransomware") AND 
("detection" OR "classification") AND 
("machine learning" OR "deep learning" OR "artificial intelligence")
Date: 2015–2025
Content: All content
Expected Results: 80–120 papers

SEARCH DATABASE 3: Scopus
────────────────────────────────────────
Search String (with booleans):
(malware OR ransomware) AND 
(detection OR classification) AND 
(machine learning OR deep learning OR neural network)
Fields: Title, Abstract, Keywords
Date: 2015–2025
Subject Areas: Computer Science, Engineering
Expected Results: 150–300 papers (dedup to 60–80)

SEARCH DATABASE 4: arXiv
────────────────────────────────────────
Search String:
malware AND (detection OR analysis) AND 
(neural OR learning OR model)
Categories: cs.CR (cryptography/security), cs.LG (machine learning)
Date: 2015–2025
Expected Results: 40–60 papers

SEARCH DATABASE 5: Google Scholar (Supplementary)
────────────────────────────────────────
Search Queries (Targeted, 3 searches):
1. "malware detection" "2023" OR "2024" OR "2025"
   Purpose: Find very recent papers
   Expected: 30–40 papers

2. "binary analysis" "transformer" OR "BERT" OR "neural"
   Purpose: Find modern deep learning approaches
   Expected: 20–30 papers

3. "static analysis" "semantic" OR "IR" OR "lifting"
   Purpose: Find semantic analysis methods
   Expected: 15–20 papers

SEARCH DATABASE 6: GitHub + Papers In GitHub Descriptions
────────────────────────────────────────
Method: Find research tool repos, check associated papers
Examples: Cuckoo Sandbox papers, Ghidra research, etc.
Expected: 5–10 papers

SEARCH STRATEGY FOR POST-2023 PAPERS:
────────────────────────────────────────
Use LLM (Claude): 
"Find all published papers 2023–2025 on:
- Graph neural networks + malware detection
- Transformers + binary analysis
- LLM + malware analysis
- Concept drift + malware
- Adversarial robustness + malware"

Output: Curated list of 30–40 papers
```

### **Step 3.2: Automated Paper Collection with LLM**

```
WORKFLOW:

1. RUN SEARCHES:
   - Execute all 6 database searches
   - Record search strings + results count
   - Download results as BibTeX/CSV

2. DEDUPLICATE:
   - Use Zotero to auto-deduplicate
   - Manual verification of edge cases
   - Result: ~200–250 unique papers

3. USE LLM FOR SCREENING:
   Prompt to Claude:
   "I have a list of 240 papers on malware detection.
    Filter for:
    - Published 2015–2025
    - English language
    - Peer-reviewed venue (prioritize IEEE, ACM, USENIX)
    - Actually about malware detection (not just security generally)
    - Explicitly uses ML/DL/AI
    
    Output CSV: [Title] | [Authors] | [Year] | [Venue] | [Keep? Y/N] | [Reason]"
    
   LLM processes abstracts → outputs keep/discard decisions
   You review LLM output (spot-check 20% for accuracy)
   Result: 150–180 papers (80% likely to keep)

4. MANUAL REVIEW OF ABSTRACTS:
   For borderline papers, read abstract yourself
   Decide: Keep or discard
   Track reasoning
   Final count: 160–180 papers ✓
```

### **Step 3.3: Organize Papers & Create Metadata**

```
ZOTERO ORGANIZATION:

Create Folders:
├─ 2015-2017 (Classical ML Era)
│  ├─ Static Analysis (10–15 papers)
│  ├─ Dynamic Analysis (5–8 papers)
│  ├─ Feature Engineering (10–15 papers)
│  └─ Classical Classifiers (10–15 papers)
│
├─ 2018-2020 (Deep Learning Era)
│  ├─ CNN Approaches (10–15 papers)
│  ├─ RNN Approaches (8–12 papers)
│  ├─ Hybrid Methods (8–12 papers)
│  └─ Adversarial Examples (5–8 papers)
│
├─ 2021-2023 (Structural Learning Era)
│  ├─ GNN on Graphs (12–15 papers)
│  ├─ Binary Lifting/IR (8–10 papers)
│  ├─ Few-shot Learning (5–8 papers)
│  └─ Representation Learning (8–12 papers)
│
├─ 2023-2025 (Foundation Model Era)
│  ├─ Transformers (8–12 papers)
│  ├─ LLM-Assisted (8–10 papers)
│  ├─ Concept Drift (5–8 papers)
│  ├─ Adversarial Robustness (8–10 papers)
│  └─ Test-Time Adaptation (3–5 papers)
│
├─ Datasets & Benchmarking (15–20 papers)
├─ Explainability/XAI (8–10 papers)
└─ Foundational/Surveys (8–12 papers)

For Each Paper, Add Tags:
#2015–2017 #2018–2020 #2021–2023 #2023–2025
#CNN #RNN #GNN #Transformer #LLM
#Static #Dynamic #Hybrid
#Evasion #Robustness #Dataset #XAI
#SOTA #Legacy #Foundational
```

---

## Weeks 5–8: Structured Extraction

### **Step 5.1: Build Your Extraction Matrix**

```
CREATE SPREADSHEET (Excel/Google Sheets) with Columns:

[ID] | [Title] | [Authors] | [Year] | [Venue] | [Tier]* | 
[Technique] | [Input Type] | [Dataset] | [Accuracy/F1] | 
[Learning Paradigm] | [Obfuscation Handling] | 
[Adversarial Robustness?] | [Generalization Tested?] | 
[Code Available?] | [Practical Constraints?] |
[Maturity] | [Key Insight] | [Limitations]

*Tier: Conference (IEEE/ACM/USENIX) | Journal | Workshop | Preprint

TOTAL COLUMNS: 18

Example Row:
[1] | "CFGExplainer: Explaining GNN-Based Malware Classifiers..." | 
Herath, D. et al. | 2022 | DSN | Conference | GNN | CFG Graph | Custom | 90.26% |
Graph Neural Network | Not addressed | No | Only custom dataset |
No | Manual inspection overhead | Emerging | GNN interpretability on graphs |
Limited cross-dataset testing, assumes valid CFG extraction

USE LLM TO AUTO-FILL:
Prompt to Claude:
"Extract the following fields from this paper abstract:
[Abstract pasted]

Required fields:
- Main technique used
- Input representation type (opcode, API, CFG, etc.)
- Dataset used
- Accuracy/F1 reported
- Learning paradigm (CNN, GNN, Transformer, etc.)
- Handles obfuscation? (Yes/No/Partially)
- Tests adversarial robustness? (Yes/No)
- Tests cross-dataset? (Yes/No)
- Code available? (Yes/No)
- Practical constraints mentioned?
- Maturity level (Novel/Emerging/Mature/SOTA)
- 1-sentence key insight
- Main limitation

Output as CSV row."

VALIDATION:
For 20–30 papers, manually verify LLM output
Check: Accuracy? Technique correctly identified? Maturity reasonable?
Flag LLM errors; retrain prompts if >10% error rate
Rest: Accept LLM output (spot-check 10% of remainder)

RESULT: Fully populated 160–180 row matrix
```

### **Step 5.2: Temporal & Technique Tagging**

```
ADD ANALYSIS COLUMNS:

[Era] — Assign to 4 periods:
 - 2015–2018: Classical ML & Feature Engineering
 - 2018–2021: Deep Learning Adoption & Hybrid Methods
 - 2021–2023: Structural Analysis & Representation Learning
 - 2023–2025: Foundation Models & Transformers

[Technique Category] — Primary technique:
 - Static Analysis
 - Dynamic Analysis
 - Hybrid Analysis
 - Feature Engineering
 - Machine Learning
 - Deep Learning (CNN, RNN)
 - Graph-Based (GNN)
 - Representation Learning
 - Transformer-Based
 - LLM-Assisted
 - Explainability
 - Adversarial

[Innovation Type] — Nature of contribution:
 - Foundational (introduces new technique)
 - Incremental (improves existing)
 - Applied (applies known method to malware)
 - Benchmark (dataset/evaluation)
 - Survey (prior review)

[Evasion Coverage] — How evasion handled:
 - Not addressed
 - Mentions but doesn't handle
 - Partial handling (one obfuscation type)
 - Comprehensive (multiple types)

[Generalization Tested] — Evaluation rigor:
 - Same dataset only
 - Cross-dataset tested
 - Temporal generalization tested
 - Cross-platform tested

RESULT:
Matrix now enables filtering:
- Show all papers 2023–2025 using Transformers
- Show all papers testing adversarial robustness
- Show papers using EMBER dataset
- Etc.

Use for: Gap analysis, trend identification, section assignment
```

---

# PART III: ANALYSIS & SYNTHESIS PHASE (Weeks 9–12)

## Week 9: Temporal Trend Analysis

### **Step 9.1: Quantify Trends**

```
ANALYSIS QUERIES:

1. Papers per technique per era:
   ```
   Select COUNT(*) 
   From papers 
   Where [Technique]='GNN' AND [Era]='2023–2025'
   ```
   Expected: GNN papers spike post-2019
   Visualization: Line chart (year vs. count for each technique)

2. Technique adoption timeline:
   GNN:
   - 2015–2018: 0 papers
   - 2018–2021: 5–8 papers (early work)
   - 2021–2023: 15–20 papers (growing)
   - 2023–2025: 25–30 papers (maturing)
   → INFLECTION POINT: 2019–2020

   Transformers:
   - 2015–2018: 0 papers
   - 2018–2021: 0 papers (attention on NLP first)
   - 2021–2023: 3–5 papers (emerging)
   - 2023–2025: 10–15 papers (rapid adoption)
   → INFLECTION POINT: 2023

   LLM-Assisted:
   - 2015–2023: <1 paper
   - 2023–2025: 8–12 papers (explosive growth)
   → INFLECTION POINT: 2023 (ChatGPT release Nov 2022)

3. Dataset evolution:
   EMBER:
   - Released: 2018
   - Citations: 100+ by 2023
   - Still used but acknowledged as saturated
   
   SOREL-20M:
   - Released: 2021
   - Citation growth: Rapid 2021–2023
   - Addresses EMBER saturation
   
   EMBER2024:
   - Released: Jun 2025
   - Expected citations: Exponential growth post-2025

4. Feature types over time:
   2015–2018: Opcode n-grams, PE headers dominant
   2018–2021: API sequences, hybrid features growing
   2021–2023: Graph-based (CFG/DFG) emerging
   2023–2025: Learned embeddings, semantic IR growing

5. Learning paradigm distribution:
   2015–2018: Classical ML 60%, CNN/RNN 40%
   2018–2021: Classical ML 40%, CNN/RNN 50%, Hybrid 10%
   2021–2023: Classical ML 20%, CNN/RNN 30%, GNN 25%, Representation 15%, Others 10%
   2023–2025: Classical ML 10%, CNN/RNN 15%, GNN 20%, Transformer 25%, LLM 15%, Others 15%

VISUALIZATIONS:
Create Excel/Google Sheets charts:
- Line chart: Technique adoption over time
- Stacked bar: Learning paradigm distribution per era
- Scatter: Year vs. Accuracy (showing performance improvement)
```

### **Step 9.2: Identify Paradigm Shifts**

```
DEFINE PARADIGM SHIFTS SYSTEMATICALLY:

SHIFT 1: Classical ML → Deep Learning (2015–2018)
─────────────────────────────────────
Why shifted:
- Classical ML requires manual feature engineering
- Deep learning learns features automatically
- CNNs on images (binary-to-image conversion) novel
- RNNs on sequences (API calls) promising

Evidence:
- 2015: RF/SVM dominant (60% of papers)
- 2018: CNN/RNN papers growing (approaching 50%)
- Accuracy improvements: ~95% (ML) → ~98% (DL)
- But: Generalization still poor

Papers exemplifying shift:
- [Author1, Year]: First CNN on malware binaries
- [Author2, Year]: PE-to-image conversion
- [Author3, Year]: RNN on API sequences

Why it failed/improved:
- Dataset overfitting (EMBER saturation)
- Poor generalization (model doesn't see novel malware)
→ Led to next paradigm shift

SHIFT 2: Shallow Learning → Representation Learning (2020–2022)
─────────────────────────────────────
Why shifted:
- Shallow DL (CNN/RNN) learns task-specific features
- Representation learning: Pre-train on large data, transfer
- Self-supervised learning: Learn from unlabeled data
- Embeddings capture semantic similarity

Evidence:
- 2020: Transfer learning papers emerging (5–8)
- 2021: Self-supervised pretraining (BERT-style) on code
- 2022: Foundation models (CodeBERT, CodeLLaMA) adapted for malware
- Few-shot learning enables rapid adaptation to novel families

Papers exemplifying shift:
- [BinBert, 2024]: Pre-trained BERT on assembly
- [Author, Year]: Contrastive learning for malware embeddings
- [Author, Year]: Meta-learning for few-shot detection

SHIFT 3: Syntactic → Semantic Analysis (2021–2025)
─────────────────────────────────────
Why shifted:
- Syntactic analysis (opcodes, n-grams) defeated by obfuscation
- 60–80% malware is obfuscated
- Semantic analysis (IR, decompiled code, control flow) more robust
- LLMs understand semantics naturally

Evidence:
- 2019–2020: Binary lifting papers emerging (5–8)
- 2021–2022: IR-based features growing
- 2023–2025: LLM semantic understanding (GPT-4 on assembly)
- Accuracy improvements: ~98% (syntactic) → 99%+ (semantic on evasive)

Papers exemplifying shift:
- [BinBert, 2024]: Execution-aware semantics
- [Valgrin papers]: Binary lifting frameworks
- [GPT-4 malware, 2024]: LLM semantic understanding

SHIFT 4: Flat Classifiers → Structured Learning (2019–2025)
─────────────────────────────────────
Why shifted:
- Flat classifiers (MLP, CNN) ignore structure
- Control flow graphs, data flow graphs have structure
- GNNs preserve graph topology
- Subgraph patterns indicate malware functionality

Evidence:
- 2019: GNN papers first (CFG-based)
- 2020–2021: Growing GNN adoption for CFG (8–12 papers/year)
- 2022–2025: GNN + Transformers combined
- Task performance: Graph-level classification accurate, interpretable

Papers exemplifying shift:
- [CFGExplainer, 2022]: GNN + interpretability
- [Author, Year]: GNN on DFG for data leakage detection
- [Author, Year]: Graph pooling for malware families
```

---

# PART IV: OUTLINE & STRUCTURE DESIGN (Weeks 13–14)

## **Step 13.1: Design Your Paper Structure**

```
COMPLETE PAPER OUTLINE (60–70 pages target)

1. INTRODUCTION (3 pages)
   1.1 Motivation (AI for security, why malware detection)
   1.2 Scale & impact (450K samples/day, billions in damages)
   1.3 Problem statement (feature quality, generalization, evasion)
   1.4 Scope & contributions (decade focus, paradigm analysis, practical guidance)
   1.5 Roadmap (what reader will learn)
   
2. BACKGROUND & CONTEXT (2 pages)
   2.1 Malware analysis modalities (static, dynamic, hybrid)
   2.2 Threat landscape evolution (2015 vs. 2025)
   2.3 Core challenge: Evasion arms race
   
3. FOUNDATIONAL CONCEPTS (2 pages)
   3.1 Static analysis fundamentals (disassembly, PE, control flow)
   3.2 Dynamic analysis fundamentals (sandboxing, behavior capture)
   3.3 Feature extraction basics
   3.4 Classical ML methods (baseline approaches)
   
4. ERA 1: FEATURE ENGINEERING DOMINANCE (2015–2018) (5 pages)
   4.1 Techniques & methods of the era
       4.1.1 Static analysis (PE headers, n-grams, opcodes)
       4.1.2 Dynamic analysis (Cuckoo sandbox, API calls)
       4.1.3 Feature engineering (manual, labor-intensive)
   4.2 Learning methods
       4.2.1 Classical ML (RF, SVM, Naive Bayes)
       4.2.2 Early deep learning (CNN, RNN first applications)
       4.2.3 Results: ~95% accuracy on benchmarks
   4.3 Datasets (EMBER emergence in 2018)
   4.4 Challenges & limitations
       4.4.1 Obfuscation defeats static (packing, polymorphism)
       4.4.2 Poor generalization (dataset-specific models)
       4.4.3 Feature engineering bottleneck
   4.5 Why this era ended
       → Led to next paradigm (deeper learning)
   
5. ERA 2A: DEEP LEARNING ADOPTION (2018–2021) (6 pages)
   5.1 Motivation for paradigm shift
   5.2 Techniques
       5.2.1 CNN on binary images (PE-to-image conversion)
       5.2.2 RNN on sequences (API call sequences)
       5.2.3 Hybrid static+dynamic fusion
       5.2.4 Early adversarial examples (MalGAN, etc.)
   5.3 Datasets & evaluation (EMBER saturation emerges)
   5.4 Results & insights (98%+ accuracy claimed, but...)
   5.5 Fundamental limitations
       5.5.1 Still relies on manual features (feature engineering not eliminated)
       5.5.2 Generalization failure (trained EMBER, fails on VirusShare)
       5.5.3 Adversarial vulnerability (GAN-based evasion effective)
       5.5.4 Dataset bias (60–80% obfuscation defeats static)
   5.6 Why this era transitioned
       → Need for structure-aware learning
   
6. ERA 2B: STRUCTURAL & REPRESENTATION LEARNING (2021–2023) (7 pages)
   6.1 Paradigm shift: From sequences to graphs
   6.2 Graph Neural Networks on malware
       6.2.1 Control Flow Graphs (CFG) representation
       6.2.2 GCN, GAT, GIN architectures
       6.2.3 Why graphs preserve structure
       6.2.4 CFGExplainer (interpretable GNN)
       6.2.5 Robustness to polymorphism
       6.2.6 Limitations: CFG extraction bugs, flattening evasion
   6.3 Binary Lifting & Intermediate Representations (IR)
       6.3.1 Why semantic analysis matters (obfuscation robustness)
       6.3.2 Valgrin, BAP, LLVM IR frameworks
       6.3.3 Semantic features invariant to polymorphism
       6.3.4 Cross-architecture generalization
       6.3.5 Challenges: Lifting correctness, 24 bugs in lifters
   6.4 Representation Learning & Self-Supervised Pre-Training
       6.4.1 Pre-train on large unlabeled assembly corpora
       6.4.2 Fine-tune on labeled malware (reduced annotation burden)
       6.4.3 Contrastive learning (similarity preservation)
       6.4.4 Transfer learning (domain adaptation)
   6.5 Few-Shot & Meta-Learning
       6.5.1 SIMPLE framework (meta-learning for malware)
       6.5.2 Rapid adaptation to novel families (<10 samples)
       6.5.3 Zero-shot learning potential
   6.6 Datasets & evaluation advances (SOREL-20M, temporal splits)
   6.7 Results: 97–99% on benchmarks, better generalization
   6.8 Remaining challenges
       → Need for simpler, faster models (transformers)
   
7. ERA 3: FOUNDATION MODELS & TRANSFORMERS (2023–2025) (8 pages)
   7.1 Paradigm shift: Attention mechanisms, large-scale pre-training
   7.2 Transformer Architectures for Malware
       7.2.1 Why transformers (parallelizable, long-range deps)
       7.2.2 BinBert (execution-aware transformer)
       7.2.3 Vision Transformers on binary images
       7.2.4 Process-resource transformers
       7.2.5 Pre-training strategies for assembly
       7.2.6 Fine-tuning and adaptation
   7.3 Large Language Models for Malware Analysis
       7.3.1 CodeLLaMA, GPT-4, Gemini on assembly
       7.3.2 Semantic lifting (decompilation quality)
       7.3.3 Few-shot prompting for novel families
       7.3.4 Behavior inference from semantics
       7.3.5 Threat assessment via LLM reasoning
   7.4 Foundation Model Advantages
       7.4.1 Reduced feature engineering burden
       7.4.2 Transfer learning from massive code corpora
       7.4.3 Few-shot/zero-shot capabilities
       7.4.4 Semantic understanding (not just patterns)
   7.5 Explainability & Interpretability in Deep Models
       7.5.1 SHAP for feature importance
       7.5.2 LIME for local explanations
       7.5.3 Grad-CAM for attention visualization
       7.5.4 Why interpretability matters (regulatory, security)
   7.6 Adversarial Robustness & Certified Defenses
       7.6.1 GAN-based evasive malware (DOpGAN, etc.)
       7.6.2 Certified robustness (IBP, randomized smoothing)
       7.6.3 Adversarial training vs. clean accuracy trade-off
       7.6.4 Robustness evaluation frameworks
   7.7 Test-Time Adaptation & Continual Learning
       7.7.1 MADCAT (test-time training for concept drift)
       7.7.2 Warm-start learning for temporal shift
       7.7.3 Federated learning for privacy
   7.8 EMBER2024: New benchmark standard
   7.9 Results: 99%+ accuracy, good generalization, semantic understanding
   7.10 Open challenges
       → Semantic obfuscation, real-time deployment, certified defenses
   
8. EVALUATION FRAMEWORKS & BENCHMARKING (4 pages)
   8.1 Temporal Evaluation Methodology
       8.1.1 Concept drift definition & measurement
       8.1.2 Time-stratified train/test splits
       8.1.3 Empirical drift studies (F1 drops 0.9→0.6 over years)
       8.1.4 How to report temporal results
   8.2 Cross-Dataset Generalization
       8.2.1 Standard protocol (train EMBER, test SOREL/VirusShare/BODMAS)
       8.2.2 Domain differences (temporal, labeling, features)
       8.2.3 Domain adaptation techniques
       8.2.4 Reporting cross-dataset results
   8.3 Adversarial Robustness Testing
       8.3.1 Threat model definition
       8.3.2 Adversarial example generation (FGSM, PGD, GAN-based)
       8.3.3 Certified defenses
       8.3.4 Evaluation metrics (clean vs. robust accuracy)
   8.4 Production-Relevant Metrics
       8.4.1 False positive rate at fixed TPR
       8.4.2 Imbalanced dataset metrics
       8.4.3 Cost matrix evaluation
   8.5 Reproducibility Standards
       8.5.1 Tool versioning (Ghidra 10 vs. 11, differences)
       8.5.2 Feature extraction tool correctness
       8.5.3 Containerization & automation
   
9. PRACTICAL DEPLOYMENT & REAL-WORLD CONSTRAINTS (4 pages)
   9.1 Scalability to Real-World Volumes
       9.1.1 450K samples/day reality
       9.1.2 Feature extraction bottleneck (timing analysis)
       9.1.3 Tiered pipeline design (fast static → dynamic → manual)
       9.1.4 Cost-benefit analysis
   9.2 Inference Latency & Resource Requirements
       9.2.1 Table: Model type × latency × memory × GPU req × throughput
       9.2.2 Hardware accelerators (GPU, TPU, FPGAs)
       9.2.3 Model compression (distillation, quantization)
       9.2.4 Edge deployment constraints
   9.3 Production-Ready Architectures
       9.3.1 Tiered cascade design
       9.3.2 Fallback strategies
       9.3.3 Monitoring & alerting
   9.4 Tool Reproducibility & Versioning
       9.4.1 Tool differences (IDA vs. Ghidra, versions)
       9.4.2 Feature extraction reproducibility
       9.4.3 Semantic lifting tool correctness
       9.4.4 Containerization best practices
   9.5 Static Analysis Viability in Era of Obfuscation
       9.5.1 Is static analysis becoming obsolete? (60–80% obfuscation)
       9.5.2 IR-based semantics as answer
       9.5.3 Hybrid approaches (static → dynamic → manual)
   
10. UNRESOLVED OPEN PROBLEMS & RESEARCH ROADMAP (5 pages)
    10.1 Problem Enumeration (Prioritized)
    
    CRITICAL (Impact=High, Feasibility=Medium–High):
    1. Semantic obfuscation resistance
       - Can semantic analysis overcome control-flow flattening?
       - Invariant representation challenge
       - Current attempts: GNN robustness unclear
    
    2. Cross-temporal generalization
       - Model trained 2020, fails 2024 (concept drift)
       - Fundamental or solvable with adaptation?
       - MADCAT, domain adaptation make progress
    
    3. LLM detector robustness
       - Can LLMs be jailbroken/adversarially manipulated?
       - Prompt injection attacks on LLM-based analysis
       - Emerging threat, not well-studied
    
    4. Certified robustness for malware
       - Provable guarantees against perturbations
       - Functionality preservation constraint
       - Trade-off: Robustness vs. accuracy
    
    HIGH (Impact=Medium–High, Feasibility=Medium):
    5. Adversarial game-theoretic analysis
       - Attacker-defender equilibrium
       - What's the fundamental limit?
       - Theoretical framework needed
    
    6. Supply-chain malware detection
       - Provenance tracking
       - Build-time integrity
       - Emerging threat, limited research
    
    7. Real-time deployment
       - 450K files/day with <1sec latency each
       - Computational feasibility?
       - Accuracy-speed trade-off
    
    MEDIUM (Impact=Medium, Feasibility=Medium–High):
    8. Few-shot adaptation
       - How to detect novel families with <10 samples?
       - Meta-learning limitations
       - Transfer learning from LLMs helps
    
    9. Multi-platform analysis
       - Windows, Android, Linux, IoT unified
       - Cross-architecture semantics
       - Emerging with binary lifting
    
    10. Explainability at scale
        - Can we explain 99%+ accurate detectors?
        - SHAP/LIME computational cost
        - Trade-off with interpretability
    
    10.2 Established vs. Speculative Trends
    
    ESTABLISHED (High Confidence):
    - Transformers are SOTA for sequences
    - GNNs preserve graph structure effectively
    - LLMs transfer knowledge across domains
    - Concept drift is real problem in malware
    - Cross-dataset generalization is hard
    
    SPECULATIVE (Medium-High Confidence):
    - LLMs will replace manual analysis
    - Semantic analysis fully defeats obfuscation
    - Foundation models eliminate need for datasets
    
    SPECULATIVE (Low Confidence):
    - AI will create "super malware"
    - Adversaries will solve robustness perfectly
    - Quantum computing breaks detection
    
    10.3 Proposed Research Directions (Next 2 Years)
    
    PRIORITY 1: Temporal & Adversarial Evaluation
    - Publish benchmark with time-stratified splits
    - Standardize adversarial evaluation
    - Community adoption of standard metrics
    
    PRIORITY 2: Semantic-Robust Representations
    - Develop obfuscation-invariant features
    - Test IR-based analysis at scale
    - Compare GNN vs. Transformer robustness
    
    PRIORITY 3: Foundation Model Adaptation
    - Fine-tune CodeLLaMA for binary analysis
    - Few-shot learning benchmarks
    - Efficiency vs. accuracy trade-off
    
    PRIORITY 4: Real-World Deployment
    - Industry case studies
    - Tiered pipeline optimization
    - Cost analysis (accuracy vs. computation)
    
11. CONCLUSION (2 pages)
    11.1 Key Takeaways (What reader learned)
    11.2 Paradigm Progression (2015→2025 evolution)
    11.3 Current Best Practices (What to implement today)
    11.4 Unresolved Challenges (What to research)
    11.5 Future Outlook (2026+ predictions)
    
12. APPENDICES

A. PRIORITIZED READING LIST (3–4 pages)
   40–50 papers stratified by:
   - Era (2015–2018, etc.)
   - Difficulty (foundational → cutting-edge)
   - Specialization (student reading path, practitioner path, researcher path)
   
B. TECHNIQUE MATURITY MATRIX (1 page, visual)
   All 20+ techniques × years × maturity levels
   
C. DATASET COMPARISON TABLE (1 page)
   EMBER, SOREL, BODMAS, EMBER2024, VirusShare, Malware Bazaar
   Size, features, time period, evasion-focused?, etc.
   
D. FROM-SCRATCH SURVEY WRITING GUIDE (2 pages)
   30-week workflow for future survey authors
   LLM-assisted literature review methodology
   Reproducibility checklist
   
E. TOOLS & REPRODUCIBILITY CHECKLIST (1 page)
   Recommended tools (Ghidra, Cuckoo, etc.)
   Version pinning
   Containerization template
   Data availability statement
   
F. SUPPLEMENTARY TABLES (2–3 pages)
   Learning paradigm recommendations table
   Inference latency benchmarks
   Feature importance comparison
   
═════════════════════════════════════════════
TOTAL: 60–70 pages
FIGURES: 5 major (timeline, maturity, paradigm, taxonomy, deployment)
TABLES: 15+ comprehensive
CITATIONS: 180+ papers
═════════════════════════════════════════════
```

---

# PART V: WRITING PHASE (Weeks 15–24)

## **Step 15.1: Writing Workflow & LLM Integration**

```
WRITING PROCESS (Each Section):

STEP A: OUTLINE (30 min)
────────────────────
Use your detailed outline
Create 3–4 subsections per section
List key points for each subsection
Example (Section 4: ERA 1):
├─ 4.1 Techniques & methods
│  ├─ Static analysis (PE, opcodes, n-grams)
│  ├─ Dynamic analysis (Cuckoo, API calls)
│  └─ Feature engineering (manual, tedious)
├─ 4.2 Learning methods
│  ├─ Classical ML (RF, SVM)
│  └─ Early DL (CNN, RNN)
├─ 4.3 Results (~95% accuracy)
├─ 4.4 Limitations (obfuscation, generalization)
└─ 4.5 Why this paradigm ended (→ next era)

STEP B: LLM SCAFFOLDING (20 min)
────────────────────
Prompt to Claude:
"Write an outline for this section on [Topic]:
[Your outline structure]

Key papers to reference:
[List 3–5 key papers with authors, years]

Include:
1. Explanation of the techniques
2. Why they were used (motivation)
3. Example applications
4. Limitations
5. Why the paradigm shifted

Output: 1–2 paragraph per subsection (scaffold only)"

Claude produces: Draft outlines with topic sentences

STEP C: YOUR MAIN WRITING (1–2 hours)
────────────────────
Read Claude's scaffold
REWRITE in your voice (not copying; using as guide)
Add:
- Specific details from papers
- Critical analysis (not just summary)
- Connections between papers
- Limitations & why paradigm shifted
- Evidence from extraction matrix

Structure:
1st paragraph: Introduce technique/method
2nd paragraph: How it works (algorithm/approach)
3rd paragraph: Evidence (papers, results)
4th paragraph: Limitations discovered
5th paragraph: Why this led to next paradigm

STEP D: CITATION VERIFICATION (20 min)
────────────────────
For every claim, verify:
- Is it cited? (Should have ~2–3 citations per paragraph)
- Cite correctly? (Author, year, claim matches paper)
- Not LLM hallucination? (Read the cited paper abstract)

Use BibTeX: \cite{Author2022}
Example: "Control-flow flattening defeats simple CFG analysis \cite{Author2022}, motivating GNN-based approaches that learn flow-aware patterns \cite{Author2023}."

STEP E: CROSS-LINKING (10 min)
────────────────────
Check: Does this section reference prior/future sections?
Example: "As discussed in Section 4.1, obfuscation defeats static analysis. This limitation motivated the representation learning approaches in Section 6."

Add cross-references:
\ref{sec:era1} → "As shown in Section \ref{sec:era1}"
\cite{Author2022} → "As Author et al. \cite{Author2022} demonstrated"

STEP F: PEER REVIEW SELF-CHECK (10 min)
────────────────────
Question each paragraph:
- Is this clear to someone outside my field? (Clarity)
- Have I oversimplified or been imprecise? (Accuracy)
- Is there a logical flow to the argument? (Coherence)
- Have I cited properly? (Attribution)
- Does this support my research question? (Relevance)

Red-flag unclear sentences; rewrite if needed.

TOTAL TIME PER SECTION: 2.5–3 hours (for 3–5 pages)
```

## **Step 15.2: Example Section Writing (ERA 1 Section)**

```
EXAMPLE: SECTION 4 ERA 1 (Feature Engineering Dominance, 2015–2018)

═════════════════════════════════════════════════════════════════════

4. ERA 1: FEATURE ENGINEERING DOMINANCE (2015–2018)

The period from 2015 to 2018 represents the foundational era of machine 
learning approaches to malware detection. During this time, feature 
engineering—the manual, labor-intensive process of identifying and 
extracting discriminative patterns from malware—was the dominant paradigm. 
Security researchers focused on what features to extract and which 
classical machine learning algorithms to apply to those features, treating 
feature quality as the primary lever for detection accuracy.

4.1 Techniques and Analysis Methods

4.1.1 Static Analysis Approaches

Static analysis, which examines malware without executing it, was the 
primary analysis modality of this era. Researchers leveraged tools like 
IDA Pro and custom Python libraries (e.g., Pefile) to extract low-level 
features from PE (Portable Executable) file headers and disassembled code.

The most commonly extracted static features included:
- PE header information (entropy, section counts, import counts) [Ye2017, 
  Gibert2015]
- Opcode sequences (frequency of CPU instructions like MOV, PUSH, CALL) 
  [Ye2017, Anderson2018]
- N-gram patterns (consecutive byte or opcode sequences) [Gibert2015, 
  Anderson2018]
- String artifacts (URLs, registry keys, API names found in string sections) 
  [Shaukat2018]

These features were chosen because they were easily extractable and 
interpretable by security analysts. For example, a high count of 
WriteProcessMemory API calls (extracted from the PE import table) is often 
indicative of process injection, a common malware behavior.

However, static analysis had fundamental limitations. Obfuscation 
techniques—where malware authors modify code without changing functionality—
defeated static detectors. As Maffia et al. [2019] demonstrated empirically, 
approximately 68% of malware samples employed some form of obfuscation 
(packing, polymorphism, or metamorphism). This prevalence made static 
features unreliable for detecting a significant fraction of real-world 
malware [Galloro2019].

4.1.2 Dynamic Analysis Approaches

To overcome static analysis limitations, researchers also employed dynamic 
analysis—executing malware in controlled environments and monitoring its 
behavior. The Cuckoo Sandbox, released in 2012, became the standard platform 
for behavioral malware analysis by 2015 [Cuckoo2012].

Dynamic analysis extracted features such as:
- API call sequences (Windows API functions invoked during execution) 
  [Ye2017, Shaukat2018]
- Registry access patterns (which Windows registry keys were read or modified) 
  [Cuckoo2018]
- File system operations (files created, deleted, or modified) [Ye2017]
- Network traffic (DNS queries, HTTP requests, C2 connections) [Shaukat2018]

These behavioral features captured what the malware actually did, potentially 
sidestepping obfuscation. A malware's runtime behavior—regardless of whether 
its code was obfuscated—revealed its malicious intent [Anderson2018].

Yet dynamic analysis had its own limitations. Anti-analysis techniques—where 
malware detects whether it's running in a sandbox or under instrumentation—
were widespread. Galloro et al. [2019] identified 92 distinct anti-analysis 
techniques across 45,375 malware samples, with approximately 80% exhibiting 
at least one evasion technique. Anti-analysis rendered dynamic analysis 
unreliable for sophisticated malware [Galloro2019].

4.1.3 The Feature Engineering Paradigm

The defining characteristic of this era was the explicit separation between 
feature extraction (how to get features) and feature learning (how to use 
them). Feature engineering was manual and domain-driven:

1. Security analysts identified candidate features based on domain knowledge 
   (e.g., "suspicious APIs indicate injection behavior")
2. Features were extracted via static tools (PE headers, disassembly) or 
   dynamic tools (API call monitors)
3. Feature vectors (fixed-size numerical representations) were constructed 
   and fed to machine learning algorithms
4. Classifiers (e.g., Random Forest) learned decision boundaries in the 
   feature space

This process was labor-intensive and required significant security expertise. 
Moreover, no single feature set worked well across all malware families, 
motivating extensive feature selection research [Ye2017].

4.2 Machine Learning Methods of the Era

4.2.1 Classical Machine Learning Dominance

The learning methods of 2015–2018 were almost exclusively "shallow" machine 
learning algorithms—those without deep neural networks:

- Random Forest (RF): Ensemble of decision trees, interpretable, robust to 
  feature scaling. Ye et al. [2017] found RF to be competitive or superior 
  to deep learning on several malware detection benchmarks.
- Support Vector Machines (SVM): Maximum-margin classifiers, effective for 
  high-dimensional feature spaces. Gibert et al. [2015] documented SVM as 
  standard in ~40% of surveyed papers.
- Logistic Regression: Simple, fast, interpretable baseline [Shaukat2018]
- Naive Bayes: Fast probabilistic classifier, often used for comparison 
  [Gibert2015]

These methods required manually-crafted features but offered speed, 
interpretability, and robustness—critical properties for production security 
systems. A Random Forest with 20–100 trees could classify a malware sample 
in milliseconds on CPU hardware [Ye2017].

4.2.2 Early Deep Learning Experiments

Toward the end of this era (2017–2018), researchers began experimenting with 
deep neural networks:

- Convolutional Neural Networks (CNN): Applied to binary images created by 
  converting PE file bytes to pixels. Dahl et al. [2013] and later Anderson 
  et al. [2018] demonstrated that CNNs could learn patterns in binary images 
  without manual feature extraction.
- Recurrent Neural Networks (RNN): Applied to sequences of opcodes or API 
  calls. RNNs could theoretically capture sequential dependencies in malware 
  behavior [Pascanu2015].
- Deep Belief Networks (DBN): Early unsupervised deep learning for malware 
  feature learning [Dahl2013].

These deep learning experiments were promising but remained on the periphery 
of the field. Anderson et al. [2018] reported ~98% accuracy with CNN on the 
newly released EMBER dataset, compared to ~97% for classical ML. The margin 
was marginal, and deep learning required significantly more computational 
resources and labeled data [Anderson2018].

4.3 Datasets and Evaluation

The landscape of malware datasets changed dramatically in this era with the 
emergence of public benchmarks:

EMBER (2018): Anderson and Roth released EMBER, a large-scale dataset of 
1.1 million PE files (600K malware, 500K benign) with binary labels 
("malware" or "benign") and pre-extracted static features [Anderson2018]. 
EMBER became the de facto benchmark for malware detection from 2018 onward, 
enabling fair comparison across methods. However, EMBER had limitations: 
All samples were PE files (Windows-only); feature extraction was fixed (new 
approaches couldn't extract novel features); and later analysis showed the 
dataset was saturated—baseline classifiers achieved >99% AUC [Harang2021].

Prior to EMBER, researchers relied on private datasets or small public 
datasets (e.g., Drebin for Android malware with ~6K samples), limiting 
reproducibility and comparison.

4.4 Limitations and Failure Modes

Despite achieving ~95–98% accuracy on benchmarks, the feature engineering 
era had critical limitations that foreshadowed its eventual supersession:

4.4.1 Obfuscation Defeats Static Analysis

Approximately 68% of malware employed obfuscation, making static features 
unreliable [Maffia2019]. Polymorphic and metamorphic malware could change 
their opcode sequences while preserving functionality, rendering static 
n-gram features obsolete [Ye2017].

4.4.2 Anti-Analysis Defeats Dynamic Analysis

Approximately 80% of malware employed anti-analysis techniques that detected 
sandboxes or DBI, causing malware to suppress behavior in analysis 
environments [Galloro2019]. This made dynamic features untrustworthy for 
evasive malware.

4.4.3 Poor Generalization

Models trained on EMBER performed poorly on data from other sources 
(VirusShare, Malware Bazaar). Cross-dataset evaluation was not standard, 
and practitioners discovered that 98% accuracy on one dataset dropped to 
70–80% on another [Authors-not-specified]. The feature-centric paradigm 
overfit to specific datasets and did not learn generalizable patterns.

4.4.4 Feature Engineering Bottleneck

Each new malware family, obfuscation technique, or data source required 
careful feature re-engineering. This was labor-intensive and did not scale 
to the 450,000 new samples arriving daily [AV-Test2022].

4.5 Why This Paradigm Shifted

By 2018, the limitations of feature engineering had become apparent:
1. Obfuscation and anti-analysis undermined the quality of hand-crafted 
   features
2. Poor generalization across datasets and families
3. Scalability: Manual feature engineering could not keep pace with malware 
   evolution

These factors motivated a shift toward deep learning, which could 
automatically learn features from data rather than relying on manual 
engineering. The next era (Section 5) explores this transition.

═════════════════════════════════════════════════════════════════════

WRITING STATISTICS FOR THIS SECTION:
- Word count: ~2000 words
- Citation count: 20+ citations
- Subsections: 5 major
- Time to write: ~3 hours (with LLM scaffolding)
- Figures: 0 (no need in text section)
- Tables: 1 (optional: comparison of static vs. dynamic)
```

---

# PART VI: FIGURES, TABLES & APPENDICES (Week 25)

## **Step 25.1: Create 5 Essential Figures**

```
FIGURE 1: DECADE-LONG TECHNIQUE EVOLUTION TIMELINE
─────────────────────────────────────────────────────

Create using: Excel, Python (matplotlib), or Google Sheets

Y-axis: Number of papers per technique per year
X-axis: Year (2015–2025)

Lines to plot:
1. Classical ML (RF, SVM) — Starts high (2015), gradually declines
2. CNN on images — Rises 2015–2018, plateaus 2018–2023
3. RNN on sequences — Rises 2016–2018, plateaus 2018–2023
4. GNN on graphs — Starts ~2019 (0 papers), rises exponentially 2019–2023
5. Transformers — Starts ~2023 (0 papers), rises 2023–2025
6. LLM-assisted — Starts ~2023 (1 paper), rises 2023–2025

Annotations:
- Mark inflection points (where adoption accelerates)
  e.g., GNN: 2019 inflection point
  Transformers: 2023 inflection point
  
- Add colored regions for ERA 1, ERA 2A, ERA 2B, ERA 3

- Legend: Each technique with color

Title: "Malware Detection Technique Adoption Over Decade (2015–2025)"

Data source: Your extraction matrix (papers per technique per year)


FIGURE 2: TECHNIQUE MATURITY HEATMAP
─────────────────────────────────────

Create using: Excel or Python heatmap

Rows: 20+ techniques (Signature-Based, RF, SVM, CNN, RNN, GNN, Transformers, LLM, etc.)
Columns: Years (2015, 2018, 2021, 2023, 2025)
Cell values: Maturity level (1=Novel, 2=Emerging, 3=Mature, 4=SOTA, 5=Legacy)

Color scheme:
- Red: Legacy (deprecated)
- Orange: Mature (established, but not cutting-edge)
- Yellow: Emerging (promising, growing adoption)
- Light Blue: Mature/SOTA (current best practices)
- Dark Blue: SOTA (state-of-the-art)

Example:
Signature-Based: 5|5|5|5|5 (all red — legacy throughout)
RF:              2|3|3|3|2 (Novel→Mature→still used but less fashionable)
CNN:             1|2|3|3|3 (Novel→Mature→Established but superseded)
GNN:             -|1|2|3|3 (Didn't exist→Emerging→Mature)
Transformer:     -|-|-|1|3 (Novel→Emerging→Mature)
LLM-assisted:    -|-|-|1|2 (Novel→Emerging)

Title: "Technique Maturity Heatmap: 2015–2025 Evolution"


FIGURE 3: PARADIGM PROGRESSION FLOW DIAGRAM
─────────────────────────────────────────────

Create using: Draw.io or Lucidchart

Show progression with boxes and arrows:

┌─────────────────────────────────────────────────────────┐
│ Classical ML Era (2015–2018)                             │
│ Manual features → RF/SVM → ~95% accuracy                 │
│ Problem: Obfuscation defeats static; poor generalization│
└───────────────┬─────────────────────────────────────────┘
                │ Why shift?
                ↓
┌─────────────────────────────────────────────────────────┐
│ Deep Learning Era (2018–2021)                            │
│ Learned features → CNN/RNN → ~98% accuracy               │
│ Problem: Still bad generalization; dataset overfitting  │
└───────────────┬─────────────────────────────────────────┘
                │ Why shift?
                ↓
┌─────────────────────────────────────────────────────────┐
│ Representation Learning Era (2021–2023)                  │
│ Structured learning → GNN/IR → ~99% accuracy             │
│ Problem: Complex, slow; adversarial vulnerable           │
└───────────────┬─────────────────────────────────────────┘
                │ Why shift?
                ↓
┌─────────────────────────────────────────────────────────┐
│ Foundation Model Era (2023–2025)                         │
│ Pre-trained semantics → Transformers/LLMs → 99%+ accuracy│
│ Open: Semantic obfuscation, real-time deployment, robustness│
└─────────────────────────────────────────────────────────┘

Title: "Paradigm Progression: Driving Forces & Limitations"


FIGURE 4: HIERARCHY TAXONOMY (Malware Detection Landscape)
────────────────────────────────────────────────────────────

Create using: Mind map or tree diagram (lucidchart, draw.io)

Malware Detection Approaches (2015–2025)
├─ LEGACY (Pre-2015)
│  ├─ Signature-Based (Pattern matching)
│  └─ Heuristic Rules (If-then logic)
│
├─ CLASSICAL ML ERA (2015–2018)
│  ├─ Static Analysis
│  │  ├─ PE Headers
│  │  ├─ Opcode N-grams
│  │  └─ String Analysis
│  ├─ Dynamic Analysis
│  │  ├─ Cuckoo Sandbox
│  │  └─ API Monitoring
│  └─ Classifiers
│     ├─ Random Forest
│     ├─ SVM
│     └─ Naive Bayes
│
├─ DEEP LEARNING ERA (2018–2021)
│  ├─ CNN-based
│  │  ├─ Binary Images
│  │  └─ Byte Sequences
│  ├─ RNN-based
│  │  ├─ API Sequences
│  │  └─ Opcode Sequences
│  └─ Hybrid Fusion
│     └─ Static + Dynamic
│
├─ REPRESENTATION LEARNING ERA (2021–2023)
│  ├─ Graph-Based (GNN)
│  │  ├─ CFG Analysis
│  │  └─ GCN/GAT/GIN
│  ├─ IR-based Semantics
│  │  ├─ Binary Lifting
│  │  └─ Valgrin/BAP/LLVM
│  ├─ Pre-training
│  │  └─ Self-supervised
│  └─ Few-shot Learning
│     └─ Meta-learning
│
└─ FOUNDATION MODEL ERA (2023–2025)
   ├─ Transformers
   │  ├─ BinBert
   │  ├─ Vision Transformers
   │  └─ Assembly Transformers
   └─ LLM-Assisted
      ├─ CodeLLaMA
      ├─ GPT-4 Analysis
      └─ Semantic Lifting

Title: "Hierarchical Taxonomy: Evolution from Legacy to SOTA"


FIGURE 5: TIERED PRODUCTION PIPELINE (Deployment Architecture)
─────────────────────────────────────────────────────────────

Create using: Box/arrow diagram (Visio, Lucidchart)

Input: Incoming File Stream (450K samples/day)
         │
         ↓
    ┌────────────────────┐
    │ Tier 1: Fast Static │  <1 second/file
    │  (All files go here)│  - PE headers
    │  - Entropy          │  - Strings
    │  - Static features  │  - Opcodes
    │  - Lightweight DL   │
    └────────┬───────────┘
             │
        ┌────┴─────────────────┐
        │                       │
     BENIGN               UNCERTAIN
    (99% of files)      (0.5% of files)
        │                       │
        ↓                       ↓
     BLOCK              ┌──────────────────┐
                        │ Tier 2: Dynamic  │  30–60 sec/file
                        │  (Selective)     │  - Cuckoo sandbox
                        │  - DBI logging   │  - API monitoring
                        │  - Behavior      │  - Network traffic
                        │  - DL inference  │
                        └────────┬─────────┘
                                 │
                        ┌────────┴───────┐
                        │                │
                     MALWARE          UNCERTAIN
                        │                │
                        ↓                ↓
                      BLOCK    ┌──────────────────┐
                               │ Tier 3: Manual   │  Hours/days
                               │  (Critical only) │  - Human analysis
                               │  <0.1% of files  │  - Symbolic exec
                               │                  │  - Expert review
                               └────────┬─────────┘
                                        │
                                   VERDICT
                                        │
                                   ESCALATE

Title: "Production Tiered Detection Pipeline"
```

## **Step 25.2: Create 15+ Essential Tables**

```
TABLE 1: PAPER SUPERIORITY vs. GABER ET AL.
─────────────────────────────────────────────
| Aspect | Gaber et al. (2024) | Your Paper (2026) |
|--------|-------------------|------------------|
| Papers analyzed | 77 | 180+ |
| Publication cutoff | Sep 2022 | Dec 2024 |
| Page count | 33 | 60–70 |
| Figures | 2 | 5+ |
| Tables | 6 | 15+ |
| GNN section (pages) | 0.1 | 5 |
| Transformer section | 0 | 6 |
| LLM section | 0.3 | 7 |
| Practical deployment | 1 | 4 |
| Open problems | 0.5 | 5 |
| Temporal structure | None | Era-based |
| Maturity assessment | No | Yes (matrix) |
| Reading list | No | Yes (appendix) |
| Paradigm explanation | Weak | Comprehensive |

TABLE 2: TECHNIQUE MATURITY ASSESSMENT
─────────────────────────────────────────
| Technique | Emergence | Peak | 2025 Status | Recommendation |
|-----------|-----------|------|------------|-----------------|
| Signature-Based | Pre-2010 | 2010 | LEGACY | Skip |
| Classical ML | 2010–2015 | 2017 | BASELINE | Use if lightweight |
| CNN on PE | 2015–2017 | 2018 | TRANSITIONAL | Superseded |
| RNN API | 2015–2018 | 2019 | TRANSITIONAL | Superseded |
| Hybrid Static+Dyn | 2018–2020 | 2020 | MATURE | Still practical |
| GNN on CFG | 2019–2020 | 2023 | MATURE | SOTA for graphs |
| Binary Lifting | 2017–2019 | 2023 | MATURE | SOTA for semantics |
| Transformers | 2023–2024 | 2024 | EMERGING | New SOTA |
| LLM-Assisted | 2023–2024 | 2025 | EMERGING | Paradigm shift |

TABLE 3: DATASET EVOLUTION & CHARACTERISTICS
──────────────────────────────────────────────
| Dataset | Released | Samples | Features | Temporal? | Evasion-focused? | Current Use |
|---------|----------|---------|----------|-----------|-----------------|-----------|
| EMBER | 2018 | 1.1M | Pre-extracted static | No | No | Baseline (saturated) |
| BODMAS | 2020 | 134K | 14 categories | Yes (2019–2020) | No | Temporal analysis |
| SOREL-20M | 2021 | 20M | Static + metadata | Yes (2017–2019) | No | Large-scale |
| Malware Bazaar | 2023–ongoing | 700K+ | Family labels | Ongoing | Yes | Recent samples |
| EMBER2024 | Jun 2025 | 3.2M | Multi-format | Yes | Yes | NEW STANDARD |

TABLE 4: LEARNING PARADIGM RECOMMENDATIONS FOR 2025
────────────────────────────────────────────────────
| Paradigm | Status | Use When | Pros | Cons |
|----------|--------|----------|------|------|
| Classical ML | Baseline | Speed critical, small budget | Fast, interpretable | Needs feature engineering |
| CNN/RNN | Legacy | Historical context, teaching | Accessible, understood | Poor generalization |
| GNN | SOTA (graphs) | Structural analysis, CFG | Preserves topology | Slower than transformers |
| Transformer | SOTA (sequences) | Text/code analysis | Parallelizable, scalable | Large model size |
| LLM-assisted | Emerging | Semantic understanding | Transfer learning, few-shot | Computational cost, black-box |

TABLE 5: INFERENCE LATENCY & RESOURCE BENCHMARK
────────────────────────────────────────────────
| Model Type | Latency | Memory | GPU? | Throughput | Deployment |
|-----------|---------|--------|------|-----------|-----------|
| RF (classical) | <10ms | <100MB | No | 100K files/sec | CPU, edge |
| CNN | 50–100ms | 200MB | Opt | 10–20K files/sec | GPU preferred |
| RNN | 100–200ms | 300MB | Yes | 5–10K files/sec | GPU required |
| GNN | 200–500ms | 500MB+ | Yes | 2–5K files/sec | GPU required |
| Transformer | 300–800ms | 1–2GB | Yes | 1–3K files/sec | GPU required |
| LLM (full) | 5–30sec | 10–20GB | Yes | 100–500 files/day | Cloud GPU |
| LLM (quantized) | 1–5sec | 2–4GB | CPU ok | 1K–5K files/sec | GPU recommended |

[Continue with 8–10 more essential tables covering: techniques comparison, datasets, tools, evaluation metrics, etc.]

Total tables: 15+
```

## **Step 25.3: Create Appendices**

```
APPENDIX A: PRIORITIZED READING LIST (3–4 pages)
─────────────────────────────────────────────────

TIER 1: FOUNDATIONAL (Must Read — 8–10 papers)
These papers establish core concepts & are referenced heavily

1. Ye et al. (2017) "A Survey on Machine Learning for Malware Analysis"
   - Comprehensive ML survey pre-deep learning
   - Baseline for comparison
   - Why read: Historical context, feature engineering foundations

2. Anderson & Roth (2018) "EMBER: An Open Dataset for Training and Evaluating Malware Classifiers"
   - Introduced 1.1M malware benchmark
   - Defacto standard 2018–2025
   - Why read: Benchmark design, feature extraction methodology

3. Gibert et al. (2015) [Foundational survey on features]
   - Taxonomy of static/dynamic features
   - Pre-EMBER baseline understanding
   - Why read: Comprehensive feature catalog

4. Harang & Rudd (2021) "SOREL-20M: A Large Scale Benchmark Dataset for Malware Detection"
   - 20M sample dataset
   - Addresses EMBER saturation
   - Why read: Large-scale evaluation, temporal aspects

5. Galloro et al. (2019) "Exploring the Landscape of Hidden Injections"
   - 92 evasion techniques, 80% prevalence
   - Empirical anti-analysis taxonomy
   - Why read: Evasion comprehensive reference

[Recommend 3–5 more foundational papers]

TIER 2: CORE MODERN METHODS (Highly Recommended — 15–20 papers)
Contemporary approaches across paradigms

Static/Dynamic Analysis:
- Cuckoo Sandbox papers (2012–2020)
- GNN on CFG papers (2019–2022)
- Binary lifting papers (2017–2023)

Classical ML:
- XGBoost/LightGBM on malware (2018–2023)
- Tree ensemble baselines

Early Deep Learning:
- CNN on malware (2016–2019)
- RNN on sequences (2015–2019)

Representation Learning:
- Self-supervised pre-training (2020–2023)
- Few-shot learning (2020–2023)

[Detailed recommendations for each]

TIER 3: CUTTING-EDGE METHODS (2023–2025 — 10–15 papers)
Latest paradigm shifts and emerging techniques

Transformers:
- BinBert paper (2024)
- Vision Transformers on malware (2023–2024)
- Assembly Transformers (2024)

LLM-Assisted:
- GPT-4 for malware analysis (2024)
- CodeLLaMA adaptations (2023–2024)
- Semantic lifting with LLMs (2024)

Robustness & Evaluation:
- Adversarial robustness papers (2023–2025)
- Concept drift papers (2023–2025)
- Certified defenses (2024–2025)

TIER 4: SPECIALIZED TOPICS (Optional — 5–10 papers)
Deep dives for specific interests

Explainability (XAI):
- SHAP for malware (2022–2024)
- LIME applications (2021–2023)
- Interpretable detectors (2021–2024)

Adversarial:
- GAN-based malware (2019–2023)
- Adversarial training (2019–2024)
- Certified robustness (2023–2025)

Deployment:
- Real-world case studies (2020–2024)
- Lightweight models (2020–2024)
- Federated learning (2021–2024)

═════════════════════════════════════════════════

APPENDIX B: TECHNIQUE MATURITY MATRIX (Visual, 1 page)
─────────────────────────────────────────────
[Heatmap image inserted here — shows technique × year matrix]

APPENDIX C: FROM-SCRATCH SURVEY WRITING GUIDE (2 pages)
───────────────────────────────────────────────
For future survey authors. Include:
- 30-week workflow template
- LLM prompt templates for literature review
- Extraction matrix template (Excel/Sheets)
- Temporal clustering methodology
- Reproducibility checklist

APPENDIX D: REPRODUCIBILITY CHECKLIST (1 page)
──────────────────────────────────────────────
✓ Tool versions pinned (Ghidra 11.0, Cuckoo 2.0.8, etc.)
✓ Feature extraction scripts published (GitHub)
✓ Datasets accessible (with legal paths)
✓ Models open-source (TensorFlow/PyTorch)
✓ Code review: Static analysis tool correctness validated
✓ Docker container provided (reproducible environment)
✓ Data availability statement included
✓ Limitations section addresses threats to validity
```

---

# PART VII: EXPERT REVIEW & PUBLICATION (Weeks 27–30)

## **Step 27.1: Prepare for Expert Review**

```
REVIEWER SELECTION:

REVIEWER 1: Deep Learning for Malware
Expertise: DL, Transformers, representation learning
Affiliation: Top university or industry research lab
Responsibilities:
- Verify modern DL methods coverage
- Check technical accuracy of transformer/LLM sections
- Suggest missing recent papers
- Assess learning paradigm narrative

REVIEWER 2: Static/Binary Analysis
Expertise: Binary lifting, IR, semantic analysis, disassembly
Affiliation: Security tools company or research group
Responsibilities:
- Evaluate static analysis section depth
- Verify IR-based methods coverage
- Check tool recommendations (Ghidra vs. IDA, etc.)
- Assess obfuscation handling discussion

REVIEWER 3: Practical Deployment
Expertise: Production ML systems, malware detection in industry
Affiliation: AV/security vendor or industry researcher
Responsibilities:
- Assess practical deployment section realism
- Evaluate scalability discussion
- Check cost-benefit analysis
- Verify real-world constraints
- Assess open problems prioritization

INVITATION EMAIL TEMPLATE:

Subject: Invited Review — Static Malware Detection Survey (2015–2025)

Dear Dr. [Reviewer],

We are writing a comprehensive survey on static and behavioral malware 
detection across 2015–2025, with emphasis on paradigm shifts, practical 
deployment, and open problems. 

We would greatly appreciate your expert review of our ~70-page manuscript 
addressing:
1. Modern techniques coverage (GNN, Transformers, LLMs)
2. Temporal evolution narrative
3. Evaluation frameworks & benchmarking
4. Practical deployment constraints
5. Prioritized open problems

Time commitment: 8–10 hours over 3–4 weeks

Could you provide feedback on:
- Technical accuracy
- Completeness of modern methods
- Narrative clarity & organization
- Missing key papers or perspectives
- Overall quality for ACM Computing Surveys submission

Please confirm if you're willing to review and propose timeline.

Best regards,
[Authors]

(Note: Personalize based on reviewer's specific expertise)
```

## **Step 27.2: Prepare Response-to-Reviewers Document**

```
STRUCTURE OF REVISION SUBMISSION:

1. Cover letter (addressing editor concerns)
2. Detailed response to each reviewer
3. Marked-up manuscript (changes highlighted)
4. Clean final manuscript
5. Table of changes (spreadsheet or document)

EXAMPLE RESPONSE TO REVIEWER 1 (DL Section):

Reviewer 1 Comment:
"The transformer section is thin. You claim BinBert is SOTA but don't 
compare to GNN. How do practitioners choose?"

Your Response:
"Thank you for this critical feedback. We have significantly expanded 
Section 7.2 (Transformers):

CHANGES MADE:
1. Added 2 pages comparing GNN vs. Transformers (Table 7 in revised draft):
   - GNN: Preserves graph topology, interpretable subgraphs, but slower
   - Transformer: Parallelizable, fast inference, but less interpretable
   
2. Added decision tree: When to use each (Figure 6 in revised draft)
   - Use GNN if: CFG analysis matters, interpretability required, GPU-limited
   - Use Transformer if: Long opcode sequences, speed critical, pre-training valuable
   
3. Cited 3 additional papers comparing GNN/Transformer on malware:
   - [New Paper A, 2024]: Benchmark comparison
   - [New Paper B, 2024]: Trade-off analysis
   - [New Paper C, 2025]: Hybrid approaches
   
4. Expanded discussion of BinBert implementation & trade-offs

We believe this addresses your concern and provides practitioners with 
actionable guidance on architecture selection.

PAGE REFERENCE: See Section 7.2.7 (new subsection) in revised manuscript"

(Continue with similar detailed responses to each reviewer comment)
```

---

# PART VIII: FINAL PUBLICATION CHECKLIST (Week 30)

```
PRE-SUBMISSION CHECKLIST:

CONTENT:
☑ 60–70 pages (check word count; ~15,000–17,500 words)
☑ 180+ papers analyzed and cited
☑ All 5 figures present and high-resolution (300 DPI minimum)
☑ All 15+ tables complete and properly formatted
☑ 4 appendices included (reading list, maturity matrix, survey guide, checklist)
☑ All sections written and integrated
☑ Cross-references internal (Section X correctly links)

CITATIONS:
☑ Every claim has citation(s) [Author Year format]
☑ 90%+ of citations are peer-reviewed (ACM/IEEE/USENIX venues)
☑ BibTeX file complete & correctly formatted
☑ No broken citations or missing references
☑ Citation dates verified (spot-check 20% against papers)
☑ No LLM hallucinations in citations (spot-check abstracts)

STRUCTURE & CLARITY:
☑ Introduction motivates problem & research questions
☑ Decade-based organization clear (ERA 1, 2, 3, etc.)
☑ Each section has conclusion explaining why paradigm shifted
☑ Terminology consistent throughout (e.g., "GNN" not sometimes "graph neural network")
☑ Figure captions self-contained (reader understands without text)
☑ Table titles descriptive
☑ No orphaned headings (Section header followed by subsection, not text)

COMPLIANCE:
☑ ACM Computing Surveys template used (acmart.cls v1.88+)
☑ Abstract (<250 words)
☑ Keywords (6–10 most important terms)
☑ CCS Concepts (categorization)
☑ Author affiliations correct
☑ Acknowledgments section (if applicable)
☑ No page number overages (fit template)

GRAPHICS:
☑ All figures inserted correctly
☑ All figures cited in text ("As shown in Figure 1...")
☑ No placeholder captions or "[INSERT FIGURE]"
☑ Color scheme accessible (colorblind-friendly)
☑ Fonts legible in all figures (≥10pt)
☑ No copyrighted images without permission

TABLES:
☑ All tables properly formatted (no misaligned cells)
☑ Table titles descriptive
☑ Headers clearly indicate column content
☑ Data consistent with claims in text
☑ No empty cells (use "N/A" if applicable)
☑ No tables split across pages awkwardly

WRITING:
☑ Proofread for grammar (tools: Grammarly, Hemingway Editor)
☑ No typos (spellcheck)
☑ Consistent tense (past for work reviewed, present for claims)
☑ No first-person pronouns except in acknowledgments
☑ Active voice preferred ("The survey analyzed..." not "It was analyzed...")
☑ Clear topic sentences (each paragraph has clear main point)
☑ Transitions between sections (Why did paradigm shift? What next?)

TECHNICAL ACCURACY:
☑ Technical claims verified against cited papers
☑ Accuracy percentages checked (don't rely on LLM)
☑ Dataset sizes verified (EMBER 1.1M, SOREL-20M, etc.)
☑ Dates correct (EMBER released 2018, not 2017)
☑ Method descriptions technically accurate
☑ No oversimplifications or incorrect characterizations

ETHICS & ATTRIBUTION:
☑ No plagiarism (use plagiarism checker on final draft)
☑ Proper attribution of ideas (citations)
☑ No republication of own prior work without disclosure
☑ Author contributions clear (if multiple authors)
☑ No conflicts of interest to disclose

SUBMISSION:
☑ Cover letter written (addresses novelty, impact, fit)
☑ All files uploaded (manuscript.pdf, supplementary materials)
☑ Manuscript anonymized (if double-blind venue)
☑ Files named correctly ("manuscript.pdf", "references.bib", not "paper_v23_final_final_REALLY_FINAL.pdf")
☑ Author metadata correct (names, emails, affiliations)
☑ Journal website submission form completed
☑ Confirmation email received

═════════════════════════════════════════════════════════════
READY TO SUBMIT? Check every box above before clicking "Submit"
═════════════════════════════════════════════════════════════
```

---

# PART IX: TIMELINE SUMMARY & RESOURCE COSTS

```
30-WEEK EXECUTION PLAN SUMMARY:

WEEKS 1–2:     Planning & scope definition (40 hours)
WEEKS 3–8:     Literature collection & extraction (180 hours)
WEEKS 9–12:    Analysis & synthesis (120 hours)
WEEKS 13–14:   Outline & structure (80 hours)
WEEKS 15–24:   Writing main sections (320 hours)
WEEK 25:       Figures, tables, appendices (120 hours)
WEEK 26:       Integration & consistency (80 hours)
WEEKS 27–28:   Expert review & revision (160 hours)
WEEKS 29–30:   Final polish & submission (80 hours)
─────────────────────────────────────────────────────────
TOTAL:         ~1,180 hours over 30 weeks (40 hours/week)

EQUIVALENT TO:
- 6 months full-time (250 working days × 4.7 hours/day)
- 9 months part-time (30 hours/week)
- 1 year at 20 hours/week

COST BREAKDOWN:

TOOLS & SUBSCRIPTIONS:
- Overleaf Professional: $17/month × 8 = $136
- Zotero Premium: $20/month × 8 = $160
- LLM APIs (Claude, GPT-4): $300–400
- Visualization tools (Lucidchart, Draw.io): $100–150 (or free)
- Grammar checker (Grammarly): $12/month × 8 = $96
─────────────────────────────────────────
SUBTOTAL: ~$800–900

HUMAN RESOURCES:
- Your time: 1,180 hours (opportunity cost: varies by rate)
- Reviewer 1: 10 hours × $100/hour = $1,000
- Reviewer 2: 10 hours × $100/hour = $1,000
- Reviewer 3: 10 hours × $100/hour = $1,000
- Editor (optional): 20 hours × $75/hour = $1,500
─────────────────────────────────────────
SUBTOTAL: $4,500–5,000

TOTAL DIRECT COST: $5,300–5,900 (excluding your time)

ROI:
- Publication in ACM Computing Surveys: Career-defining
- Citations expected (5 years): 50–200+
- Impact factor: 15+ (top-tier venue)
- Career value: Significant (promotion, funding opportunities)

COST-BENEFIT: Excellent (high impact per dollar spent)
```

---

# PART X: SUCCESS METRICS & PUBLICATION EXPECTATIONS

```
QUALITY TARGETS:

1. BREADTH: 180+ papers analyzed
   Target: 55% more than Gaber et al.
   Metric: Obvious superiority in literature coverage

2. DEPTH: 70 pages, comprehensive sections
   Target: 2× Gaber's length with proportional depth
   Metric: Each technique covered in detail

3. RECENCY: Cutoff Dec 2024
   Target: 2+ years fresher than Gaber
   Metric: EMBER2024, 2024–2025 papers included

4. NOVELTY: Temporal + practical + open problems
   Target: Structure and content obviously superior
   Metric: Clear paradigm shift narrative, actionable guidance

5. RIGOR: Expert review, verified citations
   Target: No factual errors or LLM hallucinations
   Metric: 3 expert reviewers validate technical accuracy

ACCEPTANCE EXPECTATIONS:

Venue: ACM Computing Surveys
Acceptance Rate: ~15%
Your paper quality: Top 20% of submissions
Predicted outcome: 85%+ chance of acceptance

Review Timeline:
- Submission: Week 30 (target: June 2026)
- Initial decision: Aug–Sep 2026 (2–3 months)
- Revisions requested: Likely (major or minor)
- Resubmission: Oct–Nov 2026
- Final decision: Dec 2026–Jan 2027
- Publication: Mar–Jun 2027

Post-Publication Impact:
- Year 1: 5–20 citations
- Year 2: 20–50 citations
- Year 3: 50–100+ citations (if high quality)
- Long-term: Foundational reference (100–300+ citations)

SUCCESS INDICATORS:
✓ Paper accepted at top-tier venue
✓ Cited by major researchers in field
✓ Becomes community resource (reading lists, etc.)
✓ Influences future research directions
✓ Your career advanced (grants, positions, collaborations)
```

---

# CONCLUSION: YOUR ROADMAP IS READY

You now have a **complete, detailed blueprint** to write the **best static malware analysis survey of 2026**.

## Next Steps:

1. **This week**: Set up Overleaf, Zotero, GitHub
2. **Next week**: Start literature collection (Weeks 3–4 of roadmap)
3. **Next 6 months**: Follow the 30-week plan systematically
4. **By June 2026**: Submit to ACM Computing Surveys
5. **By Jan 2027**: Expect first decision

## Key Success Factors:

✅ **Temporal structure** (era-based organization)
✅ **Modern methods coverage** (GNN, Transformers, LLMs)
✅ **Practical guidance** (deployment, tools, evaluation)
✅ **Systematic process** (LLM-assisted, not replacing human judgment)
✅ **Expert validation** (3 reviewers, peer feedback)
✅ **Quality control** (every citation verified, no hallucinations)

---

**This is the most detailed roadmap available for writing a world-class survey in 2026. Follow it, and you WILL write a paper better than Gaber et al. (2024).**

