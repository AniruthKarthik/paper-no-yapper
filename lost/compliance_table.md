# MANDATORY SUMMARY TABLE
## Decade-Aware Compliance Assessment Matrix

| **Aspect** | **Covered in Current Paper** | **Adequate for 2025** | **Issues Identified** | **Recommended Action** |
|---|:---:|:---:|---|---|
| **1. TEMPORAL FOCUS (2015–2025)** | ✔ (80%+) | ✘ | Cutoff ~2022; post-2022 paradigm shifts severely underrepresented (transformers, LLMs, foundation models) | **Add Section 6**: "Transformer & Foundation Model Era (2023–2025)"; extend temporal reach to Dec 2024 minimum |
| **2. STATIC ANALYSIS COVERAGE** | ✔ | ✔ | PE headers, n-grams, opcodes comprehensively covered; obfuscation limitations noted | **Keep existing**; add note on IR-based semantics as modern alternative to syntactic analysis |
| **3. DYNAMIC ANALYSIS COVERAGE** | ✔ | ✔ | Cuckoo, DBI, anti-evasion techniques well-detailed (Tables 2, 3, 6) | **Keep existing**; add DRAKVUF, lightweight containerization (SecBox); update anti-evasion taxonomy post-2023 |
| **4. GRAPH NEURAL NETWORKS (GNN)** | ✘ (minimal) | ✘ | Only CFGExplainer [4] mentioned in passing; GNN paradigm for malware analysis ignored | **Create Section 5 subsection** "Graph-Based Structural Analysis (2021–2023)"; cite [web:4, 7, 10, 35]; discuss GCN/GAT/GIN on CFG |
| **5. BINARY LIFTING & IR** | ✘ | ✘ | Completely absent; critical for semantic robustness to obfuscation | **Create Section 5 subsection** "Binary Lifting & Intermediate Representations (2017–2025)"; cite [web:3, 14, 17, 20]; explain Valgrin/BAP/LLVM IR approach |
| **6. TRANSFORMER ARCHITECTURES** | ✘ | ✘ | No dedicated section; only implicit mention in LLM context; transformer paradigm not centered despite SOTA status | **Create Section 6 subsection** "Transformer-Based Malware Analysis (2023–2025)"; cite [web:5, 8, 11]; cover BinBert, Vision Transformers, assembly transformers |
| **7. LLM-ASSISTED ANALYSIS** | ✘ (4 papers only) | ✘ | Treated as sidebar to generative AI; actually represents paradigm shift from feature engineering to semantic understanding | **Create major Section 6** "LLM-Assisted & Semantic Analysis (2023–2025)"; 5+ pages; cite [web:24, 27, 28, 40]; discuss CodeLLaMA, GPT-4, semantic lifting |
| **8. FEW-SHOT & META-LEARNING** | ✘ | ✘ | Completely absent; critical for novel/zero-day malware detection | **Add to Section 6**; cite [web:23, 29]; discuss SIMPLE, graph-based meta-learning |
| **9. TEST-TIME ADAPTATION** | ✘ | ✘ | Not mentioned; important for handling concept drift | **Add to Section 6**; cite [web:35, 38]; discuss MADCAT, warm-start learning, pseudo-labeling |
| **10. ADVERSARIAL ROBUSTNESS** | ✘ (evasion only) | ✘ | Anti-analysis techniques covered extensively; **robustness of detectors against adversarial malware is minimal** | **Expand into new Section 7.6** "Adversarial Robustness & Robustness Testing"; cite [web:13, 19, 35]; cover GAN evasion, certified defenses |
| **11. EXPLAINABILITY (XAI)** | ✘ | ✘ | SHAP, LIME, Grad-CAM not mentioned; critical for trust & debugging | **Add to Section 7** "Interpretability & Explainability (2019–2025)"; cite [web:22, 25, 31]; discuss feature importance, decision explanation |
| **12. CONCEPT DRIFT & TEMPORAL** | ✘ (mentioned challenge) | ✘ | Acknowledged as problem; **no evaluation framework provided**; time-stratified train/test absent | **Add Section 7.2** "Temporal Evaluation & Concept Drift"; cite [web:32, 35, 38]; recommend time-split evaluation methodology |
| **13. CROSS-DATASET GENERALIZATION** | ✘ | ✘ | Generalization failures noted; **no systematic evaluation protocol**; train on EMBER, test on VirusShare not standard | **Add Section 7.3** "Cross-Domain & Cross-Dataset Robustness"; cite [web:33]; protocol: train on A, test on B, C, D |
| **14. DATASET: EMBER** | ✔ | ✘ | Acknowledged; **saturation issue (0.999 AUC baseline) NOT discussed** | Clarify: EMBER is saturated benchmark; note that 98–99% accuracies may reflect dataset ease, not detector capability |
| **15. DATASET: SOREL-20M** | ✔ | ✔ | Good coverage; 20M scale noted | Keep; **emphasize time-split recommendations** for temporal evaluation |
| **16. DATASET: BODMAS** | ✔ | ✔ | Timestamp-aware; good for temporal analysis | Keep; note advantages over EMBER for temporal studies |
| **17. DATASET: EMBER2024** | ✘ | ✘ | **COMPLETELY MISSING** (released June 2025); 3.2M samples, 7 tasks, evasion-focused challenge set | **Critical gap**: Add EMBER2024 to Section 4.3; major new section header; emphasize evasion-focused evaluation |
| **18. EVAL: TEMPORAL / DRIFT** | ✘ | ✘ | Concept drift acknowledged; **no temporal evaluation framework** provided; no discussion of how models degrade over time | **Add Section 7.2**: Define concept drift formally; recommend time-stratified splits; cite temporal drift empirical studies |
| **19. EVAL: CROSS-DATASET** | ✘ | ✘ | Generalization failures discussed; **no standard cross-dataset protocol**; most papers train/test on same dataset | **Add Section 7.3**: Propose protocol (train EMBER, test on SOREL/VirusShare/BODMAS); measure domain gap |
| **20. EVAL: ADVERSARIAL ROBUSTNESS** | ✘ | ✘ | No systematic evaluation of detector robustness to adversarial/evasive samples | **Add Section 7.6**: Evaluate against GAN-based evasion, certified defenses; cite adversarial robustness literature |
| **21. LEARNING PARADIGM: Classical ML** | ✔ | ✔ (baseline) | Covered adequately | Reposition: "Baseline Approach (Lightweight, Interpretable)"; clarify when RF/SVM preferable to DL |
| **22. LEARNING PARADIGM: Early DL** | ✔ | ✘ (over-emphasis) | CNN/RNN vanilla treated equal to modern architectures; not clearly marked as transitional/superseded | **Demote to subsection** "Early Deep Learning Attempts (2015–2018): Foundations & Limitations"; cite generalization failures |
| **23. LEARNING PARADIGM: Hybrid Static+Dynamic** | ✔ | ✘ | Good examples (AI-Hydra, CSPE-R); **not framed as pragmatic engineering choice** | **Reframe** as "Hybrid Fusion Approaches (2018–2021): Engineering Trade-Offs"; discuss accuracy vs. latency vs. cost |
| **24. LEARNING PARADIGM: Representation Learning** | ✘ | ✘ | Self-supervised pre-training, contrastive learning, embeddings absent | **New subsection** in Section 5: "Representation Learning & Self-Supervised Pre-Training (2020–2023)"; discuss why paradigm shifted |
| **25. LEARNING PARADIGM: Graph-Based** | ✘ | ✘ | GNN paradigm for malware analysis not positioned as SOTA for structural data | **New subsection** in Section 5: "Graph Neural Networks on Control/Data Flow (2019–2023)"; discuss GCN, GAT, GIN |
| **26. LEARNING PARADIGM: Transformers** | ✘ | ✘ | Not dedicated subsection; transformer SOTA status not established | **Major subsection** in Section 6: "Transformers for Malware Analysis (2023–2025)"; cover BinBert, Vision Transformers, assembly transformers |
| **27. LEARNING PARADIGM: Foundation Models** | ✘ | ✘ | LLM mention buried; paradigm-shifting importance underappreciated | **Major subsection** in Section 6: "Foundation Models & Large Language Models (2023–2025)"; 5+ pages; discuss transfer learning, fine-tuning |
| **28. PARADIGM CLARITY TABLE** | ✘ | ✘ | Figure 4 presents ML vs. DL as binary; conflates distinct paradigms; no "what should readers use in 2025?" guidance | **Replace Figure 4** with revised "Learning Paradigms: Recommendation for 2025" table; stratify: legacy | baseline | transitional | SOTA |
| **29. OBFUSCATION: Packing & Polymorphism** | ✔ | ✔ | Baseline obfuscation covered | Keep; remains foundational |
| **30. OBFUSCATION: Anti-Analysis (Debugger/VM/DBI)** | ✔ | ✔ | Comprehensive taxonomy (Tables 2–3, 6) with empirical prevalence data | Keep; mature coverage |
| **31. OBFUSCATION: Semantic & CFG Flattening** | ✘ | ✘ | Control-flow flattening, opaque predicates not discussed; emerging technique not covered | **Add Section 5**: "Semantic Obfuscation & Control-Flow Transformations (2021–2025)"; discuss GNN robustness to flattening |
| **32. OBFUSCATION: AI-Powered Evasion (GANs)** | ✔ (brief) | ✘ | DeepLocker, GUI attacks mentioned; **GAN-based feature evasion & adversarial malware sparse** | **Expand Section 7.6**: Comprehensive GAN taxonomy; cite [web:13] for 2025 overview of adversarial evasion |
| **33. OBFUSCATION: Evolution Narrative** | ✘ | ✘ | Obfuscation techniques listed; **no "why did this evolve in response to detection"** narrative | **Add timeline figure** showing obfuscation evolution decade-wise; explain pressure from improved detection |
| **34. PRACTICALITY: Scalability** | ✔ (implicit) | ✘ | Mentions DBI slow; no formal analysis of real-world volumes (450K samples/day) | **Add Section 7.1** "Scalability & Throughput Requirements"; discuss 450K/day reality; computational bottlenecks |
| **35. PRACTICALITY: Latency & Resource Cost** | ✔ (mentioned) | ✘ | AI-Hydra latency (60.9 sec) noted; **no systematic comparison table** across models/methods | **Add to Section 7.1**: Table with model, inference latency, memory, GPU requirement, FPS achieved |
| **36. PRACTICALITY: Tiered Detection Pipelines** | ✘ | ✘ | Hybrid static+dynamic discussed; **not framed as production pipeline** (fast → deep → manual) | **Add Section 7.1**: Design pattern with cost analysis; recommend tiered approach |
| **37. PRACTICALITY: Tooling & Infrastructure** | ✔ (named) | ✘ | Tools listed (IDA, Cuckoo, Pin); **no discussion of reproducibility, versioning, containerization** | **Add Section 11** "Tools, Infrastructure & Reproducibility"; recommend Ghidra (free, open-source); discuss Docker; version pinning |
| **38. PRACTICALITY: Anti-Analysis Arms Race** | ✔ | ✘ | Extensively covered; **doesn't question viability of static analysis** given 60–80% obfuscation prevalence | **Add commentary** in Section 7: Is static analysis becoming obsolete? Discuss IR-based & semantic alternatives |
| **39. PRACTICALITY: Cascading/Staged Pipelines** | ✘ | ✘ | Hybrid mentioned; not framed as realistic production cascade | **Add Section 7.1**: Tier 1 (static, <1s) → Tier 2 (dynamic, 30–60s if uncertain) → Tier 3 (manual, critical) |
| **40. REPRODUCIBILITY: Code & Data Availability** | ✘ | ✘ | Not discussed; papers analyzed without noting open-source availability | **Add Section 11**: Recommend papers release code (GitHub), datasets (with legal path), Dockerfiles; discuss reproducibility crisis |
| **41. TAXONOMY: Temporal Organization** | ✘ | ✘ | Current structure (functional: analysis → features → methods) lacks temporal progression | **Restructure entire paper** using decade-layered approach (proposed Section 8); show evolution not just snapshot |
| **42. TAXONOMY: Visual Figure** | ✘ | ✘ | No evolution timeline figure; no maturity matrix; Figure 4 conflates paradigms | **Create Figure 1**: Decade-wise technique evolution; Figure 2: Maturity matrix (technique × era × maturity level) |
| **43. TAXONOMY: Clarity for Practitioners** | ✘ | ✘ | Paper doesn't clearly say "use this method in 2025" vs. "this is historical" | **Add Table**: "Learning Paradigm Recommendations for 2025"; map technique → legacy | baseline | SOTA |
| **44. STRUCTURE: Logical Flow** | ✔ | ✔ | Current organization (analysis → features → data → methods) makes sense | Keep organization but add temporal layering within sections |
| **45. STRUCTURE: From-Scratch Guide** | ✘ | ✘ | No guidance for grad students writing similar survey; methodology section brief | **Add Section 9**: "Systematic Review Strategy (Best Method for 50–100 Papers)"; discuss LLM-assisted workflow |
| **46. WRITING QUALITY: Clarity** | ✔ | ✔ | Paper well-written, figures clear, tables comprehensive | Keep; minor proofreading pass |
| **47. WRITING QUALITY: Completeness** | ✔ | ✘ | Comprehensive within scope; **major topic gaps** (GNN, transformers, LLMs, robustness, drift) | Address via new sections |
| **48. CITATION: Breadth (2015–2025)** | ✔ | ✘ | 77 papers included; range spans decade; **distribution skewed toward 2018–2022** (>60% of papers) | Rebalance: aim for 30% pre-2020, 40% 2020–2022, **30% 2023–2025** |
| **49. CITATION: Pre-2015 Papers** | ✔ (foundational) | ✘ (over-cited) | Foundational papers (e.g., Ye 2017, Gibert 2015) cited throughout main narrative | Demote to background/footnotes; emphasize pre-2015 as historical not current |
| **50. CITATION: Post-2022 Papers** | ✘ | ✘ | Minimal post-2022 coverage; LLM papers sparse; paradigm-shift papers missing | Add 15–20 post-2022 papers; cite [web:3, 5, 8, 15, 24, 27, 28, 31, 32, 35, 38, 40, 41] |
| **51. CITATION: High-Impact Omissions** | ✘ | ✘ | BinBert, CFGExplainer, Security LLM, EMBER2024, SoK papers, concept drift papers absent | Systematic addition of [web:3, 4, 8, 15, 24, 27, 28, 31, 32, 35, 38, 40, 41] |
| **52. READING LIST** | ✘ | ✘ | No prioritized guide for readers; no stratification by era/difficulty | **Add Appendix**: Stratified reading list (foundations → core → modern → emerging); ~30–40 papers |
| **53. OPEN PROBLEMS: Identified** | ✔ | ✘ | Challenges mentioned; **not systematically listed or prioritized** | **Add Section 9**: Explicit open problems list; prioritize by impact + feasibility |
| **54. FUTURE DIRECTIONS: Established vs. Speculative** | ✘ | ✘ | Conclusion mentions future work; no distinction between proven trends and speculative | **Add Section 9.2**: Separate "Established (high confidence)" from "Emerging (medium)" from "Speculative (low)" |
| **55. FUTURE DIRECTIONS: Actionable** | ✘ | ✘ | No concrete research roadmap; grad students can't use to decide next project | **Add Section 9.3**: Prioritized recommendations (high impact + feasible first) |
| **56. OVERALL NARRATIVE** | ✔ | ✘ | Coherent within 2018–2022 scope; **lacks decade-aware progression** showing why techniques evolved | Restructure to show: Legacy (pre-2015) → Classical (2015–2018) → DL Emergence (2018–2021) → Structural (2021–2023) → Transformers (2023–2025) |
| **57. CONTRIBUTION POSITIONING** | ✔ | ✘ | Paper claims to cover "state-of-the-art methods"; **actually covers 2015–2022 SOTA; 2023–2025 paradigm shifts missed** | Retitle: "Static & Behavioral Malware Analysis with Machine Learning (2015–2025): Evolution from Feature Engineering to Foundation Models" |
| **58. SUBMISSION READINESS** | ✔ | ✘ | Paper structure coherent, publishable in 2023; **insufficient for 2025 standards** without major revision | **Estimate**: 4–6 weeks for moderate revision; 8–12 weeks for comprehensive rewrite |

---

## SCORING SUMMARY

| **Category** | **Current Coverage** | **2025 Adequacy** | **Severity** | **Effort to Fix** |
|---|---|---|---|---|
| **Temporal Scope** | 60% | 40% | **CRITICAL** | High (restructure) |
| **Technique Breadth** | 70% | 30% | **CRITICAL** | High (major additions) |
| **Dataset Discussion** | 80% | 50% | **HIGH** | Medium (update table) |
| **Learning Paradigm Clarity** | 50% | 20% | **CRITICAL** | High (reframe) |
| **Practical Deployment** | 40% | 30% | **HIGH** | Medium (new section) |
| **Robustness & Evaluation** | 30% | 20% | **HIGH** | High (new sections) |
| **Literature Quality** | 75% | 45% | **CRITICAL** | Medium (add citations) |
| **Taxonomy & Structure** | 65% | 35% | **HIGH** | High (restructure) |
| **Writing Quality** | 85% | 80% | **LOW** | Low (proofreading) |
| **Reproducibility** | 20% | 10% | **MEDIUM** | Low (add section) |

---

## OVERALL RECOMMENDATION

### For Current Paper Revision:
- **Quick (1 month)**: Add 2–3 pages on transformers/LLMs; update datasets to EMBER2024; clarify temporal scope in title
- **Moderate (4–6 weeks)**: Add sections on GNN, binary lifting, robustness, practical deployment; revise learning paradigm section
- **Comprehensive (8–12 weeks)**: Full restructure into decade-layered organization; complete paradigm narrative; 10+ new pages

### For From-Scratch Rewrite:
- **Timeline**: 6–9 months (30-week structured workflow with LLM assistance)
- **LLM Role**: ~40% (summarization, extraction, scaffolding); always human-validated
- **Output Quality**: Discipline-shaping survey with clear decade-first perspective + actionable research roadmap

### Submission Readiness:
- **Current**: Publishable as-is in 2023; insufficient for 2025 venue standards
- **After revision**: Competitive for top venues (IEEE S&P, USENIX Security, ACM Computing Surveys) if major content added
- **Key**: EMBER2024 addition is non-negotiable for 2025 credibility

---

**This table provides quick-reference assessment for authors, reviewers, and survey developers.**
**For detailed reasoning, see main audit report (1,500+ lines of analysis).**
