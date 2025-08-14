# 📄 Paper No Yapper

**Version:** v0.0.0

Paper No Yapper is a lightweight research paper domain/methodology/tool extractor.  
Give it a paragraph (from your paper, abstract, or notes), and it will automatically detect relevant domains, techniques, and tools — even without a large language model.  
It uses a semantic knowledge base + unsupervised text extraction for accurate results.

---

## ✨ Features (v0.0.0)
- 📥 Reads research paper abstracts or paragraphs from a text file.
- 🧠 Matches sentences against a custom **Knowledge Base** (`onto.csv`) of tools, domains, and techniques.
- 🔍 Extracts additional terms **unsupervised** when they are not in the KB.
- ⚡ Uses **Sentence Transformers** (`all-MiniLM-L6-v2`) for semantic matching.
- 📊 Outputs top matches grouped by type (Tool / Domain / Technique).
- 📄 Supports CSV, Excel, and PDF outputs for results.

---

## 📂 Input Options
You can prepare your input in two ways:

1. **Export CSV from Scopus**  
   - From your Scopus search results, export as CSV and use the abstracts/paragraphs from there.

2. **Use my [Paper Grabber Tool](https://github.com/AniruthKarthik/paper-grabber)**  
   - Bookmarklet to fetch papers, abstracts, and metadata for analysis.

---

## 🚀 Requirements
Install dependencies:

```bash
pip install pandas sentence-transformers openpyxl python-docx reportlab
````

Optional (for PDF parsing):

```bash
pip install PyPDF2
```

---

## 📦 Files

* **para.txt** → Your paragraph or abstract to analyze.
* **onto.csv** → Knowledge base in the format:

  ```
  name,type,description
  IDA Pro,tool,IDA Pro is a static analysis toolkit...
  ```
* **script2.py** → Main analysis script.

---

## ▶ Usage

```bash
python script2.py
```

* Reads `para.txt`
* Loads `onto.csv`
* Outputs the top matching **Tools**, **Domains**, and **Techniques**.

---

## 📜 Version History

* **[v0.0.0](https://github.com/AniruthKarthik/paper-no-yapper/releases/tag/v0.0.0)** – Initial config with KB matching & unsupervised extraction.

---

## 🔗 Related Repos

* **[Paper Grabber](https://github.com/AniruthKarthik/project-sandbox/paper-grabber)** – Bookmarklet to save your favourite papers into an excel sheet.

---


