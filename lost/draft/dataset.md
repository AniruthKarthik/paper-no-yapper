# Malware Datasets for Static and Hybrid Detection

This section surveys the major publicly documented malware datasets that underpin modern static and hybrid detection research, with a focus on exact sizes, label composition, and platform coverage.[web:61][web:71][web:83][web:62][web:73][web:68][web:82][web:18]

---

## 1. EMBER 2018

EMBER (Endgame Malware BEnchmark for Research) is a labeled benchmark for static detection of Windows PE malware.[web:61][web:75]

- **Scope and size**  
  - Total samples: **1.1M Windows PE files**.[web:61][web:80]  
  - Training set: **900 000** (300 000 malicious, 300 000 benign, 300 000 unlabeled).[web:61][web:80]  
  - Test set: **200 000** (100 000 malicious, 100 000 benign).[web:61]  

- **Label and class balance**  
  - Labeled portion (800 000 files) is **50% malware, 50% benign**.[web:61][web:80]  
  - Unlabeled 300 000 samples are drawn from the same collection period for semi‑supervised experiments.[web:61]  

- **Features and OS coverage**  
  - Platform: **Windows PE only** (Win32/Win64 executables and DLLs).[web:61][web:79]  
  - Each sample is represented by a **2358‑dimensional static feature vector** capturing PE header metadata, section statistics, imported APIs, byte histograms, entropy histograms, and string statistics.[web:61][web:75]  
  - Only features and hashes are released, not raw binaries, due to legal and safety concerns.[web:61][web:79]  

- **Splits and design**  
  - Files come from a commercial telemetry stream and are split into train and test based on collection time to simulate short‑range temporal drift.[web:61][web:80]  
  - The authors provide a baseline LightGBM model that outperforms MalConv without hyper‑parameter tuning.[web:61]  

- **Strengths and limitations**  
  - Strengths: Large, balanced, and easy to use for static PE classification.[web:61][web:80]  
  - Limitations: Windows‑only, purely static, and now **saturated**—simple gradient‑boosted trees can reach near‑ceiling ROC AUC, so small gains may not indicate real progress.[web:62]

---

## 2. SOREL‑20M

SOREL‑20M (Sophos/ReversingLabs‑20 Million) is a large‑scale PE dataset that combines static features, metadata, and disarmed binaries.[web:71][web:12]

- **Scope and size**  
  - **20 M PE files** with pre‑extracted features and metadata (malicious and benign).[web:71][web:12]  
  - Approximately **10 M “disarmed” malware binaries** with header flags zeroed to prevent direct execution.[web:71][web:12]  

- **Labels and tags**  
  - High‑quality malware/benign labels derived from multiple sources and internal pipelines.[web:71]  
  - Metadata includes **vendor detection counts at collection time** and additional **behavior‑like tags** (e.g., ransomware, downloader) that can serve as auxiliary targets.[web:71][web:12]  

- **Features and OS coverage**  
  - Platform: **Windows PE** (32‑bit and 64‑bit) only.[web:71]  
  - Distributed artifacts: static feature vectors, metadata, and code to load and process them; raw binaries are available only for the disarmed malicious subset.[web:71][web:12]  

- **Temporal information**  
  - Each sample carries collection timestamps and vendor detections, enabling **time‑aware training/evaluation** and analyses of labeling delay.[web:71]  

- **Strengths and limitations**  
  - Strengths: Orders of magnitude larger than EMBER, realistic vendor labels, and rich metadata for drift and robustness studies.[web:71][web:12]  
  - Limitations: Dataset is heavy (several terabytes if all artifacts are used) and remains Windows‑PE‑centric.[web:76][web:71]

---

## 3. BODMAS

BODMAS (Blue Hexagon Open Dataset for Malware Analysis) targets temporal and family‑aware PE analysis in a moderate‑size corpus.[web:77][web:83][web:86]

- **Scope and size**  
  - **57 293 malware samples** and **77 142 benign samples**.[web:77][web:83]  
  - Total: **134 435 Windows PE files** collected between **August 2019 and September 2020**.[web:83][web:86]  

- **Label composition**  
  - Binary labels: malware vs benign; benign samples are indicated by an empty family field in the metadata.[web:83]  
  - Malware family information is provided for **581 distinct families**, with a long‑tailed distribution and several families having ≥1000 instances.[web:83]  

- **Features and OS coverage**  
  - Platform: **Windows PE** only.[web:77][web:83]  
  - Released artifacts include **disarmed binaries**, feature vectors, and a metadata CSV with SHA‑256, first‑seen time, and family label.[web:77][web:83]  

- **Temporal properties**  
  - Each sample has a “first seen” timestamp, allowing **explicit temporal splits** (e.g., train on early 2019–2020, test on later months) to study concept drift.[web:83][web:86]  

- **Strengths and limitations**  
  - Strengths: Well‑curated family labels and explicit timestamps designed for temporal and few‑shot/family‑aware research.[web:83][web:86]  
  - Limitations: Considerably smaller than EMBER/SOREL and still restricted to Windows PE.[web:77][web:83]

---

## 4. EMBER2024

EMBER2024 is a next‑generation benchmark created with the original EMBER authors to provide a **multi‑platform, multi‑task, and robustness‑oriented** dataset.[web:18][web:67][web:84][web:62]

- **Scope and size**  
  - **>3.2 M files** collected from **September 2023 to December 2024** (64 weeks).[web:84][web:82]  
  - Supported file formats: **six types** – **Win32**, **Win64**, **.NET**, **APK**, **ELF**, and **PDF**.[web:84][web:67]  

- **Splits and temporal design**  
  - Training set: **2 626 000 files** from weeks 1–52.[web:62][web:84]  
  - Test set: **606 000 files** from weeks 53–64.[web:62][web:84]  
  - **Challenge set**: **6315 malicious files** that were initially undetected by any AV engine (0 detections across ≈70 products within 24 h) but later labeled malicious after at least 5 independent AV positives.[web:62][web:67][web:84]  

- **Tasks and labels**  
  - Supports **seven classification tasks**:[web:62][web:18]  
    - Binary malware vs benign.  
    - Malware family classification (≈6787 families; 2538 with ≥10 instances).[web:62]  
    - Behavior tagging (118 behavioral tags).  
    - File‑property tagging (≈30 tags).  
    - Packer recognition (52 packer types).  
    - Exploited vulnerability identification (293 CVE tags).  
    - Threat‑group attribution (43 APT groups).  
  - Labels are derived from two VirusTotal snapshots: within 24 h and after ≥90 days, to reduce early mislabeling.[web:84][web:62]  

- **Features and OS coverage**  
  - Multi‑platform: Windows (PE/.NET), Android (APK), Linux (ELF), and document malware (PDF).[web:84][web:67]  
  - Provides hashes (MD5, SHA‑1, SHA‑256, TLSH), metadata, and **EMBER feature version 3** vectors up to **2568 dimensions**, with expanded PE and general features and partial support for non‑PE formats.[web:84][web:62]  

- **Performance reference**  
  - A baseline LightGBM classifier trained on weeks 1–52 and evaluated on weeks 53–64 achieves **ROC AUC ≈0.9949** and **TPR 94.48% at 1% FPR** on the standard test set, but performance drops dramatically on the challenge set (overall PR AUC ≈0.57, and as low as 0.24 on APK).[web:84][web:62]  

- **Strengths and limitations**  
  - Strengths: Recent data, multi‑format coverage, multi‑task labels, and an explicit **evasive malware challenge set** for robustness evaluation.[web:84][web:62][web:67]  
  - Limitations: Raw files are not directly distributed; access requires VirusTotal API plus the provided retrieval scripts, and the dataset is more complex to use correctly than earlier benchmarks.[web:84][web:18]

---

## 5. MalwareBazaar

MalwareBazaar is a continuous malware sharing project maintained by abuse.ch, aimed at distributing fresh samples to defenders and researchers.[web:73][web:68]

- **Scope and size**  
  - As of December 2025, the statistics page reports **millions of samples** submitted since project inception, with daily and cumulative counts broken down by family, file type, and signature status.[web:73]  
  - The exact live count changes daily; the service routinely lists **hundreds of thousands of available hashes** with downloadable samples for authenticated users.[web:73]  

- **Labels and metadata**  
  - Each entry includes at minimum a SHA‑256 hash, file type (e.g., PE, ELF, APK), and one or more malware family names derived from AV labels or contributor tags.[web:73][web:68]  
  - Additional metadata may include YARA rule matches, signatures, tags (e.g., ransomware, banker), and first‑seen dates.[web:73]  

- **Formats and OS coverage**  
  - Multi‑format: includes Windows PE, Android APK/DEX, Linux ELF, scripts, and document formats; exact percentages vary over time but Windows PE and Android APKs make up a large fraction of recent feeds.[web:73][web:68]  

- **Usage patterns**  
  - Often used to construct **up‑to‑date test sets** or family‑specific corpora by filtering on tags or family names.[web:73][web:68]  
  - In RawMal‑TF, monthly archives exported from MalwareBazaar are used as a family‑labeled source; filenames already encode family labels such as “HEUR‑Trojan‑Banker.Win32.Qbot,” enabling direct organization into type and family.[web:68]  

- **Strengths and limitations**  
  - Strengths: Fresh, diverse, and multi‑platform corpus with structured statistics and API access.[web:73][web:68]  
  - Limitations: No canonical train/test split, incomplete or noisy labels in some cases, and access constraints; users must design their own sampling, balancing, and labeling pipelines.[web:73][web:68]

---

## 6. VirusShare

VirusShare is one of the largest publicly accessible malware repositories, distributing malware in numbered archive chunks mainly via torrents.[web:68][web:82]

- **Scope and size**  
  - As of a recent large‑scale labeling effort, at least **487 numbered archives** were available, totaling approximately **16 791 GB of malware binaries**.[web:68]  
  - Earlier work on labeling VirusShare reported **over 27 M malware samples** labeled using the VirusTotal API, collected across multiple chunks, and indexed for efficient access.[web:82]  

- **Labels and metadata**  
  - VirusShare itself provides **no ground‑truth labels** beyond presence in the corpus; labeling typically comes from querying VirusTotal and aggregating AV detections.[web:68][web:82]  
  - A 2016 project used 30 people over six months to label over 27 M samples via VirusTotal, highlighting both the scale and the rate‑limit challenges.[web:82]  
  - Subsequent tools such as ClarAVy have processed **tens of millions of VirusTotal reports** (≈39.7 M reports covering ≈1.1 B AV detections) to infer consistent family labels from heterogeneous vendor naming schemes.[web:85]  

- **Formats and OS coverage**  
  - Dominated by Windows PE binaries, but also includes other formats (scripts, documents, non‑Windows binaries) depending on the specific chunks.[web:68][web:82]  

- **Strengths and limitations**  
  - Strengths: Massive scale and long historical span make VirusShare ideal as a **raw source** for custom datasets and for studying long‑tail families.[web:68][web:82]  
  - Limitations: No first‑class labels, inconsistent AV naming, substantial noise in ground truth, and heavy storage/processing requirements.[web:82][web:85]

---

## 7. Example Smaller PE Benchmark: SOMLAP

While not as widely used as EMBER or SOREL‑20M, SOMLAP illustrates the design of a smaller static PE header dataset with explicit malware/benign ratios.[web:81]

- **Scope and size**  
  - Total: **51 409 PE samples**.[web:81]  
  - Malware: **19 809 samples** (**38.54%**), collected from VirusShare.[web:81]  
  - Benign: **31 600 samples** (**61.46%**), gathered from Windows 10 executables and DLLs.[web:81]  

- **Features and OS coverage**  
  - Platform: **Windows PE** only.[web:81]  
  - Each sample is described by **108 PE header attributes**, focusing on structural metadata rather than full byte streams.[web:81]  

- **Use cases**  
  - Suitable for exploring classical ML over header features and for experiments where a moderate, well‑documented dataset is sufficient.[web:81]

---

## 8. Comparative Summary

### 8.1 Size, labels, and platform coverage

| Dataset      | Total samples      | Malware count (% of labeled)                           | Benign count (% of labeled)            | OS / formats                          |
|-------------|--------------------|--------------------------------------------------------|----------------------------------------|---------------------------------------|
| EMBER 2018  | 1.1 M              | 400 k labeled (300 k train + 100 k test) → 50% mal of labeled; unlabeled 300 k.[web:61][web:80] | 400 k labeled (300 k train + 100 k test) → 50% benign of labeled.[web:61] | Windows PE executables/DLLs only.[web:61][web:79] |
| SOREL‑20M   | ≈20 M features     | Part of 20 M labeled corpus; additionally ≈10 M disarmed malware binaries.[web:71][web:12] | Remaining portion of 20 M labeled corpus.[web:71] | Windows PE only.[web:71] |
| BODMAS      | 134 435            | 57 293 malware (**≈42.6%**).[web:83][web:77]           | 77 142 benign (**≈57.4%**).[web:83][web:77] | Windows PE only.[web:83] |
| EMBER2024   | >3.2 M             | Multiple tasks; binary mal/ben distribution stratified per week and file type; includes 6315 evasive malicious samples in challenge set.[web:84][web:62][web:67] | Benign samples drawn from same VT stream with ≤0 AV detections after rescan.[web:84] | Win32, Win64, .NET, APK, ELF, PDF.[web:84][web:67] |
| MalwareBazaar | Millions (live) | Varies; primarily malware corpus with per‑sample family tags and AV detections; no standardized benign set.[web:73][web:68] | No dedicated benign partition; used mainly for malicious samples.[web:73] | Mixed formats (PE, APK, ELF, scripts, documents).[web:73][web:68] |
| VirusShare  | ≥27 M labeled subset; 487 archives ≈16.8 TB total.[web:68][web:82] | Predominantly malware; 27 M samples labeled as malicious in one VT‑based effort.[web:82] | No official benign partition.[web:82] | Mainly Windows PE, plus other malicious file types.[web:68][web:82] |
| SOMLAP      | 51 409             | 19 809 malware (**38.54%**).[web:81]                   | 31 600 benign (**61.46%**).[web:81]     | Windows PE only.[web:81] |

### 8.2 Design focus

- **EMBER 2018**: Balanced, static PE feature benchmark for classical ML and deep learning baselines.[web:61][web:80]  
- **SOREL‑20M**: Large‑scale, vendor‑labeled PE corpus with rich metadata and disarmed malware binaries.[web:71][web:12]  
- **BODMAS**: Timestamped PE dataset with curated family labels for temporal and family‑aware analysis.[web:83][web:86]  
- **EMBER2024**: Multi‑platform, multi‑task dataset with an explicit evasive‑malware challenge set and updated features.[web:84][web:62][web:67]  
- **MalwareBazaar**: Fresh, in‑the‑wild malware feed with rich tags, driven by defender collaboration.[web:73][web:68]  
- **VirusShare**: Massive raw malware corpus requiring external labeling, ideal as a base for custom benchmarks.[web:68][web:82][web:85]  
- **SOMLAP**: Moderate‑size PE header dataset with explicitly documented malware/benign proportions.[web:81]  

---

## 9. Implications for Evaluation

- Studies relying solely on **balanced static PE benchmarks** such as EMBER may overestimate real‑world performance; production malware prevalence is typically much lower than 50%, and datasets like BODMAS and SOREL‑20M allow more realistic ratios.[web:61][web:71][web:83]  
- Explicit timestamps in **BODMAS, SOREL‑20M, and EMBER2024** are essential for evaluating concept drift; mixing time windows in cross‑validation hides degradation over time.[web:71][web:83][web:84]  
- Robustness to **evasive or low‑detection malware** should be measured using challenge subsets (EMBER2024) or low‑AV‑score samples mined from MalwareBazaar or VirusShare, rather than only standard test splits.[web:62][web:84][web:73][web:82]  


