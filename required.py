import pandas as pd
import numpy as np

# Define the exact column list
COLUMNS = [
    "Authors", "Author full names", "Author(s) ID", "Title", "Year",
    "Source title", "Volume", "Issue", "Art. No.", "Page start",
    "Page end", "Page count", "Cited by", "DOI", "Link",
    "Affiliations", "Authors with affiliations", "Abstract",
    "Author Keywords", "Index Keywords", "Funding Details",
    "Funding Texts", "Correspondence Address", "Editors",
    "Publisher", "Sponsors", "Conference name", "Conference date",
    "Conference location", "Conference code", "ISSN", "ISBN",
    "CODEN", "Abbreviated Source Title", "Document Type",
    "Publication Stage", "Open Access", "Source", "EID"
]

# Publisher indicators for important papers
PUBLISHER_INDICATORS = [
    'IEEE', 'Elsevier', 'ACM', 'Springer', 'Nature', 'Science',
    'AAAI', 'Wiley', 'Cell Press', 'IET', 'Royal Society',
    'MIT Press', 'SIAM', 'Oxford', 'Cambridge', 'AAAS'
]

print("Loading CSV files...")

# Load CSV files with robust parsing
all_df = pd.read_csv('all.csv', 
                     dtype=str,  # Read all as string initially
                     skipinitialspace=True,
                     encoding='utf-8',
                     on_bad_lines='warn')

available_df = pd.read_csv('available.csv',
                           dtype=str,
                           skipinitialspace=True,
                           encoding='utf-8',
                           on_bad_lines='warn')

print(f"Loaded all.csv: {len(all_df)} rows, {len(all_df.columns)} columns")
print(f"Loaded available.csv: {len(available_df)} rows, {len(available_df.columns)} columns")

# Clean EID column (strip whitespace, handle NaN)
all_df['EID'] = all_df['EID'].astype(str).str.strip()
available_df['EID'] = available_df['EID'].astype(str).str.strip()

# Remove any rows where EID is 'nan' or empty
all_df = all_df[all_df['EID'].notna() & (all_df['EID'] != 'nan') & (all_df['EID'] != '')]
available_df = available_df[available_df['EID'].notna() & (available_df['EID'] != 'nan') & (available_df['EID'] != '')]

print(f"After EID cleaning: all.csv has {len(all_df)} rows, available.csv has {len(available_df)} rows")

# Create required.csv: papers in all.csv but NOT in available.csv
available_eids = set(available_df['EID'].unique())
required_df = all_df[~all_df['EID'].isin(available_eids)].copy()

print(f"\n✓ Missing (required) papers: {len(required_df)}")

# Save required.csv
required_df.to_csv('required.csv', index=False, encoding='utf-8')
print(f"✓ Saved required.csv with {len(required_df)} rows")

# Create impreq.csv: important papers
# Convert 'Cited by' to numeric for comparison, handling errors
all_df['Cited by numeric'] = pd.to_numeric(all_df['Cited by'], errors='coerce')
all_df['Cited by numeric'] = all_df['Cited by numeric'].fillna(0)

# Condition 1: Check if Source title contains any publisher indicator
def has_publisher_indicator(source_title):
    if pd.isna(source_title) or source_title == 'nan':
        return False
    source_lower = str(source_title).lower()
    return any(indicator.lower() in source_lower for indicator in PUBLISHER_INDICATORS)

all_df['has_publisher'] = all_df['Source title'].apply(has_publisher_indicator)

# Condition 2: Cited by >= 100
all_df['highly_cited'] = all_df['Cited by numeric'] >= 100

# Important papers: satisfy at least one condition
impreq_df = all_df[all_df['has_publisher'] | all_df['highly_cited']].copy()

# Remove temporary columns
impreq_df = impreq_df.drop(columns=['Cited by numeric', 'has_publisher', 'highly_cited'])

# Remove duplicates based on EID
impreq_df = impreq_df.drop_duplicates(subset=['EID'], keep='first')

print(f"\n✓ Important papers: {len(impreq_df)}")

# Save impreq.csv
impreq_df.to_csv('impreq.csv', index=False, encoding='utf-8')
print(f"✓ Saved impreq.csv with {len(impreq_df)} rows")

# Top 20 most-cited papers
print("\n" + "="*80)
print("TOP 20 MOST-CITED PAPERS")
print("="*80)

top_cited = all_df.nlargest(20, 'Cited by numeric')[['Title', 'Cited by']].copy()
for idx, (_, row) in enumerate(top_cited.iterrows(), 1):
    title = str(row['Title'])[:80] + "..." if len(str(row['Title'])) > 80 else str(row['Title'])
    cited = row['Cited by']
    print(f"{idx:2d}. [{cited:>6s} citations] {title}")

print("\n" + "="*80)
print("PROCESSING COMPLETE")
print("="*80)
print(f"✓ required.csv: {len(required_df)} papers (in all.csv but NOT in available.csv)")
print(f"✓ impreq.csv: {len(impreq_df)} important papers (unique by EID)")
print(f"✓ All columns preserved intact")
