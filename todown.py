import csv
import os
import re

def clean_text(text):
    """Remove special characters and convert to lowercase for comparison."""
    return re.sub(r'[^a-zA-Z0-9]', '', text).lower()

def pdf_exists(title, doi, pdf_files):
    """Check if a PDF exists for a given title or DOI."""
    cleaned_title = clean_text(title)
    cleaned_doi = clean_text(doi)

    for pdf_file in pdf_files:
        cleaned_pdf = clean_text(os.path.splitext(pdf_file)[0])
        if cleaned_title in cleaned_pdf or cleaned_doi in cleaned_pdf:
            return True
    return False

def main():
    """
    Reads scopus.csv, checks for existing PDFs, and writes missing papers to required.csv.
    """
    scopus_file = 'scopus.csv'
    required_file = 'required.csv'
    
    # Get all PDF files in the current directory
    pdf_files = [f for f in os.listdir('papers') if f.lower().endswith('.pdf')]

    with open(scopus_file, 'r', encoding='utf-8') as infile, \
         open(required_file, 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)
        
        # Write header to required.csv
        writer.writerow(['DOI', 'Title', 'Link','Year'])
        
        for row in reader:
            title = row.get('Title', '')
            doi = row.get('DOI', '')
            link = row.get('Link', '')
            year = row.get('Year', '')

            if not pdf_exists(title, doi, pdf_files):
                writer.writerow([doi, title, link,year])
                print(title)

if __name__ == "__main__":
    main()
