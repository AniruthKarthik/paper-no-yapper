import csv
import os

def get_normalized_dois_from_file(filename):
    """Reads a CSV file and returns a set of normalized (lowercase, stripped) DOIs."""
    dois = set()
    if not os.path.exists(filename):
        print(f"Info: {filename} not found.")
        return dois
    
    with open(filename, 'r', newline='', encoding='utf-8') as f:
        try:
            reader = csv.DictReader(f)
            for row in reader:
                if row and 'DOI' in row and row['DOI']:
                    dois.add(row['DOI'].strip().lower())
        except (csv.Error, KeyError) as e:
            print(f"Warning: Could not process {filename}. It might be empty or malformed. Error: {e}")
    return dois

def update_csv_files():
    """
    Merges unique records from grokavail.csv and claudeavail.csv into available.csv,
    and removes them from required.csv.
    """
    # --- 1. Gather DOIs from all source files ---
    grok_dois = get_normalized_dois_from_file('grokavail.csv')
    claude_dois = get_normalized_dois_from_file('claudeavail.csv')
    existing_available_dois = get_normalized_dois_from_file('available.csv')
    
    print(f"Found {len(grok_dois)} unique DOIs in grokavail.csv")
    print(f"Found {len(claude_dois)} unique DOIs in claudeavail.csv")
    
    source_dois = grok_dois.union(claude_dois)
    print(f"Found {len(source_dois)} combined unique DOIs in source files.")
    
    # --- 2. Determine which new records to add ---
    dois_to_add = source_dois - existing_available_dois
    print(f"Found {len(existing_available_dois)} existing unique DOIs in available.csv.")
    print(f"This means {len(dois_to_add)} new unique records will be added.")

    if not dois_to_add:
        print("No new records to add to available.csv.")
    else:
        # --- 3. Append new unique records to available.csv ---
        new_rows = []
        # Use a copy of the set to keep track of which DOIs we still need to find
        dois_to_find = dois_to_add.copy()

        for filename in ['grokavail.csv', 'claudeavail.csv']:
            if not os.path.exists(filename):
                continue
            with open(filename, 'r', newline='', encoding='utf-8') as infile:
                reader = csv.DictReader(infile)
                for row in reader:
                    if row and row.get('DOI'):
                        normalized_doi = row['DOI'].strip().lower()
                        if normalized_doi in dois_to_find:
                            new_rows.append(row)
                            dois_to_find.remove(normalized_doi)
        
        available_file = 'available.csv'
        available_header_present = os.path.exists(available_file) and os.path.getsize(available_file) > 0
        
        # Get header from the first new row
        fieldnames = new_rows[0].keys() if new_rows else []
        
        with open(available_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not available_header_present and fieldnames:
                writer.writeheader()
            writer.writerows(new_rows)
        print(f"Successfully appended {len(new_rows)} new records to {available_file}.")

    # --- 4. Filter required.csv ---
    required_file = 'required.csv'
    if not os.path.exists(required_file):
        print("Info: required.csv not found. Nothing to remove.")
        return
        
    temp_file = 'required.tmp'
    original_row_count = 0
    rows_removed = 0

    with open(required_file, 'r', newline='', encoding='utf-8') as infile, \
         open(temp_file, 'w', newline='', encoding='utf-8') as outfile:
        try:
            reader = csv.DictReader(infile)
            writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
            writer.writeheader()
            
            for row in reader:
                original_row_count += 1
                if row.get('DOI') and row['DOI'].strip().lower() in source_dois:
                    rows_removed += 1
                else:
                    writer.writerow(row)
        except (csv.Error, KeyError, StopIteration) as e:
             print(f"Could not process {required_file}. Error: {e}")
             os.remove(temp_file)
             return

    os.replace(temp_file, required_file)
    print(f"Removed {rows_removed} records from {required_file}. Original count: {original_row_count}, New count: {original_row_count - rows_removed}")


if __name__ == "__main__":
    update_csv_files()