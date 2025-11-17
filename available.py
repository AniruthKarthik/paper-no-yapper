
import csv

def get_required_dois(required_file):
    """Reads the 'DOI' column from the required.csv file."""
    dois = set()
    with open(required_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
            doi_index = header.index('DOI')
        except (StopIteration, ValueError):
            # Handle empty file or missing 'DOI' column
            return dois
            
        for row in reader:
            if len(row) > doi_index:
                dois.add(row[doi_index])
    return dois

def filter_scopus_file(scopus_file, required_dois, available_file):
    """Filters records from scopus.csv and writes to available.csv."""
    with open(scopus_file, 'r', encoding='utf-8') as f_scopus, \
         open(available_file, 'w', newline='', encoding='utf-8') as f_available:
        
        scopus_reader = csv.reader(f_scopus)
        writer = csv.writer(f_available)
        
        try:
            header = next(scopus_reader)
            writer.writerow(header)
            doi_index = header.index('DOI')
        except (StopIteration, ValueError):
            # Handle empty scopus file or missing 'DOI' column
            return

        for row in scopus_reader:
            if len(row) > doi_index:
                if row[doi_index] not in required_dois:
                    writer.writerow(row)

if __name__ == "__main__":
    required_dois = get_required_dois('required.csv')
    filter_scopus_file('scopus.csv', required_dois, 'available.csv')
    print("Filtered records have been saved to available.csv")
