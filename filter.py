import csv

required = "required.csv"
impreq = "impreq.csv"
output  = "newreq.csv"

# SELECT the unique comparison column here:
UNIQUE_COL = "EID"   # you can change to "DOI", "Title", etc.

# Load impreq values
with open(impreq, newline='', encoding='utf-8') as f:
    r = csv.DictReader(f)
    impreq_keys = set(row[UNIQUE_COL].strip() for row in r)

# Process required.csv
with open(required, newline='', encoding='utf-8') as fin, open(output, "w", newline='', encoding='utf-8') as fout:
    reader = csv.DictReader(fin)
    writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
    writer.writeheader()
    
    count_removed = 0
    for row in reader:
        if row[UNIQUE_COL].strip() not in impreq_keys:
            writer.writerow(row)
        else:
            count_removed += 1

print(f"Done. Removed {count_removed} rows based on column = {UNIQUE_COL}")
print(f"Output written to {output}")
