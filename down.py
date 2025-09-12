import pandas as pd
import requests
import os
import time
import sys
import select
import tty
import termios

def download_pdfs(csv_path, download_dir, email):
    # Validate CSV file existence and accessibility
    if not os.path.exists(csv_path):
        raise ValueError(f"CSV file not found at: {csv_path}")
    if not os.access(csv_path, os.R_OK):
        raise ValueError(f"CSV file at {csv_path} is not readable (check permissions).")

    # Check file size and content
    file_size = os.path.getsize(csv_path)
    print(f"CSV file size: {file_size} bytes")
    with open(csv_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        print(f"First line of CSV: {first_line}")

    # Read CSV with flexible parsing
    try:
        df = pd.read_csv(csv_path, encoding='utf-8', on_bad_lines='warn', engine='python')
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {str(e)}")

    # Check if DataFrame is empty
    print(f"CSV loaded with {len(df)} rows and {len(df.columns)} columns")
    if df.empty:
        print("CSV is empty or no valid data was parsed. Check file format, delimiter, or content.")
        return

    # Validate required columns
    required_columns = ['Title', 'Link', 'DOI']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV must have {required_columns} columns. Found: {list(df.columns)}")

    # Create download directory
    os.makedirs(download_dir, exist_ok=True)
    progress_path = os.path.join(download_dir, 'progress.txt')
    tobedown_path = os.path.join(download_dir, 'tobedown.csv')

    # Load progress if exists
    start_from = 0
    if os.path.exists(progress_path):
        try:
            with open(progress_path, 'r') as f:
                start_from = int(f.read().strip()) + 1
        except (ValueError, FileNotFoundError):
            print("Invalid or corrupted progress file. Starting from beginning.")
            start_from = 0

    # Initialize list for failed rows
    failed_rows = []

    # Set up non-blocking input
    def is_data():
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

    old_settings = termios.tcgetattr(sys.stdin)
    last_index = start_from  # Track the last processed index
    try:
        tty.setcbreak(sys.stdin.fileno())

        for i in range(start_from, len(df)):
            # Check for 'q' input
            if is_data():
                c = sys.stdin.read(1)
                if c.lower() == 'q':
                    print("Quit signal received. Saving progress...")
                    break

            row = df.iloc[i]
            number = i + 1
            last_index = i  # Update last processed index

            # Skip rows with missing values
            if pd.isna(row['DOI']) or pd.isna(row['Link']) or pd.isna(row['Title']):
                print(f"{number}. Skipping row with missing values: {row.to_dict()}")
                failed_rows.append(row)
                with open(progress_path, 'w') as f:
                    f.write(str(i))
                continue

            doi = str(row['DOI']).strip()
            link = str(row['Link']).strip()
            title = str(row['Title']).strip()

            api_url = f"https://api.unpaywall.org/v2/{doi}?email={email}"

            try:
                response = requests.get(api_url, timeout=10)
                response.raise_for_status()
                data = response.json()
                best_oa = data.get('best_oa_location')
                if best_oa and best_oa.get('url_for_pdf'):
                    pdf_url = best_oa['url_for_pdf']
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    pdf_response = requests.get(pdf_url, headers=headers, timeout=30, stream=True)
                    pdf_response.raise_for_status()
                    safe_filename = doi.replace('/', '_').replace(':', '_') + '.pdf'
                    file_path = os.path.join(download_dir, safe_filename)
                    with open(file_path, 'wb') as f:
                        for chunk in pdf_response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    print(f"{number}. Downloaded: {file_path}")
                else:
                    failed_rows.append(row)
                    print(f"{number}. No OA PDF found for DOI: {doi}, added to failed list")
            except requests.exceptions.RequestException as e:
                failed_rows.append(row)
                print(f"{number}. Error processing DOI {doi}: {str(e)}, added to failed list")

            # Save progress after each item
            with open(progress_path, 'w') as f:
                f.write(str(i))

            time.sleep(0.1)

        # Save failed rows to tobedown.csv
        if failed_rows:
            failed_df = pd.DataFrame(failed_rows)
            failed_df.to_csv(tobedown_path, index=False)
            print(f"Failed records saved to {tobedown_path} ({len(failed_rows)} rows)")
        else:
            print("No failed records; tobedown.csv not created.")

        # If completed all, remove progress file
        if last_index >= len(df) - 1:
            if os.path.exists(progress_path):
                os.remove(progress_path)
            print("All PDFs processed successfully.")
        else:
            print(f"Progress saved. Rerun to resume from item {last_index + 2}.")

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

if __name__ == "__main__":
    csv_path = "../papers.csv"
    download_dir = "./pdfs"
    email = "aniruth111gl@gmail.com"
    try:
        download_pdfs(csv_path, download_dir, email)
    except Exception as e:
        print(f"Script failed: {str(e)}")
        exit(1)
