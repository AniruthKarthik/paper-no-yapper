import os
import csv
import logging
import pdfplumber
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Set

# --- Configuration ---
PAPERS_DIR = "papers"
CSV_FILE = "result.csv"
LOG_FILE = "log.txt"
# ---------------------

def setup_logging():
    """Configures logging to file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )

def get_csv_headers() -> List[str]:
    """Returns the 32 column headers for the CSV."""
    # This list now has 32 columns, including 'Source_File' at the end
    return [
        "Paper_ID", "Title", "Authors", "Year", "Venue_Name", "Venue_Type",
        "DOI_or_Link", "Primary_Problem", "Secondary_Problems",
        "High_Level_Domains", "Malware_Types_Targeted", "Analysis_Type",
        "Analysis_Subtype", "Core_Techniques", "Specific_Algorithms_Models",
        "Contribution_Type", "Feature_Representation", "Feature_Engineering_Steps",
        "Data_Preprocessing_Steps", "Software_Tools_Used", "Libraries_Frameworks_Used",
        "Programming_Languages", "Datasets_Used", "Dataset_Composition",
        "Evaluation_Metrics", "Baselines_Compared_Against", "Reported_Performance_Summary",
        "Stated_Contributions", "Stated_Limitations", "Future_Work_Ideas",
        "Reviewer_Notes", "Source_File"  # This last one is key for tracking
    ]

def setup_csv():
    """Creates the CSV file with headers if it doesn't exist."""
    headers = get_csv_headers()
    if not os.path.exists(CSV_FILE):
        try:
            with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            logging.info(f"Created new {CSV_FILE} with headers.")
        except IOError as e:
            logging.error(f"Could not create {CSV_FILE}: {e}")
            exit()

def get_processed_files() -> Set[str]:
    """Reads the CSV to find which files have already been processed."""
    processed = set()
    if not os.path.exists(CSV_FILE):
        return processed
        
    try:
        with open(CSV_FILE, 'r', encoding='utf-8') as f:
            # Skip header
            next(f, None)
            reader = csv.reader(f)
            for row in reader:
                # Check for a valid row and get the last column
                if row and len(row) == len(get_csv_headers()):
                    processed.add(row[-1])  # Add the 'Source_File'
    except Exception as e:
        logging.warning(f"Could not read processed files from {CSV_FILE}: {e}")
        
    return processed

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts all text from a PDF file."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        logging.error(f"Could not extract text from {pdf_path}: {e}")
        return None

def build_gemini_prompt(paper_text: str, source_file: str) -> str:
    """Creates the detailed prompt for the Gemini API."""
    
    headers = get_csv_headers()
    
    # Remove 'Source_File' from instructions, as we provide it
    headers_for_prompt = ", ".join([f'"{h}"' for h in headers[:-1]])
    
    prompt = f"""
    [ROLE]
    You are a meticulous research assistant. Your task is to analyze the provided text from a scientific paper and extract specific information.

    [INPUT]
    The following is the FULL text from a paper named "{source_file}". Analyze it carefully from start to finish.

    [PAPER_TEXT_BEGIN]
    {paper_text}
    [PAPER_TEXT_END]

    [INSTRUCTIONS]
    Your entire response MUST be a single line of CSV-formatted data.
    1. The line must have exactly {len(headers)} columns.
    2. Use a comma (,) as the delimiter.
    3. Enclose EVERY field in double quotes ("").
    4. If information for a field is not found in the text, use an empty string: "".
    5. Escape any double quotes inside a field with another double quote (e.g., "This is a ""quote"".").
    6. For fields that accept multiple values (e.g., Authors, Datasets_Used), separate items with a semicolon (;) INSIDE the double quotes (e.g., "Saha, S.; Afroz, S.").

    [COLUMNS_TO_FILL]
    The {len(headers) - 1} columns you must find are:
    {headers_for_prompt}

    [FINAL_COLUMN]
    The {len(headers)}th and final column, "Source_File", MUST be exactly: "{source_file}"

    Provide ONLY the single CSV data line in your response.
    """
    return prompt.strip()

def main():
    setup_logging()
    load_dotenv()
    
    api_key = os.getenv("GEMINI_KEY")
    if not api_key:
        logging.error("GEMINI_KEY not found in .env file. Exiting.")
        return
        
    try:
        genai.configure(api_key=api_key)
        # Using a model with a la# This is the NEW, correct line
        model = genai.GenerativeModel('gemini-2.5-pro')
    except Exception as e:
        logging.error(f"Could not configure Gemini API: {e}")
        return

    setup_csv()
    
    processed_files = get_processed_files()
    if processed_files:
        logging.info(f"Found {len(processed_files)} previously processed files.")
        
    if not os.path.exists(PAPERS_DIR):
        logging.error(f"Directory not found: {PAPERS_DIR}. Please create it.")
        return

    all_pdfs = [f for f in os.listdir(PAPERS_DIR) if f.lower().endswith(".pdf")]
    files_to_process = [f for f in all_pdfs if f not in processed_files]
    
    logging.info(f"Found {len(all_pdfs)} total PDFs. {len(files_to_process)} are new.")
    
    for paper_file in files_to_process:
        logging.info(f"--- Processing: {paper_file} ---")
        pdf_path = os.path.join(PAPERS_DIR, paper_file)
        
        try:
            # 1. Extract FULL Text
            full_text = extract_text_from_pdf(pdf_path)
            if not full_text or full_text.isspace():
                logging.warning(f"Skipping {paper_file}: No text could be extracted.")
                continue

            # 2. Build Prompt with FULL text
            prompt = build_gemini_prompt(full_text, paper_file)

            # 3. Call API
            logging.info(f"  ...Sending {len(full_text)} chars (full paper) to Gemini API...")
            response = model.generate_content(
                prompt,
                generation_config={"temperature": 0.0} # We want factual, not creative
            )
            
            # 4. Save to CSV
            csv_row_text = response.text.strip()
            
            # Basic validation
            if not csv_row_text.startswith('"') or not csv_row_text.endswith(f'"{paper_file}"'):
                logging.error(f"  ...Failed to process {paper_file}: API returned malformed data.")
                logging.debug(f"  ...API Response: {csv_row_text}")
                continue

            # This is the "save immediately" step
            with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
                f.write(csv_row_text + "\n")
            
            logging.info(f"  ...Successfully processed and saved {paper_file}.")

        except Exception as e:
            logging.error(f"  ...An unexpected error occurred while processing {paper_file}: {e}")
            logging.info(f"  ...Moving to next paper.")

    logging.info("--- All new papers processed. ---")

if __name__ == "__main__":
    main()
