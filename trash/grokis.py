import csv
import os
import re
import requests
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import signal
import sys
from pathlib import Path
import shutil
from datetime import datetime

# Configuration
MAX_RETRIES = 3
MAX_WORKERS = 10
PAPERS_PER_EMAIL = 70
DOWNLOAD_DIR = 'grokdown'
REQUIRED_CSV = 'grokreq.csv'
AVAILABLE_CSV = 'grokavail.csv'
TODO_CSV = 'groktodo.csv'
TEMP_SUFFIX = '.tmp'

# Global state
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
})

# Thread-safe globals
state_lock = threading.Lock()
shutdown_event = threading.Event()
active_downloads = 0
downloaded_count = 0
failed_count = 0
processed_count = 0
current_email_index = 0
papers_since_last_switch = 0
log_file = None

# Gmail ID pool
gmail_ids = [f"gemini_researcher_{i:03d}@gmail.com" for i in range(1, 151)]


def sanitize_filename(filename):
    """Remove invalid characters from filename."""
    filename = re.sub(r'[\\/\:\*\?"<>\|]', "", filename)
    return filename[:200].strip()


def log(message, to_console=True, to_file=True):
    """Thread-safe logging to console and file."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted = f"[{timestamp}] {message}"
    
    if to_console:
        print(formatted)
    
    if to_file and log_file:
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(formatted + '\n')
        except:
            pass


def atomic_csv_write(filepath, rows, fieldnames):
    """Atomically write CSV file using temp file and rename."""
    temp_path = filepath + TEMP_SUFFIX
    try:
        with open(temp_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        # Atomic rename
        shutil.move(temp_path, filepath)
        return True
    except Exception as e:
        log(f"Error writing {filepath}: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False


def read_csv_safe(filepath):
    """Safely read CSV file with BOM handling."""
    if not os.path.exists(filepath):
        return []
    
    try:
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            return list(csv.DictReader(f))
    except Exception as e:
        log(f"Error reading {filepath}: {e}")
        return []


def normalize_fieldnames(row):
    """Remove BOM and normalize field names."""
    return {k.strip().strip('\ufeff').strip('"'): v for k, v in row.items()}


def move_paper_to_available(paper_row):
    """Move paper from required.csv to available.csv atomically."""
    with state_lock:
        # Normalize the paper row
        paper_row = normalize_fieldnames(paper_row)
        
        # Read both files
        required_papers = read_csv_safe(REQUIRED_CSV)
        available_papers = read_csv_safe(AVAILABLE_CSV)
        
        # Normalize all rows
        required_papers = [normalize_fieldnames(p) for p in required_papers]
        available_papers = [normalize_fieldnames(p) for p in available_papers]
        
        # Remove from required
        doi = paper_row.get('DOI', '').strip()
        required_papers = [p for p in required_papers if p.get('DOI', '').strip() != doi]
        
        # Add to available with timestamp
        paper_row['Downloaded'] = datetime.now().isoformat()
        available_papers.append(paper_row)
        
        # Get fieldnames
        if required_papers:
            required_fieldnames = list(required_papers[0].keys())
        else:
            required_fieldnames = list(paper_row.keys())
        
        available_fieldnames = list(paper_row.keys())
        
        # Write both files atomically
        success = True
        success &= atomic_csv_write(REQUIRED_CSV, required_papers, required_fieldnames)
        success &= atomic_csv_write(AVAILABLE_CSV, available_papers, available_fieldnames)
        
        return success


def update_paper_status(doi, status, timestamp_key=None):
    """Update status for a paper in todo.csv atomically."""
    with state_lock:
        todo_papers = read_csv_safe(TODO_CSV)
        todo_papers = [normalize_fieldnames(p) for p in todo_papers]
        
        updated = False
        for p in todo_papers:
            if p.get('DOI', '').strip() == doi:
                p['Status'] = status
                if timestamp_key and status == 'success':
                    p[timestamp_key] = datetime.now().isoformat()
                updated = True
                break
        
        if updated:
            fieldnames = sorted(list(set().union(*(set(p.keys()) for p in todo_papers))))
            atomic_csv_write(TODO_CSV, todo_papers, fieldnames)
        
        return updated


def load_all_papers():
    """Load and merge unique papers from required.csv and available.csv."""
    required = read_csv_safe(REQUIRED_CSV)
    available = read_csv_safe(AVAILABLE_CSV)
    all_papers = [normalize_fieldnames(p) for p in required + available]
    
    # Unique by DOI
    unique_papers = {}
    for p in all_papers:
        doi = p.get('DOI', '').strip()
        if doi:
            unique_papers[doi] = p
    
    return list(unique_papers.values())


def initialize_todo_csv(all_papers):
    """Initialize or update todo.csv with all papers."""
    todo_papers = []
    todo_dois = set()
    
    if os.path.exists(TODO_CSV):
        todo_papers = read_csv_safe(TODO_CSV)
        todo_papers = [normalize_fieldnames(p) for p in todo_papers]
        todo_dois = {p.get('DOI', '').strip() for p in todo_papers if p.get('DOI', '').strip()}
    
    added = False
    for p in all_papers:
        doi = p.get('DOI', '').strip()
        if doi and doi not in todo_dois:
            p['Status'] = 'pending'
            todo_papers.append(p)
            added = True
        elif doi in todo_dois:
            # Find and update if needed (e.g., new fields), but skip for simplicity
            pass
    
    if added or not os.path.exists(TODO_CSV):
        fieldnames = sorted(list(set().union(*(set(p.keys()) for p in todo_papers))))
        atomic_csv_write(TODO_CSV, todo_papers, fieldnames)
        log(f"Updated {TODO_CSV} with {len(todo_papers)} papers")
    
    return todo_papers


def download_paper(url, title, directory):
    """Download paper from URL."""
    if not url or shutdown_event.is_set():
        return False
        
    filepath = os.path.join(directory, sanitize_filename(title) + '.pdf')
    
    # Always attempt download (overwrite if exists)

    for attempt in range(MAX_RETRIES):
        if shutdown_event.is_set():
            return False
            
        try:
            # Fix URL format
            if url.startswith('//'):
                url = 'https:' + url
            elif not url.startswith('http'):
                continue
            
            response = session.get(url, stream=True, timeout=15)
            response.raise_for_status()
            
            # Check if it's actually a PDF
            content_type = response.headers.get('content-type', '').lower()
            if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
                # Download the file
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if shutdown_event.is_set():
                            return False
                        f.write(chunk)
                
                # Verify file size
                if os.path.getsize(filepath) > 1024:
                    return True
                else:
                    os.remove(filepath)
                    return False
            else:
                # If HTML, try to find PDF link
                soup = BeautifulSoup(response.content, 'html.parser')
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if '.pdf' in href.lower():
                        pdf_url = urljoin(url, href)
                        return download_paper(pdf_url, title, directory)
                return False
                
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
            continue
    
    return False


# === ALL DOWNLOAD METHODS FROM BOTH SCRIPTS ===

def get_unpaywall_url(doi, email_id):
    """Get PDF URL from Unpaywall API."""
    try:
        url = f"https://api.unpaywall.org/v2/{doi}?email={email_id}"
        response = session.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('is_oa', False):
                best_location = data.get('best_oa_location')
                if best_location and best_location.get('url_for_pdf'):
                    return best_location['url_for_pdf']
                
                oa_locations = data.get('oa_locations', [])
                for location in oa_locations:
                    if location.get('url_for_pdf'):
                        return location['url_for_pdf']
    except:
        pass
    return None


def get_arxiv_url(doi):
    """Get arXiv PDF URL."""
    try:
        if 'arxiv' in doi.lower():
            arxiv_id = doi.split('/')[-1]
            arxiv_id = re.sub(r'v\d+$', '', arxiv_id)
            return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    except:
        pass
    return None


def get_semantic_scholar_url(doi):
    """Get PDF URL from Semantic Scholar API."""
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/{doi}?fields=openAccessPdf"
        response = session.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('openAccessPdf') and data['openAccessPdf'].get('url'):
                return data['openAccessPdf']['url']
    except:
        pass
    return None


def get_openalex_url(doi):
    """Get PDF URL from OpenAlex API."""
    try:
        url = f"https://api.openalex.org/works/https://doi.org/{doi}"
        response = session.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('open_access') and data['open_access'].get('oa_url'):
                return data['open_access']['oa_url']
    except:
        pass
    return None


def get_loc_url(doi):
    """Get PDF URL from Library of Congress API."""
    try:
        url = f"https://chroniclingamerica.loc.gov/search/pages/results/?proxtext={doi}&format=json"
        response = session.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('items') and len(data['items']) > 0 and data['items'][0].get('pdf'):
                return data['items'][0]['pdf']
    except:
        pass
    return None


def get_base_url(doi):
    """Get PDF URL from BASE API."""
    try:
        url = f"https://api.base-search.net/cgi-bin/BaseHttpSearch?func=PerformSearch&query=doi:{doi}&format=json"
        response = session.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('response') and data['response'].get('docs') and len(data['response']['docs']) > 0:
                if data['response']['docs'][0].get('pdf_url'):
                    return data['response']['docs'][0]['pdf_url']
    except:
        pass
    return None


def get_core_url(doi):
    """Get PDF URL from CORE API."""
    try:
        url = f"https://api.core.ac.uk/v3/search/works?q=doi:{doi}"
        response = session.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('results') and len(data['results']) > 0:
                if data['results'][0].get('downloadUrl'):
                    return data['results'][0]['downloadUrl']
    except:
        pass
    return None


def get_doaj_url(doi):
    """Get PDF URL from DOAJ API."""
    try:
        url = f"https://doaj.org/api/v1/search/articles/doi:{doi}"
        response = session.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('results') and len(data['results']) > 0:
                links = data['results'][0].get('bibjson', {}).get('link', [])
                for link in links:
                    if link.get('type') == 'fulltext':
                        return link.get('url')
    except:
        pass
    return None


def get_scihub_url(doi):
    """Get PDF URL from Sci-Hub."""
    scihub_domains = [
        "https://sci-hub.se",
        "https://sci-hub.st",
        "https://sci-hub.ru",
        "https://sci-hub.ee"
    ]
    
    for domain in scihub_domains:
        if shutdown_event.is_set():
            return None
            
        try:
            url = f"{domain}/{doi}"
            response = session.get(url, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for PDF iframe
                pdf_iframe = soup.find('iframe', {'id': 'pdf'})
                if pdf_iframe and pdf_iframe.get('src'):
                    pdf_url = pdf_iframe['src']
                    if not pdf_url.startswith('http'):
                        pdf_url = urljoin(domain, pdf_url)
                    return pdf_url
                
                # Look for direct download links
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if '.pdf' in href or 'download' in href.lower():
                        if not href.startswith('http'):
                            href = urljoin(domain, href)
                        return href
        except:
            continue
    return None


def get_publisher_url(doi):
    """Get PDF from publisher website."""
    try:
        url = f"https://doi.org/{doi}"
        response = session.get(url, timeout=15, allow_redirects=True)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for PDF links
            for link in soup.find_all('a', href=True):
                href = link['href']
                href_lower = href.lower()
                
                if any(keyword in href_lower for keyword in ['.pdf', 'download', 'pdf']):
                    if not href.startswith('http'):
                        href = urljoin(response.url, href)
                    return href
    except:
        pass
    return None


def get_europepmc_url(doi):
    """Get PDF URL from Europe PMC API."""
    try:
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:{doi}&format=json"
        response = session.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('resultList') and data['resultList'].get('result'):
                results = data['resultList']['result']
                if results and len(results) > 0:
                    if results[0].get('fullTextUrlList'):
                        urls = results[0]['fullTextUrlList'].get('fullTextUrl', [])
                        for url_obj in urls:
                            if url_obj.get('documentStyle') == 'pdf':
                                return url_obj.get('url')
    except:
        pass
    return None


def get_pubmed_url(doi):
    """Get PDF URL from PubMed."""
    try:
        # Search for DOI in PubMed
        search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={doi}&retmode=json"
        response = session.get(search_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('esearchresult') and data['esearchresult'].get('idlist'):
                pmid_list = data['esearchresult']['idlist']
                if pmid_list:
                    pmid = pmid_list[0]
                    # Try to get full text link
                    link_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={pmid}&format=json"
                    link_response = session.get(link_url, timeout=10)
                    
                    if link_response.status_code == 200:
                        link_data = link_response.json()
                        if link_data.get('records') and len(link_data['records']) > 0:
                            pmcid = link_data['records'][0].get('pmcid')
                            if pmcid:
                                return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf"
    except:
        pass
    return None


def get_biorxiv_url(doi):
    """Get PDF URL from bioRxiv/medRxiv."""
    try:
        if 'biorxiv' in doi.lower() or 'medrxiv' in doi.lower():
            # Extract the ID and construct PDF URL
            parts = doi.split('/')
            if len(parts) >= 2:
                paper_id = parts[-1]
                if 'biorxiv' in doi.lower():
                    return f"https://www.biorxiv.org/content/{doi}.full.pdf"
                elif 'medrxiv' in doi.lower():
                    return f"https://www.medrxiv.org/content/{doi}.full.pdf"
    except:
        pass
    return None


def get_researchgate_url(doi):
    """Get PDF URL from ResearchGate."""
    try:
        search_url = f"https://www.researchgate.net/search/publication?q={doi}"
        response = session.get(search_url, timeout=15)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for download links
            for link in soup.find_all('a', href=True):
                href = link['href']
                if 'publication' in href and 'fulltext' in href.lower():
                    return urljoin("https://www.researchgate.net", href)
    except:
        pass
    return None


def try_all_methods(doi, title, email):
    """Try ALL download methods comprehensively."""
    
    methods = [
        ('Unpaywall', lambda: get_unpaywall_url(doi, email), 0.5),
        ('ArXiv', lambda: get_arxiv_url(doi), 0.3),
        ('Semantic Scholar', lambda: get_semantic_scholar_url(doi), 0.5),
        ('OpenAlex', lambda: get_openalex_url(doi), 0.5),
        ('Europe PMC', lambda: get_europepmc_url(doi), 0.5),
        ('PubMed', lambda: get_pubmed_url(doi), 0.5),
        ('bioRxiv/medRxiv', lambda: get_biorxiv_url(doi), 0.3),
        ('CORE', lambda: get_core_url(doi), 0.5),
        ('DOAJ', lambda: get_doaj_url(doi), 0.5),
        ('BASE', lambda: get_base_url(doi), 0.5),
        ('Library of Congress', lambda: get_loc_url(doi), 0.5),
        ('Publisher', lambda: get_publisher_url(doi), 1.0),
        ('ResearchGate', lambda: get_researchgate_url(doi), 1.0),
        ('Sci-Hub', lambda: get_scihub_url(doi), 2.0),  # Last resort
    ]
    
    for method_name, method_func, sleep_time in methods:
        if shutdown_event.is_set():
            return False, None
        
        try:
            pdf_url = method_func()
            if pdf_url:
                success = download_paper(pdf_url, title, DOWNLOAD_DIR)
                if success:
                    return True, method_name
                time.sleep(sleep_time)
        except Exception as e:
            continue
    
    return False, None


def process_paper(paper_row, total_count, paper_index):
    """Process a single paper download."""
    global downloaded_count, failed_count, active_downloads, processed_count
    global current_email_index, papers_since_last_switch
    
    if shutdown_event.is_set():
        return
    
    with state_lock:
        active_downloads += 1
    
    try:
        paper_row = normalize_fieldnames(paper_row)
        title = paper_row.get('Title', '').strip()
        doi = paper_row.get('DOI', '').strip()
        
        if not title or not doi:
            return
        
        # Get email for this download
        with state_lock:
            email = gmail_ids[current_email_index]
            papers_since_last_switch += 1
            if papers_since_last_switch >= PAPERS_PER_EMAIL:
                current_email_index = (current_email_index + 1) % len(gmail_ids)
                papers_since_last_switch = 0
                log(f"Email switched to: {gmail_ids[current_email_index]}")
        
        with state_lock:
            processed_count += 1
            current_processed = processed_count
        
        log(f"[{current_processed}/{total_count}] Downloading: {title[:50]}...")
        
        # Try download
        success, method = try_all_methods(doi, title, email)
        
        if success and not shutdown_event.is_set():
            with state_lock:
                downloaded_count += 1
            
            # Update status in todo
            updated = update_paper_status(doi, 'success', 'ReDownloaded')
            
            # Move from required to available if not already downloaded
            downloaded_key = paper_row.get('Downloaded', '').strip()
            if not downloaded_key:
                if move_paper_to_available(paper_row):
                    log(f"[{current_processed}/{total_count}] ✓ SUCCESS via {method}: {title[:40]}...")
                else:
                    log(f"[{current_processed}/{total_count}] ✓ Downloaded but CSV update failed: {title[:40]}...")
            else:
                log(f"[{current_processed}/{total_count}] ✓ SUCCESS via {method}: {title[:40]}...")
        else:
            with state_lock:
                failed_count += 1
            update_paper_status(doi, 'failed')
            log(f"[{current_processed}/{total_count}] ✗ FAILED: {title[:40]}...")
    
    finally:
        with state_lock:
            active_downloads -= 1


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    log("\n🛑 Shutdown signal received. Finishing active downloads...")
    shutdown_event.set()


def main():
    """Main function."""
    global downloaded_count, failed_count, processed_count
    
    # Setup log file
    log_file = 'log.txt'
    
    # Clear previous log
    if os.path.exists(log_file):
        os.remove(log_file)
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create directories
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    log("="*60)
    log("Paper Download Manager - pll.py (Re-download Mode)")
    log("="*60)
    
    # Initialize CSV files if they don't exist
    if not os.path.exists(REQUIRED_CSV):
        log(f"ERROR: {REQUIRED_CSV} not found!")
        return
    
    if not os.path.exists(AVAILABLE_CSV):
        with open(AVAILABLE_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Title', 'DOI', 'Downloaded'])
        log(f"Created {AVAILABLE_CSV}")
    
    # Load and initialize all papers and todo
    all_papers = load_all_papers()
    todo_papers = initialize_todo_csv(all_papers)
    
    initial_success = sum(1 for p in todo_papers if p.get('Status') == 'success')
    pending_count = len(todo_papers) - initial_success
    
    log(f"Total papers: {len(todo_papers)}")
    log(f"Already re-downloaded: {initial_success}")
    log(f"Pending re-downloads: {pending_count}")
    log(f"Parallel workers: {MAX_WORKERS}")
    log(f"Download methods: 14 sources (Unpaywall, ArXiv, Semantic Scholar, OpenAlex, Europe PMC, PubMed, bioRxiv, CORE, DOAJ, BASE, LOC, Publisher, ResearchGate, Sci-Hub)")
    log(f"Output directory: {DOWNLOAD_DIR}/")
    log(f"Progress tracking: {TODO_CSV}")
    log(f"Log file: {log_file}")
    log("Press Ctrl+C to gracefully shutdown")
    log("="*60)
    log("")
    
    if pending_count == 0:
        log("No papers pending. All re-downloaded!")
        return
    
    # Filter pending papers (status != success)
    papers = [p for p in todo_papers if p.get('Status') != 'success']
    total_count = len(papers)
    
    # Process papers
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(process_paper, paper, total_count, idx): paper 
                for idx, paper in enumerate(papers, 1)
            }
            
            for future in as_completed(futures):
                if shutdown_event.is_set():
                    break
                try:
                    future.result()
                except Exception as e:
                    log(f"Task error: {e}")
    
    except Exception as e:
        log(f"Unexpected error: {e}")
    
    # Wait for active downloads to finish
    while active_downloads > 0:
        log(f"Waiting for {active_downloads} active downloads to complete...")
        time.sleep(1)
    
    # Final summary
    todo_papers = read_csv_safe(TODO_CSV)
    todo_papers = [normalize_fieldnames(p) for p in todo_papers]
    total_success = sum(1 for p in todo_papers if p.get('Status') == 'success')
    total_failed = sum(1 for p in todo_papers if p.get('Status') == 'failed')
    
    log("")
    log("="*60)
    log("RE-DOWNLOAD SUMMARY")
    log("="*60)
    log(f"Total papers: {len(todo_papers)}")
    log(f"Successfully re-downloaded (total): {total_success}")
    log(f"Failed (total): {total_failed}")
    log(f"New downloads this run: {downloaded_count}")
    log(f"New failures this run: {failed_count}")
    log(f"Processed this run: {processed_count}/{pending_count}")
    if len(todo_papers) > 0:
        success_rate = (total_success / len(todo_papers) * 100)
        log(f"Overall success rate: {success_rate:.1f}%")
    log(f"Downloads location: {DOWNLOAD_DIR}/")
    log(f"Progress: {TODO_CSV}")
    log(f"Available papers: {AVAILABLE_CSV}")
    log(f"Log saved to: {log_file}")
    log("="*60)


if __name__ == "__main__":
    main()
