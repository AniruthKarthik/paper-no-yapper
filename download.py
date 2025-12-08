import csv
import os
import re
import requests
import time
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin

MAX_RETRIES = 2
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
})

def sanitize_filename(filename):
    """Remove invalid characters from filename."""
    filename = re.sub(r'[\\/\:\*\?"<>\|]', "", filename)
    return filename[:200].strip()

def download_paper(url, title, directory):
    """Download paper from URL."""
    if not url:
        return False
        
    filepath = os.path.join(directory, sanitize_filename(title) + '.pdf')
    
    # Skip if already exists
    if os.path.exists(filepath):
        return True

    for attempt in range(MAX_RETRIES):
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
                        f.write(chunk)
                
                # Verify file size (basic validation)
                if os.path.getsize(filepath) > 1024:  # > 1KB
                    print(f"✓ Downloaded: {title[:50]}...")
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

def get_unpaywall_url(doi, email_id):
    """Get PDF URL from Unpaywall API."""
    try:
        url = f"https://api.unpaywall.org/v2/{doi}?email={email_id}"
        response = session.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('is_oa', False):
                # Try best location first
                best_location = data.get('best_oa_location')
                if best_location and best_location.get('url_for_pdf'):
                    return best_location['url_for_pdf']
                
                # Try any open access location
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
        # Check if DOI contains arxiv
        if 'arxiv' in doi.lower():
            # Extract arXiv ID
            arxiv_id = doi.split('/')[-1]
            # Remove version if present
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
            
            if data.get('response') and data['response'].get('docs') and len(data['response']['docs']) > 0 and data['response']['docs'][0].get('pdf_url'):
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
            
            if data.get('results') and len(data['results']) > 0 and data['results'][0].get('downloadUrl'):
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
            
            if data.get('results') and len(data['results']) > 0 and data['results'][0]['bibjson'].get('link'):
                for link in data['results'][0]['bibjson']['link']:
                    if link.get('type') == 'fulltext':
                        return link.get('url')
    except:
        pass
    return None

def get_scihub_url(doi):
    """Get PDF URL from Sci-Hub."""
    scihub_domains = [
        "https://sci-hub.se",
        "https://sci-hub.st"
    ]
    
    for domain in scihub_domains:
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

def load_progress(progress_file):
    """Load progress from file."""
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                content = f.read()
                for line in content.split('\n'):
                    if 'Last processed index:' in line:
                        return int(line.split(':')[-1].strip())
        except:
            pass
    return 0

def save_progress(progress_file, index, downloaded_count, total_count):
    """Save progress to file."""
    try:
        with open(progress_file, 'w') as f:
            f.write(f"Papers downloaded: {downloaded_count}\n")
            f.write(f"Total papers: {total_count}\n")
            f.write(f"Last processed index: {index}\n")
            if index > 0:
                f.write(f"Success rate: {(downloaded_count/index*100):.1f}%\n")
    except:
        pass

def main():
    """Main function to download papers."""
    required_file = 'allreq.csv'
    download_dir = 'reqs'

    # Generate 150 placeholder Gmail IDs
    gmail_ids = [f"gemini_researcher_{i:03d}@gmail.com" for i in range(1, 151)]
    current_email_index = 0
    papers_since_last_switch = 0
    PAPERS_PER_EMAIL = 70

    # Create download directory
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    progress_file = os.path.join(download_dir, 'progress.txt')
    failed_file = os.path.join(download_dir, 'failed.csv')
    
    # Load previous progress
    start_index = load_progress(progress_file)
    downloaded_count = 0
    failed_records = []

    # Count total papers
    try:
        with open(required_file, 'r', encoding='utf-8') as f:
            total_count = sum(1 for line in f) - 1  # Subtract header
    except:
        print(f"Error: Cannot read {required_file}")
        return

    print(f"Total papers to process: {total_count}")
    if start_index > 0:
        print(f"Resuming from index {start_index}")

    # Process papers
    try:
        with open(required_file, 'r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            
            for current_index, row in enumerate(reader):
                # Skip already processed
                if current_index < start_index:
                    continue
                
                title = row.get('Title', '').strip()
                doi = row.get('DOI', '').strip()

                if not title or not doi:
                    print(f"Skipping row {current_index} - missing data")
                    continue

                print(f"\n[{current_index + 1}/{total_count}] Processing: {title[:40]}...")

                # Try download methods in order
                download_successful = False
                
                # Get current email for Unpaywall
                current_email = gmail_ids[current_email_index]

                # Method 1: Unpaywall (fastest for open access)
                if not download_successful:
                    pdf_url = get_unpaywall_url(doi, current_email)
                    if pdf_url:
                        download_successful = download_paper(pdf_url, title, download_dir)
                        if download_successful:
                            print(f"  → Success via Unpaywall (Email: {current_email})")
                        time.sleep(0.5)
                
                # Method 2: ArXiv (fast for preprints)
                if not download_successful:
                    pdf_url = get_arxiv_url(doi)
                    if pdf_url:
                        download_successful = download_paper(pdf_url, title, download_dir)
                        if download_successful:
                            print(f"  → Success via ArXiv")
                        time.sleep(0.3)
                
                # Method 3: Semantic Scholar
                if not download_successful:
                    pdf_url = get_semantic_scholar_url(doi)
                    if pdf_url:
                        download_successful = download_paper(pdf_url, title, download_dir)
                        if download_successful:
                            print(f"  → Success via Semantic Scholar")
                        time.sleep(0.5)

                # Method 4: OpenAlex
                if not download_successful:
                    pdf_url = get_openalex_url(doi)
                    if pdf_url:
                        download_successful = download_paper(pdf_url, title, download_dir)
                        if download_successful:
                            print(f"  → Success via OpenAlex")
                        time.sleep(0.5)

                # Method 5: Library of Congress
                if not download_successful:
                    pdf_url = get_loc_url(doi)
                    if pdf_url:
                        download_successful = download_paper(pdf_url, title, download_dir)
                        if download_successful:
                            print(f"  → Success via Library of Congress")
                        time.sleep(0.5)

                # Method 6: BASE
                if not download_successful:
                    pdf_url = get_base_url(doi)
                    if pdf_url:
                        download_successful = download_paper(pdf_url, title, download_dir)
                        if download_successful:
                            print(f"  → Success via BASE")
                        time.sleep(0.5)

                # Method 7: CORE
                if not download_successful:
                    pdf_url = get_core_url(doi)
                    if pdf_url:
                        download_successful = download_paper(pdf_url, title, download_dir)
                        if download_successful:
                            print(f"  → Success via CORE")
                        time.sleep(0.5)

                # Method 8: DOAJ
                if not download_successful:
                    pdf_url = get_doaj_url(doi)
                    if pdf_url:
                        download_successful = download_paper(pdf_url, title, download_dir)
                        if download_successful:
                            print(f"  → Success via DOAJ")
                        time.sleep(0.5)

                # Method 9: Publisher website
                if not download_successful:
                    pdf_url = get_publisher_url(doi)
                    if pdf_url:
                        download_successful = download_paper(pdf_url, title, download_dir)
                        if download_successful:
                            print(f"  → Success via Publisher")
                        time.sleep(1.0)
                
                # Method 10: Sci-Hub (last resort)
                if not download_successful:
                    pdf_url = get_scihub_url(doi)
                    if pdf_url:
                        download_successful = download_paper(pdf_url, title, download_dir)
                        if download_successful:
                            print(f"  → Success via Sci-Hub")
                        time.sleep(2.0)  # Be respectful

                # Record result
                if download_successful:
                    downloaded_count += 1
                else:
                    print(f"  ✗ Failed to download")
                    failed_records.append({
                        'Index': current_index,
                        'Title': title,
                        'DOI': doi
                    })
                
                # Update email rotation
                papers_since_last_switch += 1
                if papers_since_last_switch >= PAPERS_PER_EMAIL:
                    current_email_index = (current_email_index + 1) % len(gmail_ids)
                    papers_since_last_switch = 0
                    print(f"\n--- Switched to email: {gmail_ids[current_email_index]} ---")
                
                # Save progress every paper
                save_progress(progress_file, current_index + 1, downloaded_count, total_count)
                
                # Small delay between papers
                time.sleep(0.2)

    except KeyboardInterrupt:
        print("\nDownload interrupted by user")
    except Exception as e:
        print(f"Error processing file: {e}")

    # Save failed downloads
    if failed_records:
        try:
            with open(failed_file, 'w', newline='', encoding='utf-8') as outfile:
                fieldnames = ['Index', 'Title', 'DOI']
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(failed_records)
            print(f"\nFailed downloads saved to: {failed_file}")
        except:
            print("Could not save failed downloads file")

    # Final summary
    print(f"\n" + "="*50)
    print(f"DOWNLOAD SUMMARY")
    print(f"="*50)
    print(f"Successfully downloaded: {downloaded_count}")
    print(f"Failed downloads: {len(failed_records)}")
    print(f"Total processed: {downloaded_count + len(failed_records)}")
    if total_count > 0:
        print(f"Success rate: {(downloaded_count/total_count*100):.1f}%")
    print(f"Downloads saved to: {download_dir}")

if __name__ == "__main__":
    main()
