import csv
import os
import re
import requests
import time
from bs4 import BeautifulSoup
import json

def sanitize_filename(filename):
    """Remove invalid characters from a string to make it a valid filename."""
    return re.sub(r'[\\/\:\*\?"<>\|]', "", filename)

def download_paper(url, title, directory):
    """Download a paper from a URL and save it with a sanitized title."""
    try:
        response = requests.get(url, stream=True, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()  # Raise an exception for bad status codes

        # Clean the title to create a valid filename
        filename = sanitize_filename(title) + '.pdf'
        filepath = os.path.join(directory, filename)

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Successfully downloaded: {title}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"Failed to download {title}: {e}")
        return False

def get_unpaywall_url(doi):
    """Get a download URL from the Unpaywall API."""
    try:
        url = f"https://api.unpaywall.org/v2/{doi}?email=gemini-demo@google.com"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        if data.get('best_oa_location') and data['best_oa_location'].get('url_for_pdf'):
            return data['best_oa_location']['url_for_pdf']
    except requests.exceptions.RequestException as e:
        print(f"Unpaywall API request failed for {doi}: {e}")
    except json.JSONDecodeError:
        print(f"Failed to decode JSON from Unpaywall for {doi}")
    return None

def get_scihub_url(doi):
    """Get a download URL from Sci-Hub."""
    # Note: Sci-Hub domains change frequently. This is a placeholder domain.
    sci_hub_domains = [
        "https://sci-hub.se",
        "https://sci-hub.st",
        "https://sci-hub.ru",
    ]
    for domain in sci_hub_domains:
        try:
            url = f"{domain}/{doi}"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            pdf_url = soup.find('iframe', {'id': 'pdf'})
            if pdf_url:
                return pdf_url['src']
        except requests.exceptions.RequestException:
            continue
    return None

def get_google_scholar_url(doi):
    """Search Google Scholar for a PDF link."""
    try:
        url = f"https://scholar.google.com/scholar?q={doi}"
        response = requests.get(url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        pdf_link = soup.find('div', {'class': 'gs_or_ggsm'})
        if pdf_link and pdf_link.find('a'):
            return pdf_link.find('a')['href']
    except requests.exceptions.RequestException as e:
        print(f"Google Scholar search failed for {doi}: {e}")
    return None

def get_publisher_url(doi):
    """Try to find a PDF on the publisher's website."""
    try:
        url = f"https://doi.org/{doi}"
        response = requests.get(url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # This is a generic search for a link with "pdf" in the href
        for link in soup.find_all('a', href=True):
            if 'pdf' in link['href']:
                return link['href']
    except requests.exceptions.RequestException as e:
        print(f"Publisher website search failed for {doi}: {e}")
    return None

def get_arxiv_url(doi):
    """Search arXiv for a PDF link."""
    try:
        # Assuming the DOI is in the format of an arXiv ID
        arxiv_id = doi.split('/')[-1]
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        return url
    except Exception:
        return None

def get_core_url(doi):
    """Search CORE for a PDF link."""
    try:
        url = f"https://api.core.ac.uk/v3/search/works?q=doi:({doi})"
        response = requests.get(url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        data = response.json()
        if data.get('results') and data['results'][0].get('downloadUrl'):
            return data['results'][0]['downloadUrl']
    except requests.exceptions.RequestException as e:
        print(f"CORE API request failed for {doi}: {e}")
    except (json.JSONDecodeError, IndexError):
        print(f"Failed to get CORE URL for {doi}")
    return None


def main():
    """Read required.csv and download the papers."""
    required_file = 'required.csv'
    download_dir = 'pypapers'

    # Create the download directory if it doesn't exist
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    with open(required_file, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            title = row.get('Title', '')
            doi = row.get('DOI', '')

            if not (title and doi):
                continue

            print(f"Attempting to download: {title}")

            # Method 1: Unpaywall
            pdf_url = get_unpaywall_url(doi)
            if pdf_url and download_paper(pdf_url, title, download_dir):
                time.sleep(1)
                continue

            # Method 2: Sci-Hub
            pdf_url = get_scihub_url(doi)
            if pdf_url and download_paper(pdf_url, title, download_dir):
                time.sleep(5) # Be respectful of Sci-Hub's servers
                continue

            # Method 3: Google Scholar
            pdf_url = get_google_scholar_url(doi)
            if pdf_url and download_paper(pdf_url, title, download_dir):
                time.sleep(2)
                continue

            # Method 4: Publisher's website
            pdf_url = get_publisher_url(doi)
            if pdf_url and download_paper(pdf_url, title, download_dir):
                time.sleep(2)
                continue

            # Method 5: arXiv
            pdf_url = get_arxiv_url(doi)
            if pdf_url and download_paper(pdf_url, title, download_dir):
                time.sleep(1)
                continue
            
            # Method 6: CORE
            pdf_url = get_core_url(doi)
            if pdf_url and download_paper(pdf_url, title, download_dir):
                time.sleep(1)
                continue

            print(f"Could not find a downloadable PDF for: {title}")


if __name__ == "__main__":
    main()