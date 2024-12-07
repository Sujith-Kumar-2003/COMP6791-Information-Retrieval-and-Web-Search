import os
import requests
from bs4 import BeautifulSoup
import argparse
from urllib.parse import urljoin


def get_year_links(base_url):
    """Fetch year links from the base URL."""
    year_links = []
    try:
        response = requests.get(base_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            links = soup.find_all('a', href=True)  # Extract all <a> tags with href

            # Debug: Print all links extracted
            print(f"Extracted links from {base_url}: {[link['href'] for link in links]}")

            for link in links:
                href = link['href']
                # Filter for year links based on expected patterns
                if href.endswith('.html') and any(str(year) in href for year in range(2000, 2030)):
                    full_link = urljoin(base_url, href)  # Resolve relative URLs
                    year_links.append(full_link)
        else:
            print(f"Failed to fetch {base_url}: {response.status_code}")
    except Exception as e:
        print(f"Error fetching {base_url}: {e}")
    return year_links


def get_topic_links(year_page_url, max_links):
    """Fetch topic links from the year page, limited by max_links."""
    topic_links = []
    try:
        response = requests.get(year_page_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            links = soup.find_all('a', href=True)
            for link in links:
                if len(topic_links) >= max_links:
                    break
                href = link['href']
                if 'id/eprint' in href:
                    full_link = href if href.startswith("http") else f"https://spectrum.library.concordia.ca{href}"
                    topic_links.append(full_link)
        else:
            print(f"Failed to fetch {year_page_url}: {response.status_code}")
    except Exception as e:
        print(f"Error fetching {year_page_url}: {e}")
    return topic_links

def get_pdf_link(topic_url):
    """Extract the PDF link from a topic page."""
    try:
        response = requests.get(topic_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            link = soup.find('a', class_='ep_document_link', href=lambda x: x and x.endswith('.pdf'))
            if link:
                href = link['href']
                pdf_url = href if href.startswith("http") else f"https://spectrum.library.concordia.ca{href}"
                if "HowtoPrepareYourThesisForDepositinSpectrum.pdf" not in pdf_url:
                    return pdf_url
        else:
            print(f"Failed to fetch {topic_url}: {response.status_code}")
    except Exception as e:
        print(f"Error fetching {topic_url}: {e}")
    return None

def scrape_and_download(base_url, faculty_name, output_dir, max_files):
    """Scrape year links, topic links, find PDFs, and download them into faculty-specific folder."""
    faculty_folder = os.path.join(output_dir, faculty_name)
    os.makedirs(faculty_folder, exist_ok=True)  # Create faculty folder

    total_downloaded = 0
    print(f"Scraping base URL: {base_url}")
    year_links = get_year_links(base_url)
    print(f"Found {len(year_links)} year links.")

    for year_url in year_links:
        print(f"Scraping year page: {year_url}")
        topic_links = get_topic_links(year_url, max_files)

        print(f"Found {len(topic_links)} topic links on {year_url}.")

        for topic_url in topic_links:
            if total_downloaded >= max_files:
                print(f"Reached file download limit of {max_files}.")
                return

            print(f"Scraping topic page: {topic_url}")
            pdf_link = get_pdf_link(topic_url)

            if pdf_link:
                file_name = pdf_link.split('/')[-1]
                output_path = os.path.join(faculty_folder, file_name)

                if not os.path.exists(output_path):
                    print(f"Downloading: {pdf_link}")
                    try:
                        response = requests.get(pdf_link, stream=True)
                        if response.status_code == 200:
                            with open(output_path, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=1024):
                                    f.write(chunk)
                            total_downloaded += 1
                            print(f"Saved: {output_path}")
                        else:
                            print(f"Failed to download {pdf_link}: {response.status_code}")
                    except Exception as e:
                        print(f"Error downloading {pdf_link}: {e}")
                else:
                    print(f"File already exists: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spectrum Web Crawler")
    parser.add_argument("--max_files", type=int, default=5, help="Maximum number of files to scrape and download (default: 50)")

    args = parser.parse_args()

    BASE_YEAR_URLS = {
        "Concordia University Press": "https://spectrum.library.concordia.ca/view/divisions/concordiauniversitypress/",
        "Arts & Science": "https://spectrum.library.concordia.ca/view/divisions/fac=5Fartsscience/",
        "Fine Arts": "https://spectrum.library.concordia.ca/view/divisions/fac=5Ffinearts/",
        "Engineering": "https://spectrum.library.concordia.ca/view/divisions/fac=5Feng/",
        "JMSB": "https://spectrum.library.concordia.ca/view/divisions/fac=5Fjmsb/",
        "Libraries": "https://spectrum.library.concordia.ca/view/divisions/libraries/",
        "Research Centres": "https://spectrum.library.concordia.ca/view/divisions/centres=5Finst/",
        "Graduate Studies": "https://spectrum.library.concordia.ca/view/divisions/fac=5Fsgs/"
    }

    output_directory = "Downloaded_PDFs"

    for faculty_name, base_url in BASE_YEAR_URLS.items():
        scrape_and_download(base_url, faculty_name, output_directory, args.max_files)
