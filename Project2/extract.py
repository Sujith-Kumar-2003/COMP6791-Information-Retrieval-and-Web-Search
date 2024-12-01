import os
import requests
from bs4 import BeautifulSoup
import argparse

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

def scrape_and_download(year_page_url, output_dir, max_files):
    """Scrape topic links, find PDFs, and download them."""
    os.makedirs(output_dir, exist_ok=True)
    downloaded_count = 0

    print(f"Scraping year page: {year_page_url}")
    topic_links = get_topic_links(year_page_url, max_files)
    print(f"Found {len(topic_links)} topic links.")

    for topic_url in topic_links:
        if downloaded_count >= max_files:
            print(f"Reached file download limit of {max_files}.")
            break

        print(f"Scraping topic page: {topic_url}")
        pdf_link = get_pdf_link(topic_url)

        if pdf_link:
            file_name = pdf_link.split('/')[-1]
            output_path = os.path.join(output_dir, file_name)

            if not os.path.exists(output_path):
                print(f"Downloading: {pdf_link}")
                try:
                    response = requests.get(pdf_link, stream=True)
                    if response.status_code == 200:
                        with open(output_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=1024):
                                f.write(chunk)
                        downloaded_count += 1
                        print(f"Saved: {output_path}")
                    else:
                        print(f"Failed to download {pdf_link}: {response.status_code}")
                except Exception as e:
                    print(f"Error downloading {pdf_link}: {e}")
            else:
                print(f"File already exists: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spectrum Web Crawler")
    parser.add_argument("--max_files", type=int, default=10, help="Maximum number of files to scrape and download (default: 80)")

    args = parser.parse_args()

    BASE_YEAR_URLS = [
        "https://spectrum.library.concordia.ca/view/document_subtype/thesis=5Fmasters/2024.html",
    ]
    output_directory = "Downloaded_PDFs"

    for i in BASE_YEAR_URLS:
        scrape_and_download(i, output_directory, args.max_files)