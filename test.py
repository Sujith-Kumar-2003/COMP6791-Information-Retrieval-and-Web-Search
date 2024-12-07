import os
import tempfile
from unittest.mock import patch, Mock
import pytest
from extract import get_topic_links, get_pdf_link, scrape_and_download

# Test 1: Fetching Topic Links Successfully
def test_get_topic_links_success():
    mock_html = '''
    <html>
        <body>
            <a href="/id/eprint/12345/">Topic 1</a>
            <a href="/id/eprint/67890/">Topic 2</a>
        </body>
    </html>
    '''
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = mock_html

        year_url = "https://example.com/view/thesis=5Fmasters/2024.html"
        max_links = 5
        result = get_topic_links(year_url, max_links)

        assert len(result) == 2
        assert "https://spectrum.library.concordia.ca/id/eprint/12345/" in result
        assert "https://spectrum.library.concordia.ca/id/eprint/67890/" in result

# Test 2: Handling Empty Topic Links
def test_get_topic_links_empty():
    mock_html = '<html><body></body></html>'
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = mock_html

        year_url = "https://example.com/view/thesis=5Fmasters/2024.html"
        max_links = 5
        result = get_topic_links(year_url, max_links)

        assert len(result) == 0

# Test 3: Extracting PDF Link Successfully
def test_get_pdf_link_success():
    mock_html = '''
    <html>
        <body>
            <a class="ep_document_link" href="/downloads/thesis.pdf">Download PDF</a>
        </body>
    </html>
    '''
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = mock_html

        topic_url = "https://example.com/id/eprint/12345/"
        result = get_pdf_link(topic_url)

        assert result == "https://spectrum.library.concordia.ca/downloads/thesis.pdf"

# Test 4: Handling Missing PDF Links
def test_get_pdf_link_none():
    mock_html = '<html><body></body></html>'
    with patch('requests.get') as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = mock_html

        topic_url = "https://example.com/id/eprint/12345/"
        result = get_pdf_link(topic_url)

        assert result is None

# Test 5: Full Workflow - Scrape and Download PDFs
@patch('requests.get')
def test_scrape_and_download(mock_get):
    # Mock year page response
    mock_year_html = '''
    <html>
        <body>
            <a href="/id/eprint/12345/">Topic 1</a>
        </body>
    </html>
    '''
    # Mock topic page response
    mock_topic_html = '''
    <html>
        <body>
            <a class="ep_document_link" href="/downloads/thesis.pdf">Download PDF</a>
        </body>
    </html>
    '''
    # Mock PDF response
    mock_pdf_content = b"PDF content"

    # Setup mocks
    mock_get.side_effect = [
        Mock(status_code=200, content=mock_year_html),  # Year page
        Mock(status_code=200, content=mock_topic_html),  # Topic page
        Mock(status_code=200, content=mock_pdf_content),  # PDF file
    ]

    # Temporary directory for downloaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        year_page_url = "https://example.com/view/thesis=5Fmasters/2024.html"
        scrape_and_download(year_page_url, temp_dir, max_files=1)

        # Check file exists in the directory
        downloaded_files = os.listdir(temp_dir)
        assert len(downloaded_files) == 1
        assert downloaded_files[0].endswith(".pdf")
