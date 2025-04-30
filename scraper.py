import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin, urlparse

def load_from_cache(cache_filename="scraped_data.json"):
    """Load cached scraped data from a file, normalizing format"""
    try:
        with open(cache_filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Convert list (legacy) to dict if needed
            if isinstance(data, list):
                return {str(i): text for i, text in enumerate(data)}
            return data
    except FileNotFoundError:
        return None

def save_to_cache(data, cache_filename="scraped_data.json"):
    """Save scraped data to a cache file (JSON)"""
    with open(cache_filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def scrape_page(url, headers):
    """Scrape a single page and return its cleaned text + soup"""
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove non-informational elements
        for tag in soup(["script", "style", "header", "footer", "nav"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        return text, soup
    except Exception as e:
        print(f"⚠️ Failed to scrape {url}: {e}")
        return "", None

def is_valid(url, base_domain):
    """Check if a URL is internal and valid"""
    parsed = urlparse(url)
    return parsed.scheme in ["http", "https"] and (parsed.netloc == base_domain or parsed.netloc == "")

def scrape_website(start_url, max_pages=50, cache_filename="scraped_data.json"):
    """Crawl a website starting from start_url and cache the content"""
    cached_data = load_from_cache(cache_filename)
    if cached_data:
        print("🔄 Using cached data...")
        return "\n\n".join(cached_data.values())  # Safe now

    headers = {"User-Agent": "Mozilla/5.0"}
    visited = set()
    to_visit = [start_url]
    base_domain = urlparse(start_url).netloc
    scraped_data = {}

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop()
        if url in visited:
            continue
        visited.add(url)

        print(f"🔎 Scraping: {url}")
        text, soup = scrape_page(url, headers)
        if text:
            scraped_data[url] = text

        if soup:
            for link_tag in soup.find_all('a', href=True):
                link = urljoin(url, link_tag['href'])
                if is_valid(link, base_domain) and link not in visited and link not in to_visit:
                    to_visit.append(link)

    save_to_cache(scraped_data, cache_filename)
    return "\n\n".join(scraped_data.values())
