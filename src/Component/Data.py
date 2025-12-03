import zipfile
from pathlib import Path
import requests
from bs4 import BeautifulSoup

PAGE_URL = "https://researchdata.gla.ac.uk/1658/"
# Store raw FAR-Trans data under src/Component/data/
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ZIP_PATH = OUT_DIR / "dataset.zip"
HEADERS = {"User-Agent": "Mozilla/5.0"}

def find_zip_url(page_url: str) -> str:
    r = requests.get(page_url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for a in soup.find_all("a", href=True):
        if a["href"].lower().endswith(".zip"):
            return requests.compat.urljoin(page_url, a["href"])
    raise RuntimeError("ZIP link not found on the page.")

def download_zip(zip_url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(zip_url, stream=True, headers=HEADERS, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)

def extract_zip(zip_file: Path, out_dir: Path, remove_zip: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_file, "r") as zf:
        zf.extractall(out_dir)
    if remove_zip:
        try:
            zip_file.unlink()
        except FileNotFoundError:
            pass

def main():
    zip_url = find_zip_url(PAGE_URL)
    download_zip(zip_url, ZIP_PATH)
    extract_zip(ZIP_PATH, OUT_DIR, remove_zip=False)

if __name__ == "__main__":
    main()
