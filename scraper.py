import os
import time
import json
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import requests
from selectorlib import Extractor
import random
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
# ====== Config ======
BASE_URL = "https://www.amazon.com/s?k="
SAVE_DIR = "amazon_data"
CATEGORY_FILE = "./amazon_15_categories.json"
MAX_VALID_PRODUCTS = 10
SELECTOR_PATH = "selectors.yml"

# ====== Setup ======
def setup_driver():
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko)",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko)",
    ]
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("window-size=1920x1080")
    options.add_argument(f"user-agent={random.choice(user_agents)}")
    options.add_argument("--lang=en-US,en;q=0.9")
    options.add_argument("--log-level=3")  # mute logs
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)

def bypass_continue_shopping(driver):
    try:
        button = driver.find_element(By.XPATH, "//button[text()='Continue shopping']")
        print("[INFO] Detected 'Continue shopping' page. Clicking...")
        button.click()
        time.sleep(5)
        return True
    except NoSuchElementException:
        print("[INFO] No 'Continue shopping' page detected.")
        return False
def extract_product_links(search_url, max_links=30):
    driver = setup_driver()
    load_cookies(driver)
    with open("after_cookie_injection.html", "w", encoding="utf-8") as f:
        f.write(driver.page_source)
    driver.save_screenshot("after_cookie_injection.png")
    driver.get(search_url)
    time.sleep(5)
    bypass_continue_shopping(driver)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()

    product_cards = soup.select("div.s-result-item")
    urls = set()
    for card in product_cards:
        if len(urls) >= max_links:
            break
        a_tag = card.select_one("a.a-link-normal")
        if not a_tag:
            continue
        href = a_tag.get("href", "")
        if "dp/" in href or "gp/aw/d/" in href:
            full_url = "https://www.amazon.com" + href.split("/ref=")[0]
            urls.add(full_url)
    return list(urls)

def scrape_product_page(url, extractor):
    try:
        driver = setup_driver()
        load_cookies(driver)
        driver.get(url)
        time.sleep(5)
        html = driver.page_source
        driver.quit()
        return extractor.extract(html)
    except Exception as e:
        print(f"[ERROR] Failed to scrape {url} → {e}")
        return None


def is_valid_product(data):
    return bool(data and data.get("name") and data.get("short_description"))

def load_cookies(driver, cookies_path="cookies.json"):
    import json
    with open(cookies_path, "r", encoding="utf-8") as f:
        cookies = json.load(f)

    driver.get("https://www.amazon.com")  # 先初始化 Amazon 域
    time.sleep(2)

    for cookie in cookies:
        cookie.pop('sameSite', None)  # remove unsupported fields
        cookie.pop('storeId', None)
        cookie.pop('id', None)
        try:
            driver.add_cookie(cookie)
        except Exception as e:
            print(f"[WARN] Could not set cookie {cookie.get('name')}: {e}")

def process_subcategory(category, subcategory, extractor):
    print(f"[INFO] Processing → {category} / {subcategory}")
    search_url = BASE_URL + quote_plus(subcategory)
    urls = extract_product_links(search_url, max_links=100)

    folder = os.path.join(SAVE_DIR, category.replace(" ", "_"))
    os.makedirs(folder, exist_ok=True)
    output_path = os.path.join(folder, f"{subcategory.replace('/', '_')}.jsonl")

    count = 0
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for url in urls:
            if count >= MAX_VALID_PRODUCTS:
                break
            data = scrape_product_page(url, extractor)
            if is_valid_product(data):
                data["source_url"] = url
                json.dump(data, outfile)
                outfile.write("\n")
                count += 1
                print(f"  [✓] {count}: {data.get('name')[:60]}...")
            else:
                print(f"  [✗] Skipped invalid item")

    print(f"[✓] Saved {count} valid items to {output_path}")

def process_all():
    with open(CATEGORY_FILE, "r", encoding="utf-8") as f:
        categories = json.load(f)

    extractor = Extractor.from_yaml_file(SELECTOR_PATH)

    for cat in categories:
        category_name = cat.get("category")
        subcats = cat.get("subcategories", [])
        for sub in subcats:
            folder = os.path.join(SAVE_DIR, category_name.replace(" ", "_"))
            output_path = os.path.join(folder, f"{sub.replace('/', '_')}.jsonl")

            if os.path.exists(output_path):
                try:
                    if os.path.getsize(output_path) > 0:
                        print(f"[SKIP] {category_name}/{sub} already processed.")
                        continue
                except Exception as e:
                    print(f"[WARN] Could not check file {output_path}: {e}")

            process_subcategory(category_name, sub, extractor)

# ====== Start ======
if __name__ == "__main__":
    process_all()
