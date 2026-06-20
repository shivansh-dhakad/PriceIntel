from __future__ import annotations

import re
import random
import time
from dataclasses import dataclass, asdict
from urllib.parse import quote_plus, urljoin

import requests
from bs4 import BeautifulSoup

from services import settings


@dataclass
class PriceResult:
    platform: str
    title: str
    price_rs: int | None
    url: str
    image_url: str | None = None
    in_stock: bool | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ── Browser-realistic headers ──────────────────────────────────────────────

_CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)

_FLIPKART_HEADERS = {
    "User-Agent": _CHROME_UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Cache-Control": "max-age=0",
}

_AMAZON_HEADERS = {
    "User-Agent": _CHROME_UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-IN,en-GB;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Referer": "https://www.google.com/",
}


def parse_rupee_price(text: str | None) -> int | None:
    if not text:
        return None
    cleaned = text.replace(",", "")
    match = re.search(r"(?:Rs\.?|₹)\s*([0-9]+)", cleaned)
    if not match:
        match = re.search(r"\b([0-9]{4,7})\b", cleaned)
    return int(match.group(1)) if match else None


def fetch_html(url: str, headers: dict | None = None, retries: int = 2) -> str:
    h = headers or _FLIPKART_HEADERS
    for attempt in range(retries + 1):
        try:
            session = requests.Session()
            response = session.get(
                url,
                headers=h,
                timeout=settings.SCRAPER_TIMEOUT,
                allow_redirects=True,
            )
            response.raise_for_status()
            return response.text
        except Exception:
            if attempt == retries:
                raise
            time.sleep(1.5 * (attempt + 1))
    return ""


def image_from_tag(image) -> str | None:
    if image is None:
        return None
    for attr in ("src", "data-src", "data-old-hires"):
        value = image.get(attr)
        if value and not value.startswith("data:"):
            return normalize_image_url(value)
    srcset = image.get("srcset")
    if srcset:
        first = srcset.split(",")[0].strip().split(" ")[0]
        return normalize_image_url(first)
    return None


def normalize_image_url(url: str | None) -> str | None:
    if not url:
        return None
    if url.startswith("//"):
        return "https:" + url
    return url


def best_image_url(results: list[dict] | list[PriceResult]) -> tuple[str | None, str | None]:
    for item in results:
        image_url = item.image_url if isinstance(item, PriceResult) else item.get("image_url")
        platform = item.platform if isinstance(item, PriceResult) else item.get("platform")
        if image_url:
            return image_url, platform
    return None, None


# ── Flipkart scraper ───────────────────────────────────────────────────────

def search_flipkart(query: str, limit: int = 3, page: int = 1) -> list[PriceResult]:
    url = f"https://www.flipkart.com/search?q={quote_plus(query)}&page={page}"
    html = fetch_html(url, headers=_FLIPKART_HEADERS)
    soup = BeautifulSoup(html, "html.parser")
    results: list[PriceResult] = []

    # Flipkart product cards — anchor tags linking to product pages
    cards = soup.select("a[href*='/p/'], a[href*='pid=']")

    for card in cards:
        href = card.get("href", "")
        if not href:
            continue

        # ── Title: try multiple known Flipkart class names ──
        title_el = card.select_one(
            "div.RG5Slk, div.KzDlHZ, div.syl9yP, div._4rR01T, "
            "div.wjcEIp, div.col-7-12 div:first-child"
        )
        # Fallback: largest text block inside the card
        if not title_el:
            title_el = max(
                (t for t in card.select("div") if len(t.get_text(strip=True)) > 15),
                key=lambda t: len(t.get_text(strip=True)),
                default=None,
            )
        title = title_el.get_text(" ", strip=True) if title_el else ""

        # ── Price: try multiple known Flipkart price classes ──
        price_el = card.select_one(
            "div.Nx9bqj, div._30jeq3, div._25b18c, div.HL05au, "
            "div._1vC4OE, div._3I9_wc"
        )
        price_text = price_el.get_text(" ", strip=True) if price_el else ""

        # Fallback: scan all divs for ₹ symbol
        if not price_text:
            for div in card.select("div"):
                txt = div.get_text(strip=True)
                if "₹" in txt or "Rs" in txt:
                    price_text = txt
                    break

        price = parse_rupee_price(price_text)

        if not title or price is None:
            continue

        # ── Image ──
        image = card.select_one("img.UCc1lI, img.UCc1lI, img[src*='rukminim'], img[src*='flixcart']")
        if not image:
            image = card.select_one("img")

        results.append(
            PriceResult(
                platform="Flipkart",
                title=title[:240],
                price_rs=price,
                url=urljoin("https://www.flipkart.com", href),
                image_url=image_from_tag(image),
                in_stock=True,
            )
        )
        if len(results) >= limit:
            break

    return dedupe(results)


# ── Amazon scraper ─────────────────────────────────────────────────────────

def search_amazon(query: str, limit: int = 3, page: int = 1) -> list[PriceResult]:
    url = f"https://www.amazon.in/s?k={quote_plus(query)}&page={page}&i=computers"
    html = fetch_html(url, headers=_AMAZON_HEADERS)
    soup = BeautifulSoup(html, "html.parser")
    results: list[PriceResult] = []

    # If Amazon returned a bot-check page, html will be tiny
    if len(html) < 5000:
        raise RuntimeError(
            "Amazon returned a bot-protection page. "
            "Try again later or use a residential proxy."
        )

    # Primary selector — standard search result cards
    cards = soup.select("div[data-component-type='s-search-result']")
    # Fallback selectors
    if not cards:
        cards = soup.select("div[data-asin][data-index]")
    if not cards:
        cards = soup.select("div.s-result-item[data-asin]")

    for card in cards:
        title_el = card.select_one("h2 span, h2.a-size-mini span, span.a-size-medium")
        price_el = card.select_one(
            "span.a-price > span.a-offscreen, "
            "span.a-price-whole"
        )
        link_el = card.select_one("h2 a, a.a-link-normal[href*='/dp/']")

        title = title_el.get_text(" ", strip=True) if title_el else ""
        href = link_el.get("href", "") if link_el else ""
        price = parse_rupee_price(price_el.get_text(" ", strip=True) if price_el else "")

        if not title or not href or price is None:
            continue

        image = card.select_one("img.s-image, img[data-image-latency='s-product-image']")
        results.append(
            PriceResult(
                platform="Amazon",
                title=title[:240],
                price_rs=price,
                url=urljoin("https://www.amazon.in", href),
                image_url=image_from_tag(image),
                in_stock=True,
            )
        )
        if len(results) >= limit:
            break

    return dedupe(results)


# ── Combined search functions ──────────────────────────────────────────────

def search_live_prices(query: str, limit_per_platform: int = 3) -> dict:
    errors = []
    results: list[PriceResult] = []

    for searcher in (search_flipkart, search_amazon):
        try:
            results.extend(searcher(query, limit=limit_per_platform))
        except Exception as exc:
            platform = "Flipkart" if searcher is search_flipkart else "Amazon"
            errors.append(f"{platform}: {exc}")

    results = sorted(dedupe(results), key=lambda item: item.price_rs or 10**9)
    return {
        "query": query,
        "results": [item.to_dict() for item in results],
        "best_image_url": best_image_url(results)[0],
        "best_image_source": best_image_url(results)[1],
        "errors": errors,
    }


def search_marketplace_page(query: str, page: int = 1, limit_per_platform: int = 24) -> dict:
    errors = []
    results: list[PriceResult] = []

    for searcher in (search_flipkart, search_amazon):
        try:
            results.extend(searcher(query, limit=limit_per_platform, page=page))
        except Exception as exc:
            platform = "Flipkart" if searcher is search_flipkart else "Amazon"
            errors.append(f"{platform}: {exc}")

    return {
        "query": query,
        "page": page,
        "results": [item.to_dict() for item in dedupe(results)],
        "errors": errors,
    }


def dedupe(results: list[PriceResult]) -> list[PriceResult]:
    seen = set()
    unique = []
    for item in results:
        key = (item.platform, item.url.split("?")[0])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique
