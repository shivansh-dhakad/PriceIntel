from __future__ import annotations

import os
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from recommendor import build_feature_matrix, get_filter_options, get_similar_laptops, load_and_clean
from services import settings, supabase_store
from services.groq_analysis import analyze_laptop
from services.price_scraper import search_live_prices


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "cleaned_data.csv"
MODEL_PATH = BASE_DIR / "laptop_model.pkl"

app = Flask(__name__)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


def money(value: float | int | str | None) -> str:
    try:
        return f"Rs {float(value):,.0f}"
    except (TypeError, ValueError):
        return "Price unavailable"


def clean_scalar(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def empty_filter_options() -> dict:
    return {
        "brands": [],
        "os": [],
        "gpu_tiers": [],
        "ram_options": [],
        "ssd_options": [],
        "use_cases": ["gaming", "editing", "student", "office", "any"],
        "price_min": 0,
        "price_max": 0,
    }


def source_key_index(source_key: str | None, fallback: int) -> int:
    if source_key and source_key.startswith("csv-"):
        try:
            return int(source_key.removeprefix("csv-"))
        except ValueError:
            pass
    return fallback


def load_from_supabase() -> pd.DataFrame:
    rows = []
    for fallback_id, laptop in enumerate(supabase_store.list_laptops()):
        specs = dict(laptop.get("raw_specs") or {})
        specs.update(
            {
                "id": source_key_index(laptop.get("source_key"), fallback_id),
                "source_key": laptop.get("source_key"),
                "Brand": laptop.get("brand") or specs.get("Brand"),
                "Model": laptop.get("model") or specs.get("Model"),
                "Price (Rs)": laptop.get("price_rs") or specs.get("Price (Rs)"),
                "RAM_GB": laptop.get("ram_gb") or specs.get("RAM_GB"),
                "SSD_GB": laptop.get("ssd_gb") or specs.get("SSD_GB"),
                "GPU_Tier": laptop.get("gpu_tier") or specs.get("GPU_Tier"),
                "CPU_Category": laptop.get("cpu_category") or specs.get("CPU_Category"),
                "CPU_Brand": laptop.get("cpu_brand") or specs.get("CPU_Brand"),
                "GPU_Brand": laptop.get("gpu_brand") or specs.get("GPU_Brand"),
                "Weight": laptop.get("weight_kg") or specs.get("Weight"),
                "Display Size": laptop.get("display_size") or specs.get("Display Size"),
                "Operating System": laptop.get("os") or specs.get("Operating System"),
                "Display Touchscreen": laptop.get("touchscreen") or specs.get("Display Touchscreen"),
                "image_url": laptop.get("image_url") or specs.get("image_url"),
                "image_source": laptop.get("image_source") or specs.get("image_source"),
            }
        )
        rows.append(specs)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["Price (Rs)"] = pd.to_numeric(df["Price (Rs)"], errors="coerce")
    df = df.dropna(subset=["Price (Rs)"]).reset_index(drop=True)
    for column in ("RAM_GB", "SSD_GB"):
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)
    return df


def load_laptop_dataframe() -> pd.DataFrame:
    if settings.USE_SUPABASE:
        if not supabase_store.is_configured():
            raise RuntimeError("Supabase is enabled but not configured.")
        return load_from_supabase()
    return load_and_clean(str(DATA_PATH))


@lru_cache(maxsize=1)
def laptop_store():
    df = load_laptop_dataframe()
    df = df.reset_index(drop=True)
    if "id" not in df.columns:
        df["id"] = df.index
    if df.empty:
        return df, None, empty_filter_options()
    _, feature_matrix, _ = build_feature_matrix(df)
    options = get_filter_options(df)
    return df, feature_matrix, options


@lru_cache(maxsize=1)
def price_model():
    with MODEL_PATH.open("rb") as handle:
        return pickle.load(handle)


def prediction_options(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    fields = [
        "Brand",
        "Operating System",
        "Capacity",
        "RAM Type",
        "SSD Capacity",
        "GPU_Brand",
        "GPU_Tier",
        "CPU_Brand",
        "CPU_Category",
    ]
    options = {field: sorted(df[field].dropna().astype(str).unique().tolist()) for field in fields}
    options["Display Size"] = sorted(df["Display Size"].dropna().astype(float).unique().tolist())
    options["Resolution"] = sorted(
        {
            f"{int(row.Resolution_X)} x {int(row.Resolution_Y)}"
            for row in df[["Resolution_X", "Resolution_Y"]].dropna().itertuples(index=False)
        }
    )
    return options


def first_option(df: pd.DataFrame, column: str, fallback):
    if column not in df:
        return fallback
    value = df[column].dropna().mode()
    if value.empty:
        return fallback
    return clean_scalar(value.iloc[0])


def parse_resolution(value: str | None) -> tuple[int, int]:
    if not value:
        return 1920, 1080
    normalized = value.lower().replace("*", "x")
    parts = [part.strip() for part in normalized.split("x")]
    if len(parts) != 2:
        return 1920, 1080
    try:
        return int(float(parts[0])), int(float(parts[1]))
    except ValueError:
        return 1920, 1080


def numeric_arg(args, key: str, fallback: float) -> float:
    try:
        value = args.get(key)
        if value in (None, ""):
            return fallback
        return float(value)
    except (TypeError, ValueError):
        return fallback


def build_prediction_input(args) -> pd.DataFrame:
    df, _, _ = laptop_store()
    resolution_x, resolution_y = parse_resolution(args.get("resolution"))
    display_size = numeric_arg(args, "display_size", 15.6)
    ppi = numeric_arg(
        args,
        "pixel_density",
        ((resolution_x**2 + resolution_y**2) ** 0.5) / display_size if display_size else 141,
    )

    row = {
        "Brand": args.get("brand") or first_option(df, "Brand", "HP"),
        "Weight": numeric_arg(args, "weight", 1.7),
        "Operating System": args.get("os") or first_option(df, "Operating System", "Windows 11"),
        "Display Size": display_size,
        "Pixel Density": ppi,
        "Clock-speed": numeric_arg(args, "clock_speed", 4.0),
        "Capacity": args.get("capacity") or first_option(df, "Capacity", "16 GB"),
        "RAM Type": args.get("ram_type") or first_option(df, "RAM Type", "DDR4"),
        "SSD Capacity": args.get("ssd_capacity") or first_option(df, "SSD Capacity", "512 GB"),
        "Warranty": numeric_arg(args, "warranty", 1),
        "Resolution_X": resolution_x,
        "Resolution_Y": resolution_y,
        "GPU_Brand": args.get("gpu_brand") or first_option(df, "GPU_Brand", "Intel"),
        "GPU_Tier": args.get("gpu_tier") or first_option(df, "GPU_Tier", "Integrated"),
        "CPU_Brand": args.get("cpu_brand") or first_option(df, "CPU_Brand", "Intel"),
        "CPU_Category": args.get("cpu_category") or first_option(df, "CPU_Category", "i5-1Gen"),
    }
    return pd.DataFrame([row])


def comparable_price_range(specs: pd.Series, predicted: float) -> dict:
    df, _, _ = laptop_store()
    if df.empty:
        return {
            "low": round(predicted * 0.88, -2),
            "high": round(predicted * 1.12, -2),
            "average": round(predicted, -2),
            "sample_size": 0,
        }
    matches = df.copy()
    for column in ("Brand", "CPU_Brand", "GPU_Tier", "Capacity", "SSD Capacity"):
        matches = matches[matches[column].astype(str) == str(specs[column])]

    if len(matches) < 5:
        matches = df[
            (df["CPU_Brand"].astype(str) == str(specs["CPU_Brand"]))
            & (df["GPU_Tier"].astype(str) == str(specs["GPU_Tier"]))
            & (df["Capacity"].astype(str) == str(specs["Capacity"]))
            & (df["SSD Capacity"].astype(str) == str(specs["SSD Capacity"]))
        ]

    if len(matches) < 5:
        matches = df[
            (df["GPU_Tier"].astype(str) == str(specs["GPU_Tier"]))
            & (df["Capacity"].astype(str) == str(specs["Capacity"]))
        ]

    prices = pd.to_numeric(matches["Price (Rs)"], errors="coerce").dropna()
    if len(prices) >= 3:
        low = float(prices.quantile(0.25))
        high = float(prices.quantile(0.75))
        average = float(prices.mean())
        sample_size = int(len(prices))
    else:
        low = predicted * 0.88
        high = predicted * 1.12
        average = predicted
        sample_size = int(len(prices))

    low = min(low, predicted * 0.94)
    high = max(high, predicted * 1.06)
    return {
        "low": round(low, -2),
        "high": round(high, -2),
        "average": round(average, -2),
        "sample_size": sample_size,
    }


def laptop_payload(row: pd.Series, include_details: bool = False) -> dict:
    price = clean_scalar(row.get("Price (Rs)"))
    ram = clean_scalar(row.get("RAM_GB"))
    ssd = clean_scalar(row.get("SSD_GB"))
    weight = clean_scalar(row.get("Weight"))
    gpu_tier = clean_scalar(row.get("GPU_Tier")) or "Unknown"
    cpu_category = clean_scalar(row.get("CPU_Category")) or "Unknown"
    source_key = clean_scalar(row.get("source_key")) or f"csv-{int(row['id'])}"
    stored_laptop = (
        supabase_store.get_laptop_by_source_key(source_key)
        if not clean_scalar(row.get("image_url"))
        else None
    )

    usage = []
    if gpu_tier in {"High", "Mid", "Entry Gaming"}:
        usage.extend(["Gaming", "Video editing"])
    if ram and ram >= 16:
        usage.extend(["Coding", "Multitasking"])
    if weight and weight <= 1.5:
        usage.append("Travel")
    if not usage:
        usage.append("Everyday work")

    payload = {
        "id": int(row["id"]),
        "source_key": source_key,
        "brand": clean_scalar(row.get("Brand")) or "Unknown",
        "model": clean_scalar(row.get("Model")) or "Unnamed laptop",
        "image_url": clean_scalar(row.get("image_url")) or (stored_laptop or {}).get("image_url"),
        "image_source": clean_scalar(row.get("image_source")) or (stored_laptop or {}).get("image_source"),
        "price": price,
        "price_label": money(price),
        "ram_gb": ram,
        "ssd_gb": ssd,
        "gpu_tier": gpu_tier,
        "cpu_category": cpu_category,
        "cpu_brand": clean_scalar(row.get("CPU_Brand")) or "Unknown",
        "gpu_brand": clean_scalar(row.get("GPU_Brand")) or "Unknown",
        "weight": weight,
        "display_size": clean_scalar(row.get("Display Size")),
        "os": clean_scalar(row.get("Operating System")) or "Unknown OS",
        "touchscreen": clean_scalar(row.get("Display Touchscreen")) or "Unknown",
        "screen_category": clean_scalar(row.get("Screen_Category")) or "Unknown",
        "ppi": clean_scalar(row.get("Pixel Density")),
        "value_score": score_laptop(row),
        "best_for": list(dict.fromkeys(usage))[:3],
        "live_prices": [],
    }

    if include_details:
        payload["details"] = {
            "RAM": clean_scalar(row.get("Capacity")),
            "RAM type": clean_scalar(row.get("RAM Type")),
            "SSD": clean_scalar(row.get("SSD Capacity")),
            "Battery": clean_scalar(row.get("Battery Type")),
            "Wi-Fi": clean_scalar(row.get("Wi-Fi Version")),
            "Warranty": clean_scalar(row.get("Warranty")),
            "Resolution": resolution_label(row),
            "Clock speed": clean_scalar(row.get("Clock-speed")),
        }
        payload["insights"] = build_insights(row)
        payload["live_prices"] = cached_live_prices(payload)
        payload["ai_enabled"] = bool(settings.GROQ_API_KEY)

    return payload


def laptop_query(laptop: dict) -> str:
    return f"{laptop.get('brand', '')} {laptop.get('model', '')}".strip()


def laptop_key(laptop: dict) -> str:
    return laptop_query(laptop).lower().replace(" ", "-")[:120]


def cached_live_prices(laptop: dict) -> list[dict]:
    query = laptop_query(laptop)
    cached = supabase_store.get_latest_prices(query, limit=6)
    return [format_price_record(item) for item in cached]


def format_price_record(item: dict) -> dict:
    return {
        "platform": item.get("platform"),
        "title": item.get("title"),
        "price_rs": item.get("price_rs"),
        "price_label": money(item.get("price_rs")),
        "url": item.get("url"),
        "image_url": item.get("image_url"),
        "in_stock": item.get("in_stock"),
        "scraped_at": item.get("scraped_at"),
    }


def resolution_label(row: pd.Series) -> str:
    x = clean_scalar(row.get("Resolution_X"))
    y = clean_scalar(row.get("Resolution_Y"))
    if not x or not y:
        return "Unknown"
    return f"{int(x)} x {int(y)}"


def score_laptop(row: pd.Series) -> int:
    price = float(row.get("Price (Rs)") or 0)
    ram = float(row.get("RAM_GB") or 0)
    ssd = float(row.get("SSD_GB") or 0)
    weight = float(row.get("Weight") or 3)
    gpu_tier = row.get("GPU_Tier")

    score = 48
    score += min(ram, 32) * 0.9
    score += min(ssd, 1000) / 70
    score += {"High": 18, "Mid": 13, "Entry Gaming": 9, "Basic Dedicated": 6, "Integrated": 2}.get(gpu_tier, 4)
    score += max(0, 2.2 - weight) * 5
    if price:
        score -= max(0, price - 70000) / 9000
    return int(max(35, min(98, round(score))))


def build_insights(row: pd.Series) -> dict:
    ram = float(row.get("RAM_GB") or 0)
    ssd = float(row.get("SSD_GB") or 0)
    weight = float(row.get("Weight") or 0)
    gpu = row.get("GPU_Tier")
    display = float(row.get("Display Size") or 0)

    pros = []
    cons = []
    if ram >= 16:
        pros.append("Comfortable memory for coding and multitasking")
    else:
        cons.append("RAM may feel tight for heavy multitasking")
    if ssd >= 512:
        pros.append("Good SSD capacity for apps and project files")
    else:
        cons.append("Storage may need external backup sooner")
    if gpu in {"High", "Mid"}:
        pros.append("Dedicated graphics tier suits games and creative work")
    elif gpu == "Integrated":
        cons.append("Not ideal for modern AAA gaming")
    if weight and weight <= 1.5:
        pros.append("Easy to carry daily")
    elif weight and weight >= 2.3:
        cons.append("Heavy for frequent travel")
    if display and display >= 16:
        pros.append("Large display helps with editing and split-screen work")

    best_for = []
    if gpu in {"High", "Mid", "Entry Gaming"}:
        best_for.append("Gaming")
    if ram >= 16:
        best_for.append("Programming")
    if gpu in {"High", "Mid"} and ram >= 16:
        best_for.append("Video editing")
    if weight and weight <= 1.5:
        best_for.append("Students and travel")
    if not best_for:
        best_for.append("Office and everyday browsing")

    not_for = []
    if gpu == "Integrated":
        not_for.append("High-end gaming")
    if weight and weight >= 2.3:
        not_for.append("Frequent travelers")
    if ram < 16:
        not_for.append("Large ML or editing workloads")

    return {
        "pros": pros[:3],
        "cons": cons[:3],
        "best_for": best_for[:3],
        "not_for": not_for[:3] or ["Users needing a very specific niche configuration"],
    }


def filtered_laptops(args):
    df, _, _ = laptop_store()
    if df.empty:
        return df
    result = df.copy()

    q = (args.get("q") or "").strip().lower()
    if q:
        result = result[
            result["Brand"].astype(str).str.lower().str.contains(q, na=False)
            | result["Model"].astype(str).str.lower().str.contains(q, na=False)
            | result["CPU_Category"].astype(str).str.lower().str.contains(q, na=False)
            | result["GPU_Brand"].astype(str).str.lower().str.contains(q, na=False)
        ]

    min_price = args.get("min_price", type=float)
    max_price = args.get("max_price", type=float)
    if min_price is not None:
        result = result[result["Price (Rs)"] >= min_price]
    if max_price is not None:
        result = result[result["Price (Rs)"] <= max_price]

    for key, column in (("brand", "Brand"), ("os", "Operating System")):
        value = args.get(key)
        if value and value.lower() not in {"any", "all"}:
            result = result[result[column].astype(str).str.lower() == value.lower()]

    use_case = args.get("use_case")
    if use_case and use_case != "any":
        if use_case == "gaming":
            result = result[result["GPU_Tier"].isin(["High", "Mid", "Entry Gaming"])]
        elif use_case == "editing":
            result = result[result["GPU_Tier"].isin(["High", "Mid", "Basic Dedicated"])]
        elif use_case == "student":
            result = result[(result["Weight"] <= 1.8) & (result["Price (Rs)"] <= 80000)]
        elif use_case == "office":
            result = result[result["Price (Rs)"] <= 90000]

    min_ram = args.get("min_ram", type=int)
    min_ssd = args.get("min_ssd", type=int)
    max_weight = args.get("max_weight", type=float)
    if min_ram is not None:
        result = result[result["RAM_GB"] >= min_ram]
    if min_ssd is not None:
        result = result[result["SSD_GB"] >= min_ssd]
    if max_weight is not None:
        result = result[result["Weight"] <= max_weight]

    touchscreen = args.get("touchscreen")
    if touchscreen in {"Yes", "No"}:
        result = result[result["Display Touchscreen"] == touchscreen]

    sort_by = args.get("sort", "value")
    if sort_by == "price_desc":
        result = result.sort_values("Price (Rs)", ascending=False)
    elif sort_by == "ram":
        result = result.sort_values(["RAM_GB", "SSD_GB"], ascending=False)
    elif sort_by == "portable":
        result = result.sort_values(["Weight", "Price (Rs)"], ascending=[True, True])
    elif sort_by == "value":
        result = result.assign(_score=result.apply(score_laptop, axis=1)).sort_values("_score", ascending=False)
    else:
        result = result.sort_values("Price (Rs)", ascending=True)

    limit = min(args.get("limit", default=24, type=int), 80)
    return result.head(limit)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/filters")
def api_filters():
    _, _, options = laptop_store()
    return jsonify(options)


@app.route("/api/prediction-options")
def api_prediction_options():
    df, _, _ = laptop_store()
    return jsonify(prediction_options(df))


@app.route("/api/predict-price")
def api_predict_price():
    specs = build_prediction_input(request.args)
    try:
        predicted = float(price_model().predict(specs)[0])
    except Exception as exc:
        return jsonify({"error": f"Could not predict price: {exc}"}), 500

    predicted = max(0, predicted)
    range_data = comparable_price_range(specs.iloc[0], predicted)
    payload = {
        "predicted_price": round(predicted, -2),
        "predicted_label": money(predicted),
        "range_low": range_data["low"],
        "range_high": range_data["high"],
        "range_label": f"{money(range_data['low'])} - {money(range_data['high'])}",
        "average_price": range_data["average"],
        "average_label": money(range_data["average"]),
        "sample_size": range_data["sample_size"],
        "specs": {key: clean_scalar(value) for key, value in specs.iloc[0].to_dict().items()},
    }
    return jsonify(payload)


@app.route("/api/search")
def api_search():
    result = filtered_laptops(request.args)
    return jsonify([laptop_payload(row) for _, row in result.iterrows()])


@app.route("/api/laptops/<int:laptop_id>")
def api_laptop(laptop_id: int):
    df, _, _ = laptop_store()
    if df.empty:
        return jsonify({"error": "No laptops are available in Supabase. Run scripts/sync_csv_to_supabase.py first."}), 503
    match = df[df["id"] == laptop_id]
    if match.empty:
        return jsonify({"error": "Laptop not found"}), 404
    payload = laptop_payload(match.iloc[0], include_details=True)
    saved_analysis = supabase_store.get_analysis(laptop_key(payload))
    if saved_analysis:
        payload["insights"] = saved_analysis.get("analysis", payload["insights"])
        payload["ai_cached"] = True
    return jsonify(payload)


@app.route("/api/similar/<int:laptop_id>")
def api_similar(laptop_id: int):
    df, feature_matrix, _ = laptop_store()
    if df.empty or feature_matrix is None:
        return jsonify({"error": "No laptops are available in Supabase. Run scripts/sync_csv_to_supabase.py first."}), 503
    result = get_similar_laptops(df, feature_matrix, laptop_index=laptop_id, top_n=5)
    if result["error"]:
        return jsonify({"error": result["error"]}), 404

    similar = []
    for _, row in result["similar"].iterrows():
        match = df[(df["Brand"] == row["Brand"]) & (df["Model"] == row["Model"])].head(1)
        if not match.empty:
            similar.append(laptop_payload(match.iloc[0]))
    return jsonify(similar)


@app.route("/api/compare")
def api_compare():
    ids = [int(item) for item in request.args.get("ids", "").split(",") if item.isdigit()]
    df, _, _ = laptop_store()
    if df.empty:
        return jsonify([])
    rows = df[df["id"].isin(ids)].head(2)
    return jsonify([laptop_payload(row, include_details=True) for _, row in rows.iterrows()])


@app.route("/api/assistant")
def api_assistant():
    args = request.args.copy()
    budget = args.get("budget", type=float)
    if budget:
        args = args.copy()
        args["max_price"] = str(budget)
    result = filtered_laptops(args).head(6)
    return jsonify([laptop_payload(row) for _, row in result.iterrows()])


@app.route("/api/market")
def api_market():
    """Return stored Amazon/Flipkart listings from Supabase."""
    query = (request.args.get("q") or "").strip()
    platform = (request.args.get("platform") or "").strip().lower()
    page = max(1, request.args.get("page", default=1, type=int))

    rows = supabase_store.list_market_prices(
        query=query,
        platform=platform,
        page=page,
        page_size=48,
    )

    for item in rows:
        if item.get("price_rs"):
            item["price_label"] = money(item["price_rs"])
        else:
            item["price_label"] = "Price unavailable"

    return jsonify({
        "query": query,
        "platform": platform,
        "page": page,
        "results": rows,
        "stored_in_supabase": supabase_store.is_configured(),
    })


@app.route("/api/market/stats")
def api_market_stats():
    """Return aggregate stats for the stored marketplace catalog."""
    stats = supabase_store.get_market_price_stats()
    return jsonify(stats)


@app.route("/api/market/scrape", methods=["POST"])
def api_market_scrape():
    """Scrape Amazon & Flipkart for a query and save results to Supabase."""
    import hashlib
    import re
    from urllib.parse import urlsplit, urlunsplit
    from services.price_scraper import search_marketplace_page

    data = request.get_json(silent=True) or {}
    query = (data.get("query") or request.args.get("q") or "laptop").strip()
    pages = min(max(1, int(data.get("pages", 1))), 5)
    limit_per = min(max(1, int(data.get("limit_per_platform", 24))), 24)

    BRANDS = [
        "HP", "Dell", "Lenovo", "Asus", "Acer", "Apple", "MSI",
        "Samsung", "Microsoft", "LG", "Infinix", "Honor", "Xiaomi",
        "Realme", "Avita", "Gigabyte", "Zebronics", "Fujitsu",
    ]

    def canonical_url(url: str) -> str:
        parts = urlsplit(url)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))

    def make_source_key(platform: str, url: str) -> str:
        digest = hashlib.sha1(canonical_url(url).encode()).hexdigest()[:20]
        return f"marketplace-{platform.lower()}-{digest}"

    def parse_brand(title: str) -> str | None:
        normalized = title.lower()
        for brand in BRANDS:
            if re.search(rf"\b{re.escape(brand.lower())}\b", normalized):
                return brand
        return None

    if not supabase_store.is_configured():
        return jsonify({"error": "Supabase is not configured."}), 503

    total_saved = 0
    all_errors: list[str] = []

    for page in range(1, pages + 1):
        live = search_marketplace_page(query, page=page, limit_per_platform=limit_per)
        if live.get("errors"):
            all_errors.extend(live["errors"])

        for item in live.get("results", []):
            if not item.get("url") or not item.get("title"):
                continue

            platform = item.get("platform", "Marketplace")
            brand = parse_brand(item["title"])
            src_key = make_source_key(platform, item["url"])

            # Save as a laptop record (shows in Discover view)
            laptop = supabase_store.upsert_laptop({
                "source_key": src_key,
                "brand": brand,
                "model": item["title"][:240],
                "price_rs": item.get("price_rs"),
                "image_url": item.get("image_url"),
                "image_source": platform,
                "raw_specs": {
                    "Brand": brand,
                    "Model": item["title"],
                    "Price (Rs)": item.get("price_rs"),
                    "image_url": item.get("image_url"),
                    "image_source": platform,
                    "marketplace": platform,
                    "product_url": item["url"],
                    "search_query": query,
                },
            })

            # Save price record (shows in Market view)
            supabase_store.upsert_price(
                query=query,
                platform=platform,
                title=item["title"],
                price_rs=item.get("price_rs"),
                url=item["url"],
                image_url=item.get("image_url"),
                in_stock=item.get("in_stock"),
                laptop_id=(laptop or {}).get("id"),
            )
            total_saved += 1

    stats = supabase_store.get_market_price_stats()
    return jsonify({
        "query": query,
        "pages_scraped": pages,
        "saved": total_saved,
        "errors": all_errors,
        "stats": stats,
    })


@app.route("/api/live-price")
def api_live_price():
    query = (request.args.get("q") or "").strip()
    if len(query) < 2:
        return jsonify({"error": "Enter a laptop model or search term."}), 400

    live = search_live_prices(query)
    stored = []
    for item in live["results"]:
        saved = supabase_store.upsert_price(
            query=query,
            platform=item["platform"],
            title=item["title"],
            price_rs=item["price_rs"],
            url=item["url"],
            image_url=item.get("image_url"),
            in_stock=item.get("in_stock"),
        )
        stored.append(format_price_record(saved) if saved else {**item, "price_label": money(item.get("price_rs"))})

    return jsonify(
        {
            "query": query,
            "results": stored,
            "best_image_url": live.get("best_image_url"),
            "best_image_source": live.get("best_image_source"),
            "errors": live["errors"],
            "stored_in_supabase": supabase_store.is_configured(),
        }
    )


@app.route("/api/laptops/<int:laptop_id>/refresh")
def api_refresh_laptop(laptop_id: int):
    df, _, _ = laptop_store()
    if df.empty:
        return jsonify({"error": "No laptops are available in Supabase. Run scripts/sync_csv_to_supabase.py first."}), 503
    match = df[df["id"] == laptop_id]
    if match.empty:
        return jsonify({"error": "Laptop not found"}), 404

    payload = laptop_payload(match.iloc[0], include_details=True)
    query = laptop_query(payload)
    live = search_live_prices(query)
    stored_prices = []
    for item in live["results"]:
        saved = supabase_store.upsert_price(
            query=query,
            platform=item["platform"],
            title=item["title"],
            price_rs=item["price_rs"],
            url=item["url"],
            image_url=item.get("image_url"),
            in_stock=item.get("in_stock"),
        )
        stored_prices.append(format_price_record(saved) if saved else {**item, "price_label": money(item.get("price_rs"))})

    if live.get("best_image_url"):
        payload["image_url"] = live["best_image_url"]
        payload["image_source"] = live.get("best_image_source")
        supabase_store.update_laptop_image(
            payload["source_key"],
            live["best_image_url"],
            live.get("best_image_source"),
        )

    ai_analysis = analyze_laptop(payload, stored_prices)
    if ai_analysis:
        payload["insights"] = ai_analysis
        supabase_store.upsert_analysis(
            laptop_key=laptop_key(payload),
            query=query,
            analysis=ai_analysis,
            model=settings.GROQ_MODEL,
        )

    payload["live_prices"] = stored_prices
    payload["price_errors"] = live["errors"]
    payload["ai_generated"] = bool(ai_analysis)
    payload["stored_in_supabase"] = supabase_store.is_configured()
    return jsonify(payload)


if __name__ == "__main__":
    app.run(debug=True)
