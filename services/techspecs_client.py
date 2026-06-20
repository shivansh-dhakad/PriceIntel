"""TechSpecs API client — search laptops and fetch full specs."""
from __future__ import annotations

import re
from typing import Any

import requests

from services import settings


def _headers() -> dict:
    return {
        "accept": "application/json",
        "content-type": "application/json",
        "x-api-key": settings.TECHSPECS_API_KEY,
        "x-api-id": settings.TECHSPECS_API_ID,
    }


def is_configured() -> bool:
    return bool(settings.TECHSPECS_API_KEY and settings.TECHSPECS_API_ID)


def search_laptops(keyword: str = "laptop", page: int = 1) -> dict[str, Any]:
    """Search TechSpecs for laptop products. Returns the raw API response."""
    resp = requests.get(
        f"{settings.TECHSPECS_BASE_URL}/products/search",
        headers=_headers(),
        params={"keyword": keyword, "category": "Laptops", "page": page},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def get_product(product_id: str) -> dict[str, Any] | None:
    """Fetch full specs for a single product by TechSpecs ID."""
    resp = requests.get(
        f"{settings.TECHSPECS_BASE_URL}/products/{product_id}",
        headers=_headers(),
        timeout=15,
    )
    if not resp.ok:
        return None
    data = resp.json()
    return data.get("data") or data


# ── Data extraction helpers ────────────────────────────────────────────────

def _get(obj: Any, *keys: str, default: Any = None) -> Any:
    """Safely navigate nested dicts."""
    for key in keys:
        if not isinstance(obj, dict):
            return default
        obj = obj.get(key, default)
        if obj is None:
            return default
    return obj


def _parse_ram_gb(text: str | None) -> int | None:
    if not text:
        return None
    m = re.search(r"(\d+)\s*GB", str(text), re.IGNORECASE)
    return int(m.group(1)) if m else None


def _parse_ssd_gb(text: str | None) -> int | None:
    if not text:
        return None
    text = str(text)
    m_tb = re.search(r"(\d+(?:\.\d+)?)\s*TB", text, re.IGNORECASE)
    if m_tb:
        return int(float(m_tb.group(1)) * 1024)
    m_gb = re.search(r"(\d+)\s*GB", text, re.IGNORECASE)
    return int(m_gb.group(1)) if m_gb else None


def _parse_weight_kg(text: str | None) -> float | None:
    if not text:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*kg", str(text), re.IGNORECASE)
    return float(m.group(1)) if m else None


def _parse_display_size(text: str | None) -> float | None:
    if not text:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*inch", str(text), re.IGNORECASE)
    return float(m.group(1)) if m else None


def _gpu_tier(gpu_model: str | None) -> str:
    if not gpu_model:
        return "Integrated"
    g = gpu_model.lower()
    if any(k in g for k in ("rtx 40", "rtx 39", "rtx 38")):
        return "High"
    if any(k in g for k in ("rtx 30", "rtx 20", "rx 6800", "rx 6900", "rx 7")):
        return "High"
    if any(k in g for k in ("rtx 3060", "rtx 3070", "rx 6700", "rx 6600")):
        return "Mid"
    if any(k in g for k in ("gtx 16", "rtx 3050", "rx 5", "mx5", "mx4")):
        return "Entry Gaming"
    if any(k in g for k in ("mx3", "mx2", "mx1", "940", "930", "920")):
        return "Basic Dedicated"
    if "intel" in g or "integrated" in g or "uhd" in g or "iris" in g:
        return "Integrated"
    return "Integrated"


def extract_laptop_row(product: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a TechSpecs full product dict into a row suitable for
    supabase_store.upsert_laptop().
    """
    ts_id = product.get("_id") or product.get("id", "")
    prod  = product.get("Product", {})
    design = product.get("Design", {})
    inside = product.get("Inside", {})
    display_data = product.get("Display", {})

    brand    = prod.get("Brand", "")
    family   = prod.get("Family", "")
    model_name = prod.get("Model Name") or prod.get("Part Number") or ""
    model    = f"{family} {model_name}".strip() if family else model_name

    cpu      = inside.get("CPU", {})
    ram      = inside.get("RAM", {})
    gpu      = inside.get("GPU", {})
    ssd_data = inside.get("SSD", {})
    software = inside.get("Software", {})
    body     = design.get("Body", {})
    wireless = inside.get("Wireless", {})
    battery  = inside.get("Battery", {})

    ram_gb   = _parse_ram_gb(ram.get("Capacity"))
    ssd_gb   = _parse_ssd_gb(ssd_data.get("Total SSD Capacity") or ssd_data.get("Capacity"))
    weight   = _parse_weight_kg(body.get("Weight"))
    display  = _parse_display_size(display_data.get("Diagonal"))
    gpu_model = (gpu.get("Dedicated Card Model") or "").split(",")[0].strip()
    gpu_tier  = _gpu_tier(gpu_model or gpu.get("Integrated Card Model", ""))

    cpu_brand = cpu.get("Brand", "")
    cpu_model_str = cpu.get("Model", "") or cpu.get("Family", "")
    cpu_cat   = cpu.get("Generation") or cpu_model_str

    os_ver    = software.get("Operating System Version", "")
    touchscreen = "Yes" if "Touchscreen" in str(product.get("Yes", {})) else "No"

    return {
        "source_key": f"techspecs-{ts_id}",
        "brand": brand,
        "model": model[:240],
        "ram_gb": ram_gb,
        "ssd_gb": ssd_gb,
        "weight_kg": weight,
        "display_size": display,
        "os": os_ver,
        "touchscreen": touchscreen,
        "gpu_tier": gpu_tier,
        "gpu_brand": (gpu.get("Integrated Card Brand") or "").split(",")[0].strip() or None,
        "cpu_brand": cpu_brand,
        "cpu_category": cpu_cat[:100] if cpu_cat else None,
        "raw_specs": {
            "Brand": brand,
            "Model": model,
            "Price (Rs)": None,
            "RAM_GB": ram_gb,
            "SSD_GB": ssd_gb,
            "Weight": weight,
            "Display Size": display,
            "Operating System": os_ver,
            "Display Touchscreen": touchscreen,
            "GPU_Tier": gpu_tier,
            "CPU_Brand": cpu_brand,
            "CPU_Category": cpu_cat,
            # TechSpecs-specific rich fields
            "techspecs_id": ts_id,
            "cpu_model": cpu_model_str,
            "cpu_cores": cpu.get("Number of Cores"),
            "cpu_clock_ghz": cpu.get("Clock Speed"),
            "ram_type": ram.get("Type"),
            "ram_max_gb": _parse_ram_gb(ram.get("Maximum Capacity")),
            "display_resolution": display_data.get("Resolution (H x W)"),
            "display_definition": display_data.get("Definition"),
            "display_aspect": display_data.get("Aspect Ratio"),
            "gpu_dedicated_model": gpu_model or None,
            "gpu_dedicated_memory": gpu.get("Dedicated Card Memory", "").split(",")[0].strip() or None,
            "wifi_standard": wireless.get("WiFi Standards"),
            "bluetooth": wireless.get("Bluetooth Version"),
            "battery_info": battery.get("Battery Life") or battery.get("Capacity") if battery else None,
            "keyboard_backlight": design.get("Keyboard", {}).get("Backlight Color"),
            "colors": body.get("Colors"),
            "thickness_mm": body.get("Thickness_mm"),
            "release_date": product.get("Release Date"),
        },
    }
