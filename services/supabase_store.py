from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import math

from services import settings


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def client():
    if not settings.USE_SUPABASE or not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
        return None
    try:
        from supabase import create_client

        return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
    except Exception:
        return None


def is_configured() -> bool:
    return client() is not None


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if math.isnan(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return clean_value(value.item())
    if isinstance(value, dict):
        return {key: clean_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_value(item) for item in value]
    return value


def upsert_laptop(row: dict[str, Any]) -> dict[str, Any] | None:
    sb = client()
    if sb is None:
        return None

    payload = laptop_payload(row)
    response = sb.table("laptops").upsert(payload, on_conflict="source_key").execute()
    return response.data[0] if response.data else None


def laptop_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = clean_value({
        "source_key": row.get("source_key"),
        "brand": row.get("brand"),
        "model": row.get("model"),
        "price_rs": row.get("price_rs"),
        "ram_gb": row.get("ram_gb"),
        "ssd_gb": row.get("ssd_gb"),
        "gpu_tier": row.get("gpu_tier"),
        "cpu_category": row.get("cpu_category"),
        "cpu_brand": row.get("cpu_brand"),
        "gpu_brand": row.get("gpu_brand"),
        "weight_kg": row.get("weight_kg"),
        "display_size": row.get("display_size"),
        "os": row.get("os"),
        "touchscreen": row.get("touchscreen"),
        "raw_specs": row.get("raw_specs") or {},
        "updated_at": now_iso(),
    })
    if row.get("image_url"):
        payload["image_url"] = row.get("image_url")
        payload["image_source"] = row.get("image_source")
    return payload


def upsert_laptops(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sb = client()
    if sb is None or not rows:
        return []
    response = sb.table("laptops").upsert(
        [laptop_payload(row) for row in rows],
        on_conflict="source_key",
    ).execute()
    return response.data or []


def list_laptops(page_size: int = 1000) -> list[dict[str, Any]]:
    sb = client()
    if sb is None:
        return []

    rows: list[dict[str, Any]] = []
    start = 0
    while True:
        end = start + page_size - 1
        response = (
            sb.table("laptops")
            .select("*")
            .order("source_key")
            .range(start, end)
            .execute()
        )
        batch = response.data or []
        rows.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size
    return rows


def get_laptop_by_source_key(source_key: str) -> dict[str, Any] | None:
    sb = client()
    if sb is None or not source_key:
        return None
    response = sb.table("laptops").select("*").eq("source_key", source_key).limit(1).execute()
    return response.data[0] if response.data else None


def update_laptop_image(source_key: str, image_url: str | None, image_source: str | None = None) -> None:
    sb = client()
    if sb is None or not source_key or not image_url:
        return
    sb.table("laptops").update(
        {
            "image_url": image_url,
            "image_source": image_source,
            "updated_at": now_iso(),
        }
    ).eq("source_key", source_key).execute()


def upsert_price(
    *,
    query: str,
    platform: str,
    title: str,
    price_rs: float | int | None,
    url: str,
    in_stock: bool | None = None,
    image_url: str | None = None,
    laptop_id: str | None = None,
) -> dict[str, Any] | None:
    sb = client()
    if sb is None:
        return None

    payload = {
        "laptop_id": laptop_id,
        "query": query,
        "platform": platform,
        "title": title,
        "price_rs": price_rs,
        "url": url,
        "image_url": image_url,
        "in_stock": in_stock,
        "scraped_at": now_iso(),
    }
    response = sb.table("laptop_prices").upsert(
        payload,
        on_conflict="platform,url",
    ).execute()
    return response.data[0] if response.data else None


def get_latest_prices(query: str, limit: int = 6) -> list[dict[str, Any]]:
    sb = client()
    if sb is None or not query:
        return []
    response = (
        sb.table("laptop_prices")
        .select("*")
        .ilike("query", f"%{query[:60]}%")
        .order("scraped_at", desc=True)
        .limit(limit)
        .execute()
    )
    return response.data or []


def upsert_analysis(
    *,
    laptop_key: str,
    query: str,
    analysis: dict[str, Any],
    model: str,
) -> dict[str, Any] | None:
    sb = client()
    if sb is None:
        return None
    payload = {
        "laptop_key": laptop_key,
        "query": query,
        "analysis": analysis,
        "model": model,
        "generated_at": now_iso(),
    }
    response = sb.table("laptop_ai_analyses").upsert(payload, on_conflict="laptop_key").execute()
    return response.data[0] if response.data else None


def get_analysis(laptop_key: str) -> dict[str, Any] | None:
    sb = client()
    if sb is None:
        return None
    response = (
        sb.table("laptop_ai_analyses")
        .select("*")
        .eq("laptop_key", laptop_key)
        .order("generated_at", desc=True)
        .limit(1)
        .execute()
    )
    return response.data[0] if response.data else None


def list_market_prices(
    *,
    query: str = "",
    platform: str = "",
    page: int = 1,
    page_size: int = 48,
) -> list[dict[str, Any]]:
    """Return stored Amazon/Flipkart listings from the laptop_prices table."""
    sb = client()
    if sb is None:
        return []

    offset = (max(1, page) - 1) * page_size
    qs = (
        sb.table("laptop_prices")
        .select("*")
        .order("scraped_at", desc=True)
    )

    if query:
        qs = qs.ilike("title", f"%{query[:80]}%")

    if platform and platform.lower() in {"amazon", "flipkart"}:
        qs = qs.ilike("platform", platform)

    qs = qs.range(offset, offset + page_size - 1)
    response = qs.execute()
    return response.data or []


def get_market_price_stats() -> dict[str, Any]:
    """Return aggregate stats for stored marketplace listings."""
    sb = client()
    if sb is None:
        return {"total": 0, "amazon": 0, "flipkart": 0}
    try:
        total_resp = sb.table("laptop_prices").select("id", count="exact").execute()
        amazon_resp = sb.table("laptop_prices").select("id", count="exact").ilike("platform", "Amazon").execute()
        flipkart_resp = sb.table("laptop_prices").select("id", count="exact").ilike("platform", "Flipkart").execute()
        return {
            "total": total_resp.count or 0,
            "amazon": amazon_resp.count or 0,
            "flipkart": flipkart_resp.count or 0,
        }
    except Exception:
        return {"total": 0, "amazon": 0, "flipkart": 0}

