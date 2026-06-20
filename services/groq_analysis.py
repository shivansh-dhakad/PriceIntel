from __future__ import annotations

import json
from typing import Any

from services import settings


def analyze_laptop(laptop: dict[str, Any], live_prices: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    if not settings.GROQ_API_KEY:
        return None

    from groq import Groq

    client = Groq(api_key=settings.GROQ_API_KEY)
    prompt = {
        "laptop": laptop,
        "live_prices": live_prices or [],
        "required_json_shape": {
            "pros": ["short practical point"],
            "cons": ["short practical point"],
            "best_for": ["use case"],
            "not_for": ["use case"],
            "verdict": "two sentence buying advice",
        },
    }

    completion = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a concise laptop buying analyst for Indian shoppers. "
                    "Return only valid JSON. Do not invent exact live prices; use provided live_prices only."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=True)},
        ],
    )
    content = completion.choices[0].message.content or "{}"
    return json.loads(content)
