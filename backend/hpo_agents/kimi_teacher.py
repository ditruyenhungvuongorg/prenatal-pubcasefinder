"""Optional Kimi API helper for *de-identified lexicon drafting only*.

Do not send patient reports or protected health information through this helper.
Every returned synonym remains a candidate until a domain expert approves it.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any
from urllib.parse import urlparse
from urllib import error, request

from .schemas import validate_hpo_id


_IDENTIFIER_HINT_RE = re.compile(
    r"(?:\b(?:h[oọ]\s*t[eê]n|t[eê]n\s*b[eệ]nh\s*nh[aâ]n|m[aã]\s*b[eệ]nh\s*[aá]n|"
    r"s[oố]\s*h[oồ]\s*s[oơ]|[dđ]i[eệ]n\s*tho[aạ]i|[dđ][iị]a\s*ch[iỉ]|ng[aà]y\s*sinh)\b|"
    r"[\w.+-]+@[\w.-]+\.\w+)",
    re.IGNORECASE,
)


def draft_vietnamese_synonyms(
    hpo_id: str,
    english_label: str,
    root_phrase_vi: str,
    *,
    api_key: str | None = None,
    model: str = "kimi-k2.6",
    base_url: str = "https://api.moonshot.ai/v1",
    timeout_seconds: int = 90,
    deidentified_attestation: bool = False,
) -> dict[str, Any]:
    """Ask Kimi for candidates; never call this with patient-derived text."""

    validate_hpo_id(hpo_id)
    if not deidentified_attestation:
        raise ValueError(
            "Set deidentified_attestation=True only after confirming the three term fields "
            "contain no patient-derived or identifying text"
        )
    fields = {
        "english_label": str(english_label).strip(),
        "root_phrase_vi": str(root_phrase_vi).strip(),
    }
    for field_name, value in fields.items():
        if not value or len(value) > 200 or "\n" in value or "\r" in value:
            raise ValueError(f"{field_name} must be one non-empty ontology phrase (max 200 characters)")
        if _IDENTIFIER_HINT_RE.search(value):
            raise ValueError(f"{field_name} appears to contain identifying text")
    parsed_url = urlparse(base_url)
    if parsed_url.scheme != "https" or not parsed_url.netloc:
        raise ValueError("base_url must be an HTTPS API endpoint")
    if not isinstance(timeout_seconds, int) or not 1 <= timeout_seconds <= 300:
        raise ValueError("timeout_seconds must be an integer from 1 to 300")
    key = api_key or os.environ.get("MOONSHOT_API_KEY")
    if not key:
        raise ValueError("Set MOONSHOT_API_KEY; never paste the key into a notebook cell")
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Generate Vietnamese clinical synonym candidates for a supplied ontology term. "
                    "Return JSON only with key synonyms_vi (array of strings). Do not change or invent IDs."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "hpo_id": hpo_id,
                        "english_label": english_label,
                        "root_phrase_vi": root_phrase_vi,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }
    encoded = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=encoded,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Moonshot API failed with HTTP {exc.code}: {detail}") from exc
    content = body["choices"][0]["message"]["content"]
    if content.startswith("```json"):
        content = content[7:].rsplit("```", 1)[0].strip()
    result = json.loads(content)
    synonyms = result.get("synonyms_vi")
    if not isinstance(synonyms, list) or not all(isinstance(item, str) for item in synonyms):
        raise ValueError("Kimi response does not match the synonym schema")
    return {
        "hpo_id": hpo_id,
        "english_label": english_label,
        "root_phrase_vi": root_phrase_vi,
        "synonyms_vi": list(dict.fromkeys(item.strip() for item in synonyms if item.strip())),
        "source": model,
        "review_status": "candidate_needs_expert_review",
        "contains_patient_text": False,
        "deidentified_attestation": True,
    }
