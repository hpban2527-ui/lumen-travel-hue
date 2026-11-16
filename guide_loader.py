"""Read and validate the local travel guide."""
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

GUIDE_FILE = Path(__file__).resolve().parent / "knowledge_base.json"


class GuideError(ValueError):
    pass


def guide_digest(path=GUIDE_FILE):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_guide(path=GUIDE_FILE):
    try:
        entries = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GuideError("Could not open the city guide.") from exc

    if not isinstance(entries, list) or not 12 <= len(entries) <= 20:
        raise GuideError("The guide must contain 12 to 20 entries.")
    seen = set()
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise GuideError(f"Entry {i+1} must be an object.")
        for field in ("id", "title", "text", "source", "category"):
            if not isinstance(entry.get(field), str) or not entry[field].strip():
                raise GuideError(f"Entry {i+1} has an invalid {field}.")
        if not isinstance(entry.get("tags"), list) or not 2 <= len(entry["tags"]) <= 8:
            raise GuideError(f"Entry {i+1} must have 2-8 tags.")
        if any(not isinstance(t, str) or not t.strip() for t in entry["tags"]):
            raise GuideError(f"Entry {i+1} contains an invalid tag.")
        if not entry["id"].startswith("G") or not entry["id"][1:].isdigit():
            raise GuideError(f"Entry {i+1} has an invalid ID.")
        if entry["id"] in seen:
            raise GuideError(f"Duplicate entry: {entry['id']}")
        seen.add(entry["id"])
        src = urlparse(entry["source"])
        if src.scheme != "https" or not src.netloc:
            raise GuideError(f"Entry {entry['id']} has an invalid source URL.")
    return entries
