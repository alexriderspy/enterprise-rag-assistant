"""Download 14 policy pages from GitLab's public company handbook (CC BY-SA 4.0), pinned to one commit."""
import json
from pathlib import Path
from urllib.parse import quote

import requests

PROJECT = "gitlab-com%2Fcontent-sites%2Fhandbook"
COMMIT = "87ee69f41d1e862ad39cc621054aa5485ac4e455"  # main on 20 Oct 2026, so every run sees the same text
PAGES = [
    "people-group/time-off-and-absence/time-off-types",
    "people-group/time-off-and-absence/leave-types",
    "total-rewards/benefits/parental-leave-toolkit",
    "people-group/contracts-probation-periods",
    "people-group/relocation",
    "people-group/visas",
    "people-group/anti-harassment",
    "finance/expenses",
    "finance/spending-company-money",
    "finance/travel/_index",
    "legal/anti-corruption-policy",
    "legal/whistleblowing",
    "legal/record-retention-policy",
    "security/password-guidelines",
]
OUT = Path(__file__).parent / "raw"


def fetch(page: str) -> str:
    path = quote(f"content/handbook/{page}.md", safe="")
    url = f"https://gitlab.com/api/v4/projects/{PROJECT}/repository/files/{path}/raw"
    r = requests.get(url, params={"ref": COMMIT}, timeout=30)
    r.raise_for_status()
    return r.text


def main() -> None:
    OUT.mkdir(exist_ok=True)
    manifest, total = [], 0
    for page in PAGES:
        text = fetch(page)
        doc_id = page.split("/")[-1] if not page.endswith("_index") else page.split("/")[-2]
        (OUT / f"{doc_id}.md").write_text(text)
        url = "https://handbook.gitlab.com/handbook/" + page.removesuffix("_index").rstrip("/") + "/"
        manifest.append({"doc_id": doc_id, "url": url})
        total += len(text.split())
        print(f"{doc_id:<28} {len(text.split()):>6} words")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{len(manifest)} pages, {total:,} words")


if __name__ == "__main__":
    main()
