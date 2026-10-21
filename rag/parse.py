"""Turn handbook markdown into sections: one heading plus the text under it, with a stable ID."""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

RAW = Path(__file__).parent.parent / "data" / "raw"
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
BOLD_HEADING = re.compile(r"^\*\*(\d+\.\s+[^*]{3,80})\*\*\s*$")  # "**6. Retention Schedule**" used as a heading


@dataclass
class Section:
    id: str            # "time-off-types#sick-time", the same anchor the handbook website uses
    doc_id: str
    doc_title: str
    heading: str
    path: list[str]    # parent headings, e.g. ["Sick Time", "Sick Time - Procedures"]
    url: str
    text: str = ""
    lines: list[str] = field(default_factory=list, repr=False)


def slugify(heading: str) -> str:
    s = re.sub(r"[^\w\s-]", "", heading.lower())
    return re.sub(r"\s", "-", s.strip())


def clean(text: str) -> str:
    text = re.sub(r"\{\{[<%].*?[%>]\}\}", " ", text)             # Hugo shortcodes like {{% details %}}
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)            # images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)         # [link text](url) -> link text
    text = re.sub(r"<[^>]+>", " ", text)                         # HTML tags
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def split_front_matter(md: str) -> tuple[str, str]:
    m = re.match(r"^---\n(.*?)\n---\n", md, flags=re.S)
    if not m:
        return "", md
    title = re.search(r"^title:\s*\"?(.+?)\"?\s*$", m.group(1), flags=re.M)
    return (title.group(1) if title else ""), md[m.end():]


def parse_doc(doc_id: str, md: str, url: str) -> list[Section]:
    title, body = split_front_matter(md)
    top = Section(id=f"{doc_id}#top", doc_id=doc_id, doc_title=title, heading=title, path=[], url=url)
    sections, stack, seen = [top], [], {}
    in_code = False
    for line in body.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        m = None if in_code else HEADING.match(line)
        bold = None if in_code or m else BOLD_HEADING.match(line)
        if not (m or bold):
            sections[-1].lines.append(line)
            continue
        level, heading = (len(m.group(1)), clean(m.group(2))) if m else (2, clean(bold.group(1)))
        custom = re.search(r"\{#([\w-]+)\}$", heading)                 # "Title {#my-anchor}" sets its own anchor
        heading = re.sub(r"\s*\{#[\w-]+\}$", "", heading)
        stack = [(lvl, h) for lvl, h in stack if lvl < level] + [(level, heading)]
        anchor = custom.group(1) if custom else slugify(heading)
        seen[anchor] = seen.get(anchor, -1) + 1             # repeated headings get -1, -2 like the website
        if seen[anchor]:
            anchor = f"{anchor}-{seen[anchor]}"
        sections.append(Section(id=f"{doc_id}#{anchor}", doc_id=doc_id, doc_title=title, heading=heading,
                                path=[h for _, h in stack], url=f"{url}#{anchor}"))
    for s in sections:
        s.text = clean("\n".join(s.lines))
    return sections


def load_sections() -> list[Section]:
    manifest = json.loads((RAW / "manifest.json").read_text())
    sections = []
    for doc in manifest:
        sections += parse_doc(doc["doc_id"], (RAW / f"{doc['doc_id']}.md").read_text(), doc["url"])
    return sections


if __name__ == "__main__":
    sections = load_sections()
    with_text = [s for s in sections if s.text]
    print(f"{len(sections)} sections, {len(with_text)} with text")
    for s in with_text[1:10]:
        print(f"{s.id:<56} {' > '.join(s.path)[:42]:<42} {len(s.text):>5} chars")
