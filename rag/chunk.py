"""Two ways to cut the handbook into chunks: fixed-size (the baseline) and heading-aware."""
from dataclasses import dataclass

from rag.parse import Section


@dataclass
class Chunk:
    id: str
    title: str               # what we show as the source, e.g. "Time Off Types > Sick Time"
    text: str
    section_ids: list[str]   # every section this chunk overlaps, used for citations and evals
    url: str


def fixed_chunks(sections: list[Section], size: int = 1000) -> list[Chunk]:
    """Glue each document back together and cut it every `size` characters."""
    chunks = []
    for doc_id in dict.fromkeys(s.doc_id for s in sections):
        doc = [s for s in sections if s.doc_id == doc_id]
        text, spans = "", []
        for s in doc:
            start = len(text)
            text += (s.heading + "\n" if s.path else "") + s.text + "\n\n"
            spans.append((start, len(text), s.id))
        for n, start in enumerate(range(0, len(text), size)):
            end = start + size
            ids = [sid for a, b, sid in spans if a < end and b > start]
            chunks.append(Chunk(id=f"{doc_id}:{n}", title=f"{doc[0].doc_title} (part {n + 1})",
                                text=text[start:end], section_ids=ids, url=doc[0].url))
    return chunks


def section_chunks(sections: list[Section], max_chars: int = 1500) -> list[Chunk]:
    """One chunk per section; long sections are split on paragraph breaks.
    Every chunk starts with its heading path, so it still makes sense on its own."""
    chunks = []
    for s in sections:
        if not s.text:
            continue
        title = " > ".join([s.doc_title, *s.path])
        paras = [p for block in s.text.split("\n\n")
                 for p in (block.splitlines() if len(block) > max_chars else [block])]  # huge tables: by line
        pieces, current = [], ""
        for para in paras:
            if current and len(current) + len(para) > max_chars:
                pieces.append(current)
                current = ""
            current = f"{current}\n\n{para}" if current else para
        pieces.append(current)
        for n, piece in enumerate(pieces):
            chunks.append(Chunk(id=f"{s.id}:{n}", title=title, text=f"{title}\n\n{piece}",
                                section_ids=[s.id], url=s.url))
    return chunks


if __name__ == "__main__":
    from rag.parse import load_sections

    sections = load_sections()
    for name, chunks in (("fixed", fixed_chunks(sections)), ("sections", section_chunks(sections))):
        sizes = sorted(len(c.text) for c in chunks)
        print(f"{name:<9} {len(chunks):>4} chunks   median {sizes[len(sizes) // 2]:>4} chars   max {sizes[-1]:>5}")
