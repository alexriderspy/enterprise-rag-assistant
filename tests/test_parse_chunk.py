from rag.chunk import fixed_chunks, section_chunks
from rag.parse import clean, parse_doc

MD = """---
title: Time Off
---
Intro line with a [link](/somewhere).

## Sick Time
### Overview
Up to 25 paid sick days.

## Bereavement
### Overview
No documentation required.

**6. Retention Schedule**

Seven years.

## Travel Mode {#travel-mode}
Turn it on.
"""


def test_sections_get_website_style_ids():
    ids = [s.id for s in parse_doc("time-off", MD, "https://x/")]
    assert ids == ["time-off#top", "time-off#sick-time", "time-off#overview", "time-off#bereavement",
                   "time-off#overview-1", "time-off#6-retention-schedule", "time-off#travel-mode"]


def test_heading_path_and_clean_text():
    sections = parse_doc("time-off", MD, "https://x/")
    assert sections[4].path == ["Bereavement", "Overview"]
    assert sections[6].path == ["Travel Mode"]
    assert sections[0].text == "Intro line with a link."
    assert clean("{{% details summary=\"x\" %}}**bold**") == "bold"


def test_section_chunks_carry_their_heading_path():
    chunks = section_chunks(parse_doc("time-off", MD, "https://x/"))
    sick = next(c for c in chunks if c.section_ids == ["time-off#overview"])
    assert sick.text.startswith("Time Off > Sick Time > Overview")


def test_fixed_chunks_know_every_section_they_overlap():
    chunks = fixed_chunks(parse_doc("time-off", MD, "https://x/"), size=60)
    assert all(len(c.text) <= 60 for c in chunks)
    assert {"time-off#sick-time", "time-off#overview"} <= {sid for c in chunks for sid in c.section_ids}
