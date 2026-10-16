from langchain_text_splitters import RecursiveCharacterTextSplitter

import json
import sys
from pathlib import Path
import json
import sys
from pathlib import Path

from unstructured.chunking.title import chunk_by_title
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.auto import partition


def to_markdown(elements) -> str:
    """Turn typed elements into Markdown so every format ends up looking the same."""
    lines = []
    for el in elements:
        text = (el.text or "").strip()
        if not text:
            continue
        kind = el.category
        if kind == "Title":
            lines.append(f"\n## {text}\n")
        elif kind == "ListItem":
            lines.append(f"- {text}")
        elif kind == "Table":
            html = getattr(el.metadata, "text_as_html", None)
            lines.append(html or text)
        else:
            lines.append(text)
    return "\n".join(lines)


# load elements from file
def load_elements(
    path: str = str(
        Path(__file__).resolve().parents[1]
        / "assets"
        / "gamma_analytics_hr_policy_handbook.docx"
    ),
):
    kwargs = {}
    if Path(path).suffix.lower() == ".pdf":
        kwargs["strategy"] = "fast"
    return partition(filename=path, **kwargs)

    elem = load_elements()
    output_path = (
        Path(__file__).resolve().parents[1] / "outputs" / "parsed_document.md"
    )
    output_path.write_text(to_markdown(elem), encoding="utf-8")


def splitter_user_input_file(text):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "], chunk_size=1000
    )

    chunks = splitter.split_text(text)
    return chunks
