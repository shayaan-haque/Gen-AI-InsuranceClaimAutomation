import re
from collections import defaultdict
import tiktoken
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


MAX_CHILD_TOKENS = 450
CHILD_CONTEXT_TOKEN_RESERVE = 100

SECTION_PATTERN = re.compile(
    r"(?m)^(?P<section>\d+)\.\s+(?P<title>[A-Z][A-Z0-9 ,/&'()\-]+?)\s*$"
)
CLAUSE_PATTERN = re.compile(
    r"(?m)^(?P<clause>(?:POL|BEN|CLM|NET|EXC|SOP|GOV)-\d+\.\d+)\s{2,}(?P<title>[^\n]+)$"
)
DOCUMENT_CODE_PATTERN = re.compile(r"Document code:\s*([^\n]+)", re.IGNORECASE)

_tokenizer = tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    return len(_tokenizer.encode(text))


def _clean_lines(text: str) -> str:
    """Remove PDF line wrapping while retaining readable clause boundaries."""
    return " ".join(line.strip() for line in text.splitlines() if line.strip())


def _page_for_offset(page_ranges: list[tuple[int, int, int]], offset: int) -> int:
    previous_page = page_ranges[0][2]
    for start, end, page in page_ranges:
        if offset < start:
            return previous_page
        if start <= offset < end:
            return page
        previous_page = page
    return page_ranges[-1][2]


def _combine_pages(pages: list[Document]) -> tuple[str, list[tuple[int, int, int]]]:
    text_parts = []
    page_ranges = []
    cursor = 0

    for document in sorted(pages, key=lambda doc: doc.metadata.get("page", 0)):
        page_text = document.page_content.strip()
        start = cursor
        text_parts.append(page_text)
        cursor += len(page_text)
        page_ranges.append((start, cursor, document.metadata.get("page", 0)))
        text_parts.append("\n")
        cursor += 1

    return "".join(text_parts), page_ranges


def _fallback_chunks(document: Document) -> tuple[list[Document], dict[str, Document]]:
    """Retain support for PDFs that do not contain the expected clause IDs."""
    parent_id = f"{document.metadata.get('source', 'document')}:page:{document.metadata.get('page', 0)}"
    parent = Document(
        page_content=document.page_content,
        metadata={**document.metadata, "chunk_type": "parent", "parent_id": parent_id},
    )
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=MAX_CHILD_TOKENS,
        chunk_overlap=50,
    )
    children = splitter.split_documents([document])
    for index, child in enumerate(children):
        child.metadata.update(
            {
                "chunk_type": "child",
                "parent_id": parent_id,
                "child_index": index,
                "chunk_method": "token_fallback",
            }
        )
    return children, {parent_id: parent}


def get_parent_child_chunks(
    documents: list[Document],
) -> tuple[list[Document], dict[str, Document]]:
    """
    Build searchable child chunks and non-searchable parent sections.

    Child chunks contain complete clauses packed up to MAX_CHILD_TOKENS. Parent
    sections are kept in a dictionary and can be expanded after a child match.
    """
    documents_by_source = defaultdict(list)
    for document in documents:
        documents_by_source[document.metadata.get("source", "unknown")].append(document)

    children = []
    parents = {}

    for source, pages in documents_by_source.items():
        full_text, page_ranges = _combine_pages(pages)
        section_matches = list(SECTION_PATTERN.finditer(full_text))

        if not section_matches:
            for page in pages:
                fallback_children, fallback_parents = _fallback_chunks(page)
                children.extend(fallback_children)
                parents.update(fallback_parents)
            continue

        document_code_match = DOCUMENT_CODE_PATTERN.search(full_text)
        document_code = (
            document_code_match.group(1).strip()
            if document_code_match
            else source.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        )

        for section_index, section_match in enumerate(section_matches):
            section_start = section_match.start()
            section_end = (
                section_matches[section_index + 1].start()
                if section_index + 1 < len(section_matches)
                else len(full_text)
            )
            section_text = full_text[section_start:section_end].strip()
            section_id = section_match.group("section")
            section_title = section_match.group("title").strip()
            parent_id = f"{document_code}:section:{section_id}"
            page_start = _page_for_offset(page_ranges, section_start)
            page_end = _page_for_offset(page_ranges, max(section_start, section_end - 1))

            clause_matches = list(CLAUSE_PATTERN.finditer(section_text))
            if not clause_matches:
                # A section without clause IDs remains searchable as one child.
                parent_text = section_text
                clause_records = [
                    {
                        "id": f"SECTION-{section_id}",
                        "text": _clean_lines(section_text),
                        "start": section_start,
                        "end": section_end,
                    }
                ]
            else:
                clause_records = []
                parent_lines = [f"{section_id}. {section_title}"]
                heading_end = section_text.find("\n")
                section_intro = _clean_lines(
                    section_text[heading_end + 1 : clause_matches[0].start()]
                )
                if section_intro:
                    parent_lines.append(section_intro)
                for clause_index, clause_match in enumerate(clause_matches):
                    relative_end = (
                        clause_matches[clause_index + 1].start()
                        if clause_index + 1 < len(clause_matches)
                        else len(section_text)
                    )
                    clause_id = clause_match.group("clause")
                    clause_title = clause_match.group("title").strip()
                    body_start = clause_match.end()
                    body = _clean_lines(section_text[body_start:relative_end])
                    clause_text = f"{clause_id}  {clause_title}\n{body}".strip()
                    absolute_start = section_start + clause_match.start()
                    absolute_end = section_start + relative_end
                    clause_records.append(
                        {
                            "id": clause_id,
                            "text": clause_text,
                            "start": absolute_start,
                            "end": absolute_end,
                        }
                    )
                    parent_lines.append(clause_text)
                parent_text = "\n\n".join(parent_lines)

            parent = Document(
                page_content=parent_text,
                metadata={
                    "source": source,
                    "page": page_start,
                    "page_end": page_end,
                    "document_code": document_code,
                    "section_id": section_id,
                    "section_title": section_title,
                    "chunk_type": "parent",
                    "parent_id": parent_id,
                },
            )
            parents[parent_id] = parent

            current_records = []
            current_tokens = 0

            def emit_child(records: list[dict]):
                if not records:
                    return
                clause_ids = [record["id"] for record in records]
                child_page_start = _page_for_offset(page_ranges, records[0]["start"])
                child_page_end = _page_for_offset(page_ranges, max(records[-1]["start"], records[-1]["end"] - 1))
                context_header = (
                    f"Document code: {document_code}\n"
                    f"Section {section_id}: {section_title}\n"
                    f"Clauses: {', '.join(clause_ids)}"
                )
                children.append(
                    Document(
                        page_content=context_header
                        + "\n\n"
                        + "\n\n".join(record["text"] for record in records),
                        metadata={
                            "source": source,
                            "page": child_page_start,
                            "page_end": child_page_end,
                            "document_code": document_code,
                            "section_id": section_id,
                            "section_title": section_title,
                            "clause_ids": ",".join(clause_ids),
                            "clause_start": clause_ids[0],
                            "clause_end": clause_ids[-1],
                            "chunk_type": "child",
                            "chunk_method": "clause_aware",
                            "parent_id": parent_id,
                        },
                    )
                )

            for record in clause_records:
                record_tokens = _token_count(record["text"])
                if (
                    current_records
                    and current_tokens + record_tokens
                    > MAX_CHILD_TOKENS - CHILD_CONTEXT_TOKEN_RESERVE
                ):
                    emit_child(current_records)
                    current_records = []
                    current_tokens = 0
                current_records.append(record)
                current_tokens += record_tokens

            emit_child(current_records)

    return children, parents


def get_text_chunks(documents: list[Document]) -> list[Document]:
    """Backward-compatible helper: only searchable children are returned."""
    children,_ = get_parent_child_chunks(documents)
    return children
