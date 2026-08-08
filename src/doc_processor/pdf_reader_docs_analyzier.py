import asyncio
import base64
import csv
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv
import fitz
from openai import AsyncOpenAI
load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================
INPUT_DIR = Path(os.getenv("PATH_DATA")) / "raw" / "documents"
OUTPUT_DIR = Path(os.getenv("PATH_DATA")) / "interleaved" / "output"

OCR_OUTPUT_DIR = OUTPUT_DIR / "ocr"                    # one .md per PDF
EXTRACTED_TEXT_DIR = OUTPUT_DIR / "extracted_text"     # one .txt per PDF (referenced by doc_index.json)
RESULTS_OUTPUT_DIR = OUTPUT_DIR / "results"            # raw LLM responses, for debugging

DOC_INDEX_PATH = OUTPUT_DIR / "doc_index.json"

ACCOUNT_IDS_CSV_PATH = Path(os.getenv("PATH_DATA")) / "interleaved" / "accountTOscenario.csv"


# ------------------------------------------------------------
# Chandra vLLM
# ------------------------------------------------------------

CHANDRA_URL = "http://localhost:8000/v1"
CHANDRA_MODEL = "datalab-to/chandra-ocr-2"


# ------------------------------------------------------------
# OpenAI-compatible LLM
# ------------------------------------------------------------

LLM_URL = "https://crof.ai/v1"
LLM_MODEL = "deepseek-v4-flash"
LLM_API = os.getenv("LLMAPIKEY")

# ------------------------------------------------------------
# Workers
# ------------------------------------------------------------

# You requested exactly one of each.
OCR_WORKERS = 1
LLM_WORKERS = 1


# ------------------------------------------------------------
# Queue sizes
# ------------------------------------------------------------

# Prevent too many rendered images from staying in RAM.
OCR_QUEUE_SIZE = 4

# Finished documents waiting for first-page LLM classification.
LLM_QUEUE_SIZE = 10


# ------------------------------------------------------------
# PDF rendering
# ------------------------------------------------------------

# 2x resolution.
# Increase for small text, decrease if memory usage is high.
PDF_SCALE = 2.0


# ------------------------------------------------------------
# Words to search for
# ------------------------------------------------------------

TARGET_PHRASE = "ДОГОВОР БАНКОВСКОГО ЗАЙМА"

account_id_words: list[str] = []

if ACCOUNT_IDS_CSV_PATH.exists():
    with ACCOUNT_IDS_CSV_PATH.open(
        mode="r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.reader(file)

        for row in reader:
            # Check to avoid IndexError on empty/short lines.
            if len(row) > 0 and row[0].strip():
                account_id_words.append(row[0].strip())

ACCOUNT_ID_SET = set(account_id_words)

TARGET_WORDS = (TARGET_PHRASE, *account_id_words)


# ============================================================
# OPENAI CLIENTS
# ============================================================

chandra_client = AsyncOpenAI(
    base_url=CHANDRA_URL,
    api_key="dummy",
)


llm_client = AsyncOpenAI(
    base_url=LLM_URL,
    api_key=LLM_API,
)


# ============================================================
# QUEUE DATA
# ============================================================

@dataclass
class OCRTask:
    pdf_path: Path
    page_number: int
    total_pages: int
    image: bytes


@dataclass
class LLMTask:
    """
    Exactly one of these is created per PDF, once every page
    of that PDF has finished OCR. It always carries the first
    page's text, whether or not any target word was ever
    matched anywhere in the document.
    """
    pdf_path: Path
    first_page_text: str
    account_ids: list[str]
    matched_target_phrase: bool


@dataclass
class DocumentBuffer:
    """
    Accumulates OCR output for a single PDF while its pages
    stream in from the OCR worker, out of order is not
    expected (pages are queued strictly in order) but we key
    by page number to be safe.
    """
    total_pages: int
    pages: dict[int, str] = field(default_factory=dict)
    matched_words: set[str] = field(default_factory=set)

    def is_complete(self) -> bool:
        return len(self.pages) >= self.total_pages

    def first_page_text(self) -> str:
        return self.pages.get(1, "")

    def ordered_text(self, separator: str) -> str:
        return separator.join(
            self.pages[page_number]
            for page_number in sorted(self.pages)
        )


# ============================================================
# QUEUES
# ============================================================

ocr_queue = asyncio.Queue(
    maxsize=OCR_QUEUE_SIZE
)

llm_queue = asyncio.Queue(
    maxsize=LLM_QUEUE_SIZE
)

# Only ever touched by the single OCR worker, so no locking needed.
document_buffers: dict[Path, DocumentBuffer] = {}


# ============================================================
# PDF
# ============================================================

def render_page(page) -> bytes:
    """
    Render one PDF page into PNG bytes.
    """

    matrix = fitz.Matrix(
        PDF_SCALE,
        PDF_SCALE,
    )

    pixmap = page.get_pixmap(
        matrix=matrix,
        alpha=False,
    )

    return pixmap.tobytes("png")


# ============================================================
# KEYWORD SEARCH (spacing-tolerant)
# ============================================================

def build_flexible_pattern(keyword: str) -> re.Pattern:
    """
    Build a regex that tolerates OCR inserting whitespace
    between individual characters.

    Example, the keyword:

        ДОГОВОР БАНКОВСКОГО ЗАЙМА

    will also match OCR output like:

        Д О Г О В О Р   Б А Н К О В С К О Г О   З А Й М А

    We do this by dropping the keyword's own spaces (they get
    reintroduced as "\\s*" anyway) and joining every remaining
    character with "\\s*", then anchoring on word boundaries so
    we don't match inside unrelated longer words.
    """

    letters = [ch for ch in keyword if not ch.isspace()]

    body = r"\s*".join(re.escape(ch) for ch in letters)

    pattern = rf"\b{body}\b"

    return re.compile(pattern, flags=re.IGNORECASE)


# Pre-compiled once at startup: (original_keyword, compiled_pattern)
TARGET_PATTERNS: list[tuple[str, re.Pattern]] = [
    (keyword, build_flexible_pattern(keyword))
    for keyword in TARGET_WORDS
]


def find_keywords(text: str) -> list[str]:
    """
    Find target words/phrases in OCR text, tolerant of stray
    whitespace the OCR engine may insert between characters.
    """

    found = []

    for keyword, pattern in TARGET_PATTERNS:

        if pattern.search(text):
            found.append(keyword)

    return found


# ============================================================
# CHANDRA OCR
# ============================================================

async def chandra_ocr(
    image: bytes,
) -> str:

    image_base64 = base64.b64encode(
        image
    ).decode("utf-8")

    response = await chandra_client.chat.completions.create(
        model=CHANDRA_MODEL,

        messages=[
            {
                "role": "user",

                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Perform OCR on this document page. "
                            "Preserve the document structure, "
                            "tables, headings, numbers, "
                            "and formatting as accurately "
                            "as possible. "
                            "Return the result as Markdown."
                        ),
                    },

                    {
                        "type": "image_url",

                        "image_url": {
                            "url": (
                                "data:image/png;base64,"
                                + image_base64
                            )
                        },
                    },
                ],
            }
        ],
    )

    return (
        response
        .choices[0]
        .message
        .content
        or ""
    )


# ============================================================
# LLM CLASSIFICATION (first page only, always run)
# ============================================================

def _parse_llm_json(raw: str) -> dict:
    """
    Strip Markdown code fences if present and parse JSON.
    Falls back to a safe default on any failure.
    """

    cleaned = raw.strip()

    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        data = json.loads(cleaned)

    except json.JSONDecodeError:
        return {
            "doc_type": "UNKNOWN",
            "date": None,
        }

    return {
        "doc_type": data.get("doc_type") or "UNKNOWN",
        "date": data.get("date"),
    }


async def classify_first_page(
    first_page_text: str,
) -> dict:
    """
    Always called once per document, on the first page's OCR
    text, regardless of whether any target word was matched
    anywhere in the document.
    """

    prompt = f"""
You are analyzing the FIRST PAGE of a scanned banking document.
The OCR text below may contain recognition errors.

Determine:

1. "doc_type": a short, uppercase, snake-style category label,
   e.g. LOAN_AGREEMENT, PAYMENT_RECEIPT, STATEMENT, APPLICATION,
   OTHER. Infer it from context even if the OCR is noisy.

2. "date": the primary document date shown on this page, in
   ISO format "YYYY-MM-DD". Use null if no date is present.

Return ONLY valid JSON, with no Markdown fences and no extra
text, using exactly this structure:

{{
    "doc_type": "",
    "date": null
}}

FIRST PAGE OCR TEXT:

{first_page_text}
"""

    response = await llm_client.chat.completions.create(
        model=LLM_MODEL,

        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    raw = (
        response
        .choices[0]
        .message
        .content
        or ""
    )

    return _parse_llm_json(raw)


# ============================================================
# MARKDOWN / TEXT OUTPUT (one file per PDF, written once)
# ============================================================

def save_document_markdown(
    pdf_path: Path,
    buffer: DocumentBuffer,
):
    """
    Write ONE Markdown file per PDF, containing every page,
    written once the whole document has finished OCR.
    """

    OCR_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        OCR_OUTPUT_DIR
        / f"{pdf_path.stem}.md"
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(f"# {pdf_path.name}\n")

        for page_number in sorted(buffer.pages):
            file.write(f"\n\n## Page {page_number}\n\n")
            file.write(buffer.pages[page_number])
            file.write("\n")


def save_document_plain_text(
    pdf_path: Path,
    buffer: DocumentBuffer,
) -> Path:
    """
    Write ONE plain-text file per PDF (all pages concatenated).
    This is the file referenced by "extracted_txt" in
    doc_index.json.
    """

    EXTRACTED_TEXT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        EXTRACTED_TEXT_DIR
        / f"{pdf_path.stem}.txt"
    )

    separator = "\n\n===== Page {n} =====\n\n"

    parts = []
    for page_number in sorted(buffer.pages):
        parts.append(separator.format(n=page_number))
        parts.append(buffer.pages[page_number])

    output_file.write_text(
        "".join(parts),
        encoding="utf-8",
    )

    return output_file


def save_llm_debug_result(
    pdf_path: Path,
    classification: dict,
    account_ids: list[str],
    matched_target_phrase: bool,
):
    """
    Optional debug trail of the raw classification per PDF.
    """

    RESULTS_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        RESULTS_OUTPUT_DIR
        / f"{pdf_path.stem}.json"
    )

    data = {
        "file": pdf_path.name,
        "matched_target_phrase": matched_target_phrase,
        "account_ids": account_ids,
        "classification": classification,
    }

    output_file.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ============================================================
# doc_index.json
# ============================================================

def update_doc_index(
    pdf_path: Path,
    extracted_txt_path: Path,
    classification: dict,
    account_ids: list[str],
):
    """
    Read-modify-write doc_index.json.

    Safe because there is exactly ONE LLM worker, so calls to
    this function are never concurrent.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if DOC_INDEX_PATH.exists():
        try:
            doc_index = json.loads(
                DOC_INDEX_PATH.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError:
            doc_index = {}
    else:
        doc_index = {}

    try:
        extracted_txt_display = str(
            extracted_txt_path.relative_to(OUTPUT_DIR.parent)
        )
    except ValueError:
        extracted_txt_display = str(extracted_txt_path)

    doc_index[pdf_path.name] = {
        "extracted_txt": extracted_txt_display,
        "doc_type": classification.get("doc_type", "UNKNOWN"),
        "account_ids": sorted(account_ids),
        "date": classification.get("date"),
    }

    DOC_INDEX_PATH.write_text(
        json.dumps(doc_index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ============================================================
# PDF PRODUCER
# ============================================================

async def pdf_producer(
    pdf_paths: list[Path],
):
    """
    Read PDFs one page at a time.

    A rendered page is immediately put into the OCR queue.
    This prevents the whole PDF from being kept in memory.
    """

    for pdf_path in pdf_paths:

        print(f"[PDF] Opening: {pdf_path.name}")

        pdf = fitz.open(pdf_path)

        total_pages = pdf.page_count

        try:

            for page_number, page in enumerate(pdf, start=1):

                # Rendering is CPU work.
                # Put it into a thread so the asyncio event
                # loop isn't blocked.
                image = await asyncio.to_thread(
                    render_page,
                    page,
                )

                task = OCRTask(
                    pdf_path=pdf_path,
                    page_number=page_number,
                    total_pages=total_pages,
                    image=image,
                )

                # If OCR is busy and the queue is full, this
                # waits instead of consuming more RAM.
                await ocr_queue.put(task)

                print(
                    f"[QUEUE] {pdf_path.name} "
                    f"page {page_number}/{total_pages} → OCR"
                )

        finally:
            pdf.close()

        print(f"[PDF] Finished queuing: {pdf_path.name}")


# ============================================================
# OCR WORKER
# ============================================================

async def ocr_worker():
    """
    Exactly ONE OCR worker.

    Continuously takes pages from the OCR queue, accumulates
    them per-document, and once a document's last page has
    arrived: writes the per-PDF Markdown + text files and
    enqueues exactly one LLM classification task (first page
    text), regardless of whether any target word was matched.
    """

    while True:

        task: OCRTask = await ocr_queue.get()

        try:

            print(f"[OCR] {task.pdf_path.name} page {task.page_number}")

            # Send image to Chandra.
            text = await chandra_ocr(task.image)

            buffer = document_buffers.setdefault(
                task.pdf_path,
                DocumentBuffer(total_pages=task.total_pages),
            )

            buffer.pages[task.page_number] = text

            matched_words = find_keywords(text)

            if matched_words:
                print(
                    f"[MATCH] {task.pdf_path.name} "
                    f"page {task.page_number}: {matched_words}"
                )
                buffer.matched_words.update(matched_words)

            if buffer.is_complete():

                print(f"[DOC COMPLETE] {task.pdf_path.name}")

                # One Markdown file and one text file per PDF.
                save_document_markdown(task.pdf_path, buffer)
                extracted_txt_path = save_document_plain_text(
                    task.pdf_path, buffer
                )

                account_ids = sorted(
                    buffer.matched_words & ACCOUNT_ID_SET
                )
                matched_target_phrase = (
                    TARGET_PHRASE in buffer.matched_words
                )

                llm_task = LLMTask(
                    pdf_path=task.pdf_path,
                    first_page_text=buffer.first_page_text(),
                    account_ids=account_ids,
                    matched_target_phrase=matched_target_phrase,
                )

                # Stash the text path so the LLM worker doesn't
                # need to recompute it.
                llm_task_extra_paths[task.pdf_path] = extracted_txt_path

                await llm_queue.put(llm_task)

                # Free memory: this document is done.
                del document_buffers[task.pdf_path]

        except Exception as error:
            print(
                f"[OCR ERROR] {task.pdf_path.name} "
                f"page {task.page_number}: {error}"
            )

        finally:
            del task.image
            ocr_queue.task_done()


# Small side-channel mapping pdf_path -> extracted text file path,
# populated by the OCR worker right before enqueueing the LLMTask.
llm_task_extra_paths: dict[Path, Path] = {}


# ============================================================
# LLM WORKER
# ============================================================

async def llm_worker():
    """
    Exactly ONE LLM worker.

    Classifies the first page of every document (always, no
    matter whether target words were matched) and writes the
    result into doc_index.json.
    """

    while True:

        task: LLMTask = await llm_queue.get()

        try:

            print(f"[LLM] classifying first page: {task.pdf_path.name}")

            classification = await classify_first_page(
                task.first_page_text
            )

            save_llm_debug_result(
                pdf_path=task.pdf_path,
                classification=classification,
                account_ids=task.account_ids,
                matched_target_phrase=task.matched_target_phrase,
            )

            extracted_txt_path = llm_task_extra_paths.pop(
                task.pdf_path,
                EXTRACTED_TEXT_DIR / f"{task.pdf_path.stem}.txt",
            )

            update_doc_index(
                pdf_path=task.pdf_path,
                extracted_txt_path=extracted_txt_path,
                classification=classification,
                account_ids=task.account_ids,
            )

            print(f"[LLM DONE] {task.pdf_path.name}: {classification}")

        except Exception as error:
            print(f"[LLM ERROR] {task.pdf_path.name}: {error}")

        finally:
            llm_queue.task_done()


# ============================================================
# MAIN
# ============================================================

async def main():

    # Create output directories.
    OCR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACTED_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find PDFs.
    pdf_paths = sorted(INPUT_DIR.glob("*.pdf"))

    if not pdf_paths:
        print(f"No PDF files found in: {INPUT_DIR.resolve()}")
        return

    print(f"Found {len(pdf_paths)} PDF(s)")

    # --------------------------------------------------------
    # Start ONE OCR worker and ONE LLM worker.
    # --------------------------------------------------------

    ocr_worker_task = asyncio.create_task(ocr_worker())
    llm_worker_task = asyncio.create_task(llm_worker())

    # --------------------------------------------------------
    # Start PDF producer.
    # --------------------------------------------------------

    producer_task = asyncio.create_task(pdf_producer(pdf_paths))

    # Wait until all PDFs have been rendered and their pages
    # placed into the OCR queue.
    await producer_task
    print("[MAIN] All PDF pages queued.")

    # Wait until every page has been OCR'd (this also drains
    # every document into the LLM queue).
    await ocr_queue.join()
    print("[MAIN] OCR queue completed.")

    # Wait until every document's first page has been classified.
    await llm_queue.join()
    print("[MAIN] LLM queue completed.")

    # Stop workers.
    ocr_worker_task.cancel()
    llm_worker_task.cancel()

    await asyncio.gather(
        ocr_worker_task,
        llm_worker_task,
        return_exceptions=True,
    )

    print("[MAIN] Processing complete.")
    print(f"[MAIN] doc_index.json written to: {DOC_INDEX_PATH.resolve()}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        print("\nStopped by user.")