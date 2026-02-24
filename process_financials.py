#!/usr/bin/env python3
"""
PDF Financial Document Processor
Splits a mixed PDF into organised cost/income/bank_statement files.

Usage:
    python process_financials.py <path_to_pdf>
"""

import sys
import os
import re
import csv
import io
import time
import unicodedata
import traceback
from pathlib import Path
from datetime import datetime

# Force UTF-8 output on Windows (avoids UnicodeEncodeError for Turkish chars)
if sys.platform == "win32":
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── third-party ────────────────────────────────────────────────────────────────
import fitz                          # PyMuPDF – PDF rendering
import pdfplumber                    # text-based extraction (fast path)
from pypdf import PdfReader, PdfWriter
import pandas as pd
import numpy as np
from PIL import Image

# EasyOCR (lazy-loaded – first use downloads model files ~200 MB)
_ocr_reader = None

def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        print("[OCR] Initialising EasyOCR (Turkish + English) – this may take a minute on first run …")
        _ocr_reader = easyocr.Reader(["tr", "en"], gpu=False, verbose=False)
        print("[OCR] EasyOCR ready.")
    return _ocr_reader


# ── user config ────────────────────────────────────────────────────────────────
# Set this to a substring of your own company name (case-insensitive).
# When found after "SAYIN" on an invoice, the doc is classified as COST.
# When your company is the issuer (appears first), it is classified as INCOME.
# Leave empty ("") to rely solely on keyword scoring.
OWN_COMPANY_NAME = "NOVESYST"

# ── constants ──────────────────────────────────────────────────────────────────

TR_MONTHS = {
    "ocak": 1, "şubat": 2, "mart": 3, "nisan": 4, "mayıs": 5, "haziran": 6,
    "temmuz": 7, "ağustos": 8, "eylül": 9, "ekim": 10, "kasım": 11, "aralık": 12,
    # ascii-folded variants (for OCR misreads)
    "subat": 2, "mayis": 5, "haziran": 6, "eylul": 9, "kasim": 11, "aralik": 12,
}

MONTH_FOLDER = {
    1: "01_Ocak",   2: "02_Subat",   3: "03_Mart",    4: "04_Nisan",
    5: "05_Mayis",  6: "06_Haziran", 7: "07_Temmuz",  8: "08_Agustos",
    9: "09_Eylul",  10: "10_Ekim",   11: "11_Kasim",  12: "12_Aralik",
}

# Classification keywords (lower-case Turkish + English)
COST_KEYWORDS = [
    "fatura", "invoice", "borç", "borc", "ödenecek tutar", "odenecek tutar",
    "kdv dahil toplam", "expense", "gider", "mal bedeli", "hizmet bedeli",
    "alıcı", "alici", "satıcı", "satici", "vergi no", "vkn", "tckn",
    "e-fatura", "e-arsiv", "e-arşiv", "efatura",
]

INCOME_KEYWORDS = [
    "tahsilat", "gelir", "alınan ödeme", "alinan odeme", "receipt", "income",
    "makbuz", "irsaliye", "tahsilat makbuzu", "ödeme makbuzu", "odeme makbuzu",
    "alındı", "alindi",
]

BANK_KEYWORDS = [
    "hesap özeti", "hesap ozeti", "ekstre", "bank statement", "banka",
    "iban", "bakiye", "hesap hareketi", "hesap hareketleri",
    "swift", "bic", "debit", "credit", "balance", "account statement",
    "garanti", "akbank", "ziraat", "isbank", "yapı kredi", "yapi kredi",
    "halkbank", "vakıfbank", "vakifbank", "denizbank", "fibabanka", "qnb",
    "ödeme emri", "odeme emri", "havale", "eft",
]


# ── text extraction ────────────────────────────────────────────────────────────

def extract_text_pdfplumber(pdf_path: str, page_index: int) -> str:
    """Try pdfplumber for native-text PDFs (fast, exact)."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_index]
            txt = page.extract_text() or ""
            return txt
    except Exception:
        return ""


def render_page_to_image(pdf_path: str, page_index: int, dpi: int = 200) -> Image.Image:
    """Render a PDF page to a PIL Image using PyMuPDF."""
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    clip = page.rect
    pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def extract_text_ocr(pdf_path: str, page_index: int) -> str:
    """Render page and run EasyOCR."""
    img = render_page_to_image(pdf_path, page_index, dpi=200)
    reader = get_ocr_reader()
    img_np = np.array(img)
    results = reader.readtext(img_np, detail=0, paragraph=True)
    return "\n".join(results)


def get_page_text(pdf_path: str, page_index: int) -> tuple[str, str]:
    """
    Returns (text, method) where method is 'pdfplumber' or 'ocr'.
    Falls back to OCR when native text is sparse (<30 meaningful chars).
    """
    native = extract_text_pdfplumber(pdf_path, page_index)
    # Count non-whitespace chars
    if len(native.replace(" ", "").replace("\n", "")) >= 30:
        return native, "pdfplumber"
    # Fallback to OCR
    ocr_text = extract_text_ocr(pdf_path, page_index)
    return ocr_text, "ocr"


# ── classification ─────────────────────────────────────────────────────────────

def classify_page(text: str) -> str:
    """
    Returns 'cost', 'income', 'bank_statement', or 'unknown'.

    Order of precedence:
      1. Bank keywords dominate (bank statements mention IBAN, bakiye, ekstre, etc.)
      2. If OWN_COMPANY_NAME is configured, use invoice perspective:
           - "SAYIN <OWN_COMPANY>" → we are the BUYER → COST
           - OWN_COMPANY at top (before first SAYIN) → we are the SELLER → INCOME
      3. Fall back to keyword scoring.
    """
    t = text.lower()

    bank_score   = sum(1 for kw in BANK_KEYWORDS   if kw in t)
    cost_score   = sum(1 for kw in COST_KEYWORDS   if kw in t)
    income_score = sum(1 for kw in INCOME_KEYWORDS if kw in t)

    # ── 1. Bank statement check ───────────────────────────────────────────────
    if bank_score >= 2:
        return "bank_statement"
    if bank_score > max(cost_score, income_score):
        return "bank_statement"

    # ── 2. Perspective-aware invoice classification ───────────────────────────
    if OWN_COMPANY_NAME:
        own = OWN_COMPANY_NAME.lower()
        # Find position of own company name in text
        own_pos = t.find(own)
        # Find position of "SAYIN <own company>" pattern
        sayin_match = re.search(
            r"say[iı]n\s+.{0,50}" + re.escape(own), t, re.IGNORECASE
        )
        if sayin_match:
            return "cost"   # We are addressed as buyer → expense for us
        if own_pos != -1:
            # Check if our company appears before the first "SAYIN" (i.e. we are issuer)
            first_sayin = t.find("sayin") if "sayin" in t else t.find("sayın")
            if first_sayin == -1 or own_pos < first_sayin:
                # We appear before any "SAYIN" → we are the issuer/seller → income
                if cost_score > 0:   # document is an invoice of some sort
                    return "income"

    # ── 3. Keyword score fallback ─────────────────────────────────────────────
    if cost_score == 0 and income_score == 0:
        return "unknown"
    if income_score > cost_score:
        return "income"
    if cost_score > income_score:
        return "cost"
    # tie-break: Turkish e-fatura defaults to cost if ambiguous
    return "cost"


# ── date extraction ────────────────────────────────────────────────────────────

# Turkish month names for pattern building
TR_MON_PATTERN = "|".join(sorted(TR_MONTHS.keys(), key=len, reverse=True))

DATE_PATTERNS = [
    # DD.MM.YYYY or DD/MM/YYYY
    r"\b(\d{1,2})[./](\d{1,2})[./](20\d{2})\b",
    # DD Month YYYY (Turkish written month)
    rf"\b(\d{{1,2}})\s+({TR_MON_PATTERN})\s+(20\d{{2}})\b",
    # Month DD YYYY (English written month)
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s+(\d{1,2})[,\s]+(20\d{2})\b",
    # YYYY-MM-DD (ISO)
    r"\b(20\d{2})-(\d{2})-(\d{2})\b",
]

EN_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}


def extract_date(text: str):
    """
    Returns (datetime | None, raw_string | "").
    Tries multiple date patterns against the full text.
    """
    t = text.lower()

    # Pattern 1: numeric DD.MM.YYYY / DD/MM/YYYY
    for m in re.finditer(r"\b(\d{1,2})[./](\d{1,2})[./](20\d{2})\b", t):
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= day <= 31 and 1 <= month <= 12:
            try:
                return datetime(year, month, day), m.group(0)
            except ValueError:
                continue

    # Pattern 2: DD-MM-YYYY (dashes, common in Turkish e-fatura OCR output)
    # Also handles OCR artefacts like "02- 01- 2026" with spaces after dashes
    for m in re.finditer(r"\b(\d{1,2})-\s*(\d{1,2})-\s*(20\d{2})\b", t):
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= day <= 31 and 1 <= month <= 12:
            try:
                return datetime(year, month, day), m.group(0)
            except ValueError:
                continue

    # Pattern 3: Turkish written month  e.g. "15 Haziran 2024"
    pat3 = rf"\b(\d{{1,2}})\s+({TR_MON_PATTERN})\s+(20\d{{2}})\b"
    for m in re.finditer(pat3, t):
        day = int(m.group(1))
        month = TR_MONTHS.get(m.group(2).lower())
        year = int(m.group(3))
        if month and 1 <= day <= 31:
            try:
                return datetime(year, month, day), m.group(0)
            except ValueError:
                continue

    # Pattern 4: English written month
    pat4 = (r"\b(january|february|march|april|may|june|july|august|"
            r"september|october|november|december)\s+(\d{1,2})[,\s]+(20\d{2})\b")
    for m in re.finditer(pat4, t):
        month = EN_MONTHS.get(m.group(1))
        day = int(m.group(2))
        year = int(m.group(3))
        if month:
            try:
                return datetime(year, month, day), m.group(0)
            except ValueError:
                continue

    # Pattern 5: ISO YYYY-MM-DD (only as a last resort – comes after DD-MM-YYYY
    # to avoid misidentifying "13-01-2026" as year=2026 when the long form is searched)
    for m in re.finditer(r"\b(20\d{2})-(\d{2})-(\d{2})\b", t):
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            try:
                return datetime(year, month, day), m.group(0)
            except ValueError:
                continue

    return None, ""


# ── company extraction ─────────────────────────────────────────────────────────

def tr_fold(s: str) -> str:
    """
    ASCII-fold Turkish characters and lowercase for regex matching.
    Character-for-character (1:1) so string positions are preserved.
    This is needed because Python's re.IGNORECASE does not handle
    Turkish dotted İ (U+0130) / dotless ı (U+0131) correctly.
    """
    tbl = str.maketrans(
        "İIĞŞÇÖÜığşçöü",
        "iigscoùigscou",   # approximate — good enough for company suffix matching
    )
    # Replace Turkish chars first, then standard lower()
    return s.translate(tbl).lower()


# Company suffixes – written in tr_fold()-normalized form (ASCII, lowercase)
COMPANY_SUFFIXES_NORM = re.compile(
    r"(limi?ted\s+sirketi"        # Limited Şirketi → limited sirketi
    r"|anonim\s+sirketi"          # Anonim Şirketi
    r"|ltd\.?\s*sti\.?"           # Ltd. Şti.
    r"|a\.s\."                    # A.Ş.
    r"|koll?\.?\s*sti\."          # Koll. Şti.
    r"|sahis\s+isletmesi"         # Şahıs İşletmesi
    r"|inc\.?|corp\.?|llc\.?|gmbh\.?|co\.?\s*ltd\.?)"
)

# Label prefixes that appear before company names in Turkish invoices
_LABEL_PREFIX = re.compile(
    r"^(alici|satici|sayin|firma\s+adi|mukellef|unvan|sirket)\s*[:;]?\s*",
    re.IGNORECASE,
)

# Address-start markers — the company name ends before any of these
_ADDRESS_SPLIT = re.compile(
    r"\s+(?:blv\.|bulvari?|cad\.|caddes[iı]?|sok\.|sokak"
    r"|mah\.|mahalles[iı]?|sites[iı]?|is\s+merkez[iı]"
    r"|no\s*:|\d{5}\s+\w|vergi\s+dairesi|merkez:|tel:|faks?:|e-posta)",
    re.IGNORECASE,
)


def extract_company(text: str) -> str:
    """
    Heuristic company name extractor for Turkish financial docs.

    Strategy: search tr_fold(text) for the FIRST legal-entity suffix
    (ltd. şti., a.ş., limited şirketi, etc.) and slice the ORIGINAL
    text up to that position.  This is robust against EasyOCR paragraph
    mode (which joins company name + address into one long string) and
    against Python's re.IGNORECASE not handling Turkish Ş/İ properly.
    """
    folded = tr_fold(text)
    m = COMPANY_SUFFIXES_NORM.search(folded)
    if m:
        # Slice original text up to end of suffix (positions match 1:1)
        candidate = text[: m.end()].strip()

        # If multi-line, take only the LAST segment (has the company + suffix)
        parts = [p.strip() for p in candidate.splitlines() if p.strip()]
        candidate = parts[-1] if parts else candidate

        # Strip any label prefix ("Alıcı: ...", "SAYIN ...")
        candidate = _LABEL_PREFIX.sub("", tr_fold(candidate))
        # Re-extract from original at the same cleaned length
        # (simpler: just re-strip the label from the original too)
        candidate = _LABEL_PREFIX.sub(
            "", candidate, count=1
        ).strip()
        # Restore original-cased version
        candidate = text[: m.end()].strip()
        parts = [p.strip() for p in candidate.splitlines() if p.strip()]
        candidate = parts[-1] if parts else candidate
        # Remove label prefix from original-cased
        candidate = re.sub(
            r"^(alıcı|alici|satıcı|satici|sayın|sayin|firma\s+adı|firma\s+adi"
            r"|mükellef|mukellef|unvan)\s*[:;]?\s*",
            "", candidate, flags=re.IGNORECASE,
        ).strip()

        # Finally clip at the first address-start marker
        candidate = _ADDRESS_SPLIT.split(candidate, maxsplit=1)[0].strip(" .,;:")

        if len(candidate) > 2:
            return candidate

    # Fallback: first line that doesn't start with noise words
    noise = {"fatura", "invoice", "tarih", "date", "sayin", "sayın",
             "vergi", "kdv", "toplam", "tutar", "adres"}
    for line in [l.strip() for l in text.splitlines() if l.strip()][:10]:
        first_word = line.split()[0].lower() if line.split() else ""
        if len(line.split()) >= 2 and first_word not in noise:
            return line[:80].strip()

    return ""


# ── file naming helpers ────────────────────────────────────────────────────────

def slugify_company(name: str, max_len: int = 40) -> str:
    """Lowercase, spaces→underscore, keep Turkish chars, cap at max_len chars."""
    if not name:
        return "unknown_company"
    s = name.lower()
    # replace runs of whitespace / hyphens with underscore
    s = re.sub(r"[\s\-]+", "_", s)
    # keep letters (including Turkish/Latin-Extended), digits, underscore, dot
    s = re.sub(r"[^\w.\u00c0-\u024f]", "", s, flags=re.UNICODE)
    # collapse multiple underscores/dots
    s = re.sub(r"_+", "_", s)
    s = s.strip("_.")
    # Cap length at word boundary to avoid cutting mid-word
    if len(s) > max_len:
        s = s[:max_len].rstrip("_")
    return s or "unknown_company"


def make_output_filename(date_obj, company_raw: str) -> str:
    company_slug = slugify_company(company_raw)
    if date_obj:
        date_str = date_obj.strftime("%d.%m.%Y")
    else:
        date_str = "undated"
    return f"{date_str}_{company_slug}.pdf"


# ── folder structure ───────────────────────────────────────────────────────────

def resolve_output_path(
    base_out: Path,
    doc_type: str,
    date_obj,
    filename: str,
) -> Path:
    """Return full path for output file (creates dirs as needed)."""
    if doc_type == "bank_statement":
        folder = base_out / "bank_statements"
    elif date_obj:
        year_folder  = str(date_obj.year)
        month_folder = MONTH_FOLDER[date_obj.month]
        folder = base_out / year_folder / month_folder / doc_type
    else:
        folder = base_out / "undated" / doc_type

    folder.mkdir(parents=True, exist_ok=True)

    # Handle duplicate filenames by appending _2, _3, …
    stem, suffix = filename.rsplit(".", 1)
    candidate = folder / filename
    counter = 2
    while candidate.exists():
        candidate = folder / f"{stem}_{counter}.{suffix}"
        counter += 1
    return candidate


# ── save single page as PDF ────────────────────────────────────────────────────

def save_page_as_pdf(src_pdf_path: str, page_index: int, dest_path: Path) -> None:
    """Extract one page from the source PDF and save it as a new PDF file."""
    reader = PdfReader(src_pdf_path)
    writer = PdfWriter()
    writer.add_page(reader.pages[page_index])
    with open(dest_path, "wb") as fh:
        writer.write(fh)


# ── main processing loop ───────────────────────────────────────────────────────

def process_pdf(pdf_path: str) -> None:
    pdf_path = os.path.abspath(pdf_path)
    if not os.path.isfile(pdf_path):
        print(f"[ERROR] File not found: {pdf_path}")
        sys.exit(1)

    # Results always land in results/ inside the repo (next to this script)
    REPO_DIR = Path(__file__).resolve().parent
    base_out = REPO_DIR / "results"
    base_out.mkdir(parents=True, exist_ok=True)

    log_rows = []

    # Count pages
    with pdfplumber.open(pdf_path) as pdf_check:
        total_pages = len(pdf_check.pages)

    print(f"\n{'='*60}")
    print(f"  Processing: {Path(pdf_path).name}")
    print(f"  Total pages: {total_pages}")
    print(f"  Output dir:  {base_out}")
    print(f"{'='*60}\n")

    for page_idx in range(total_pages):
        page_num = page_idx + 1
        row = {
            "page_number":       page_num,
            "classified_as":     "unknown",
            "extracted_date":    "",
            "extracted_company": "",
            "output_filename":   "",
            "status":            "failed",
            "reason":            "",
        }

        try:
            # ── 1. Text extraction ────────────────────────────────────────────
            text, method = get_page_text(pdf_path, page_idx)

            if not text.strip():
                row["reason"] = "text extraction failed – blank page"
                print(f"[Page {page_num:>4}/{total_pages}] BLANK – skipping")
                log_rows.append(row)
                continue

            # ── 2. Classification ─────────────────────────────────────────────
            doc_type = classify_page(text)
            row["classified_as"] = doc_type

            # ── 3. Date + company ─────────────────────────────────────────────
            date_obj, date_raw = extract_date(text)
            company_raw        = extract_company(text)

            row["extracted_date"]    = date_obj.strftime("%d.%m.%Y") if date_obj else ""
            row["extracted_company"] = company_raw

            # ── 4. Output filename + path ─────────────────────────────────────
            filename   = make_output_filename(date_obj, company_raw)
            out_path   = resolve_output_path(base_out, doc_type, date_obj, filename)

            # ── 5. Save page ──────────────────────────────────────────────────
            save_page_as_pdf(pdf_path, page_idx, out_path)

            rel_path = out_path.relative_to(base_out)
            row["output_filename"] = str(out_path.relative_to(base_out.parent))
            row["status"]          = "success"

            # ── Console output ────────────────────────────────────────────────
            date_display    = row["extracted_date"] or "no date"
            company_display = company_raw[:30]       or "no company"
            print(
                f"[Page {page_num:>4}/{total_pages}] "
                f"classified as: {doc_type:<16} | "
                f"method: {method:<11} | "
                f"date: {date_display:<12} | "
                f"company: {company_display:<30} | "
                f"saved: results/{rel_path}"
            )

            if doc_type == "unknown":
                row["reason"] = "could not classify document type"
            elif not date_obj:
                row["reason"] = "no date found"

        except Exception as exc:
            row["status"] = "failed"
            row["reason"] = f"{type(exc).__name__}: {exc}"
            print(f"[Page {page_num:>4}/{total_pages}] ERROR – {row['reason']}")
            # Full traceback to stderr for debugging
            traceback.print_exc(file=sys.stderr)

        log_rows.append(row)

    # ── Write CSV log ──────────────────────────────────────────────────────────
    log_path = base_out / "processing_log.csv"
    df = pd.DataFrame(log_rows)
    df.to_csv(log_path, index=False, encoding="utf-8-sig")  # utf-8-sig for Excel compat

    # ── Summary ────────────────────────────────────────────────────────────────
    success  = df[df.status == "success"].shape[0]
    failed   = df[df.status == "failed"].shape[0]
    counts   = df[df.status == "success"].classified_as.value_counts().to_dict()

    print(f"\n{'='*60}")
    print(f"  Done!  {success}/{total_pages} pages processed successfully.")
    print(f"  Cost:          {counts.get('cost', 0)}")
    print(f"  Income:        {counts.get('income', 0)}")
    print(f"  Bank stmts:    {counts.get('bank_statement', 0)}")
    print(f"  Unknown:       {counts.get('unknown', 0)}")
    if failed:
        print(f"  Failed:        {failed}")
    print(f"  Log:           {log_path}")
    print(f"{'='*60}\n")


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Interactive fallback – ask for path
        print("PDF Financial Document Processor")
        print("-" * 40)
        path = input("Enter the full path to the PDF file: ").strip().strip('"')
    else:
        path = sys.argv[1].strip('"')

    process_pdf(path)
