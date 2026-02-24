#!/usr/bin/env python3
"""
PDF Financial Document Processor
Splits a mixed PDF into organised cost/income/bank_statement files.
Multi-page documents are merged into a single PDF.

Usage:
    python process_financials.py <path_to_pdf>
"""

import sys
import os
import re
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
# Substring of your own company name (case-insensitive).
# Used to determine invoice perspective: buyer = cost, seller = income.
OWN_COMPANY_NAME = "NOVESYST"


# ── constants ──────────────────────────────────────────────────────────────────

TR_MONTHS = {
    "ocak": 1, "şubat": 2, "mart": 3, "nisan": 4, "mayıs": 5, "haziran": 6,
    "temmuz": 7, "ağustos": 8, "eylül": 9, "ekim": 10, "kasım": 11, "aralık": 12,
    "subat": 2, "mayis": 5, "eylul": 9, "kasim": 11, "aralik": 12,
}

MONTH_FOLDER = {
    1: "01_Ocak",   2: "02_Subat",   3: "03_Mart",    4: "04_Nisan",
    5: "05_Mayis",  6: "06_Haziran", 7: "07_Temmuz",  8: "08_Agustos",
    9: "09_Eylul",  10: "10_Ekim",   11: "11_Kasim",  12: "12_Aralik",
}

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

# Strong = exclusive to real bank statements
BANK_STRONG_KEYWORDS = [
    "hesap özeti", "hesap ozeti", "ekstre",
    "hesap hareketi", "hesap hareketleri",
    "bank statement", "account statement",
]

# Weak = can appear on invoices too — need 3+ to trigger bank_statement
# Note: "iban" intentionally excluded — it appears on all invoices for payment
BANK_WEAK_KEYWORDS = [
    "bakiye", "banka", "swift", "bic", "debit", "credit", "balance",
    "ödeme emri", "odeme emri", "havale", "eft",
    "garanti", "akbank", "ziraat", "isbank", "yapı kredi", "yapi kredi",
    "halkbank", "vakıfbank", "vakifbank", "denizbank", "fibabanka", "qnb",
]

# Keep a combined list so existing code that references BANK_KEYWORDS still works
BANK_KEYWORDS = BANK_STRONG_KEYWORDS + BANK_WEAK_KEYWORDS


# ── text extraction ────────────────────────────────────────────────────────────

def extract_text_pdfplumber(pdf_path: str, page_index: int) -> str:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return pdf.pages[page_index].extract_text() or ""
    except Exception:
        return ""


def render_page_to_image(pdf_path: str, page_index: int, dpi: int = 200) -> Image.Image:
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def extract_text_ocr(pdf_path: str, page_index: int) -> str:
    img = render_page_to_image(pdf_path, page_index, dpi=200)
    reader = get_ocr_reader()
    results = reader.readtext(np.array(img), detail=0, paragraph=True)
    return "\n".join(results)


def get_page_text(pdf_path: str, page_index: int) -> tuple[str, str]:
    native = extract_text_pdfplumber(pdf_path, page_index)
    if len(native.replace(" ", "").replace("\n", "")) >= 30:
        return native, "pdfplumber"
    return extract_text_ocr(pdf_path, page_index), "ocr"


# ── classification ─────────────────────────────────────────────────────────────

def classify_page(text: str, fatura_no: str = "") -> str:
    t = text.lower()

    # ── Rule 0: fatura no present → it's an invoice, never a bank statement ──
    # Invoices list IBAN/banka for payment, which would otherwise trigger
    # the bank keyword check. Detecting a fatura number short-circuits that.
    if not fatura_no:
        fatura_no = extract_fatura_no(text)

    is_invoice = bool(fatura_no)

    # ── Rule 1: bank statement check (skipped if we know it's an invoice) ────
    if not is_invoice:
        strong = sum(1 for kw in BANK_STRONG_KEYWORDS if kw in t)
        weak   = sum(1 for kw in BANK_WEAK_KEYWORDS   if kw in t)
        if strong >= 1 or weak >= 3:
            return "bank_statement"

    # ── Rule 2: perspective-aware invoice classification ─────────────────────
    cost_score   = sum(1 for kw in COST_KEYWORDS   if kw in t)
    income_score = sum(1 for kw in INCOME_KEYWORDS if kw in t)

    if OWN_COMPANY_NAME:
        own = OWN_COMPANY_NAME.lower()
        own_pos = t.find(own)
        sayin_match = re.search(r"say[iı]n\s+.{0,50}" + re.escape(own), t, re.IGNORECASE)
        if sayin_match:
            return "cost"
        if own_pos != -1:
            first_sayin = t.find("sayin") if "sayin" in t else t.find("sayın")
            if (first_sayin == -1 or own_pos < first_sayin) and cost_score > 0:
                return "income"

    # ── Rule 3: keyword score fallback ───────────────────────────────────────
    if cost_score == 0 and income_score == 0:
        return "unknown"
    if income_score > cost_score:
        return "income"
    return "cost"


# ── date extraction ────────────────────────────────────────────────────────────

TR_MON_PATTERN = "|".join(sorted(TR_MONTHS.keys(), key=len, reverse=True))

EN_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}


def extract_date(text: str):
    t = text.lower()

    for m in re.finditer(r"\b(\d{1,2})[./](\d{1,2})[./](20\d{2})\b", t):
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= day <= 31 and 1 <= month <= 12:
            try:
                return datetime(year, month, day), m.group(0)
            except ValueError:
                continue

    for m in re.finditer(r"\b(\d{1,2})-\s*(\d{1,2})-\s*(20\d{2})\b", t):
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= day <= 31 and 1 <= month <= 12:
            try:
                return datetime(year, month, day), m.group(0)
            except ValueError:
                continue

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
    """ASCII-fold Turkish characters (1:1 mapping – positions preserved)."""
    tbl = str.maketrans("İIĞŞÇÖÜığşçöü", "iigscoùigscou")
    return s.translate(tbl).lower()


COMPANY_SUFFIXES_NORM = re.compile(
    r"(limi?ted\s+sirketi|anonim\s+sirketi"
    r"|ltd\.?\s*sti\.?|a\.s\."
    r"|koll?\.?\s*sti\.|sahis\s+isletmesi"
    r"|inc\.?|corp\.?|llc\.?|gmbh\.?|co\.?\s*ltd\.?)"
)

_ADDRESS_SPLIT = re.compile(
    r"\s+(?:blv\.|bulvari?|cad\.|caddes[iı]?|sok\.|sokak"
    r"|mah\.|mahalles[iı]?|sites[iı]?|is\s+merkez[iı]"
    r"|no\s*:|\d{5}\s+\w|vergi\s+dairesi|merkez:|tel:|faks?:|e-posta)",
    re.IGNORECASE,
)


def extract_company(text: str) -> str:
    folded = tr_fold(text)
    m = COMPANY_SUFFIXES_NORM.search(folded)
    if m:
        candidate = text[: m.end()].strip()
        parts = [p.strip() for p in candidate.splitlines() if p.strip()]
        candidate = parts[-1] if parts else candidate
        candidate = re.sub(
            r"^(alıcı|alici|satıcı|satici|sayın|sayin|firma\s+adı|firma\s+adi"
            r"|mükellef|mukellef|unvan)\s*[:;]?\s*",
            "", candidate, flags=re.IGNORECASE,
        ).strip()
        candidate = _ADDRESS_SPLIT.split(candidate, maxsplit=1)[0].strip(" .,;:")
        if len(candidate) > 2:
            return candidate

    noise = {"fatura", "invoice", "tarih", "date", "sayin", "sayın",
             "vergi", "kdv", "toplam", "tutar", "adres", "mal", "hizmet",
             "miktar", "birim", "sira", "sıra", "aciklama", "açıklama"}
    for line in [l.strip() for l in text.splitlines() if l.strip()][:10]:
        words = line.split()
        first_word = words[0].lower() if words else ""
        if len(words) >= 2 and first_word not in noise:
            return line[:80].strip()
    return ""


def _looks_like_company(name: str) -> bool:
    """Return False if the extracted name is clearly a table cell or amount, not a company."""
    if not name:
        return False
    _non_company = {
        "mal", "hizmet", "toplam", "tutar", "kdv", "vergi", "tarih", "fatura",
        "miktar", "birim", "aciklama", "açıklama", "teslim", "plaka", "tevkifat",
        "hesaplanan", "vergiler", "odenecek", "ödenecek", "yalniz", "yalnız",
    }
    first_words = [w.lower().strip(".,;:[]()|") for w in name.split()[:3]]
    return not any(w in _non_company for w in first_words)


# ── fatura no extraction ───────────────────────────────────────────────────────

def _normalize_fatura(val: str) -> str:
    """
    Fix common OCR errors in e-fatura codes.
    Turkish e-fatura format: 2-5 letter prefix + year (4 digits) + sequence (9 digits)
    Once the first digit is encountered, any 'O' is almost certainly a misread '0'.
    """
    val = val.upper()
    result = []
    found_digit = False
    for ch in val:
        if ch.isdigit():
            found_digit = True
        # After first digit, OCR 'O' = zero
        result.append('0' if (found_digit and ch == 'O') else ch)
    return "".join(result)


def extract_fatura_no(text: str) -> str:
    """
    Extract invoice number from Turkish e-fatura documents.
    Handles OCR artifacts: leading [ brackets, lowercase o instead of 0.
    The result must contain at least 2 digits to filter out plain words.
    """
    patterns = [
        r"fatura\s+no[:\s]+[\[\(]?([A-Za-z0-9]{5,30})",
        r"invoice\s+no[:\s]+[\[\(]?([A-Za-z0-9]{5,30})",
        r"fatura\s+numaras[iı][:\s]+[\[\(]?([A-Za-z0-9]{5,30})",
        r"\b([A-Za-z]{2,5}[2Z]0\d{2}[0-9oO]{9})\b",   # EFA2026... style
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = _normalize_fatura(m.group(1).strip())
            if sum(c.isdigit() for c in val) >= 2:
                return val
    return ""


# ── tutar extraction ───────────────────────────────────────────────────────────

# Turkish number format: 1.234,56  (dot=thousands separator, comma=decimal)
_AMOUNT_RE = re.compile(r"[\d]{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?")


def extract_tutar(text: str) -> str:
    """
    Extract 'Ödenecek Tutar' (total amount payable) from the text.
    Returns a clean Turkish-format amount string e.g. '1.234,56' or ''.
    """
    folded = tr_fold(text)

    # Turkish monetary amount: digits with optional thousands dots and comma decimal
    # e.g. 1.234,56 or 234.000,00 or 155.366,40
    _amount = re.compile(r"\d{1,3}(?:\.\d{3})*,\d{2}")

    for label in [r"odenecek\s+tutar", r"toplam\s+tutar"]:
        m = re.search(label, folded)
        if not m:
            continue
        # Search for the first clean amount after the label
        after = folded[m.end():]
        am = _amount.search(after)
        if am:
            # Return the matching slice from ORIGINAL text (same positions)
            orig_start = m.end() + am.start()
            orig_end   = m.end() + am.end()
            return text[orig_start:orig_end]

    return ""


# ── IBAN extraction (for bank statement grouping) ──────────────────────────────

def extract_iban(text: str) -> str:
    m = re.search(r"\bTR\d{2}[\s\d]{15,30}\b", text, re.IGNORECASE)
    if m:
        return re.sub(r"\s+", "", m.group(0)).upper()
    return ""


# ── file naming helpers ────────────────────────────────────────────────────────

def slugify_company(name: str, max_len: int = 40) -> str:
    if not name:
        return "unknown_company"
    s = name.lower()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^\w.\u00c0-\u024f]", "", s, flags=re.UNICODE)
    s = re.sub(r"_+", "_", s)
    s = s.strip("_.")
    if len(s) > max_len:
        s = s[:max_len].rstrip("_")
    return s or "unknown_company"


def make_output_filename(date_obj, company_raw: str) -> str:
    slug = slugify_company(company_raw)
    date_str = date_obj.strftime("%d.%m.%Y") if date_obj else "undated"
    return f"{date_str}_{slug}.pdf"


def resolve_output_path(base_out: Path, doc_type: str, date_obj, filename: str) -> Path:
    if doc_type == "bank_statement":
        folder = base_out / "bank_statements"
    elif date_obj:
        folder = base_out / str(date_obj.year) / MONTH_FOLDER[date_obj.month] / doc_type
    else:
        folder = base_out / "undated" / doc_type
    folder.mkdir(parents=True, exist_ok=True)

    stem, suffix = filename.rsplit(".", 1)
    candidate = folder / filename
    counter = 2
    while candidate.exists():
        candidate = folder / f"{stem}_{counter}.{suffix}"
        counter += 1
    return candidate


# ── multi-page PDF saving ──────────────────────────────────────────────────────

def save_pages_as_pdf(src_pdf_path: str, page_indices: list[int], dest_path: Path) -> None:
    """Save one or more pages from the source PDF as a single PDF file."""
    reader = PdfReader(src_pdf_path)
    writer = PdfWriter()
    for idx in page_indices:
        writer.add_page(reader.pages[idx])
    with open(dest_path, "wb") as fh:
        writer.write(fh)


# ── document grouping ──────────────────────────────────────────────────────────

def is_continuation(current: dict, prev: dict) -> bool:
    """
    Decide whether `current` page is a continuation of the `prev` page's document.

    Rules (in order):
    1. Same non-empty fatura_no → definitely same document.
    2. Same non-empty IBAN and both bank_statement → same bank statement.
    3. Current page has no fatura_no AND no company AND no date → it is a
       content/detail continuation page (e.g. invoice item list page 2).
       Grouped with previous regardless of doc_type mismatch.
    4. Everything else → new document.
    """
    # Rule 1: matching invoice numbers
    if current["fatura_no"] and current["fatura_no"] == prev["fatura_no"]:
        return True

    # Rule 2: matching IBAN for bank statements
    if (current["doc_type"] == "bank_statement"
            and prev["doc_type"] == "bank_statement"
            and current["iban"] and current["iban"] == prev["iban"]):
        return True

    # Rule 3: no real identifying info → continuation of whatever came before
    # Also catches pages where company extraction grabbed a table cell value
    if (not current["fatura_no"]
            and not current["date_obj"]
            and not _looks_like_company(current["company"])):
        return True

    return False


# ── main processing loop ───────────────────────────────────────────────────────

def process_pdf(pdf_path: str) -> None:
    pdf_path = os.path.abspath(pdf_path)
    if not os.path.isfile(pdf_path):
        print(f"[ERROR] File not found: {pdf_path}")
        sys.exit(1)

    REPO_DIR = Path(__file__).resolve().parent
    base_out = REPO_DIR / "results"
    base_out.mkdir(parents=True, exist_ok=True)

    with pdfplumber.open(pdf_path) as _pdf:
        total_pages = len(_pdf.pages)

    print(f"\n{'='*60}")
    print(f"  Processing: {Path(pdf_path).name}")
    print(f"  Total pages: {total_pages}")
    print(f"  Output dir:  {base_out}")
    print(f"{'='*60}\n")

    log_rows  = []   # one row per SOURCE page
    cost_rows = []   # one row per DOCUMENT that is a cost
    doc_count = 0    # number of documents saved

    # ── helpers ────────────────────────────────────────────────────────────────

    def flush_group(group: list[dict]) -> None:
        """Save a group of pages as one PDF and record in logs."""
        nonlocal doc_count
        # Representative metadata comes from the first page that has data
        lead = next(
            (p for p in group if p["company"] or p["date_obj"]),
            group[0],
        )
        doc_type = lead["doc_type"]
        date_obj = lead["date_obj"]
        company  = lead["company"]
        fatura_no = lead["fatura_no"]
        tutar    = lead["tutar"] or next(
            (p["tutar"] for p in group if p["tutar"]), ""
        )

        filename = make_output_filename(date_obj, company)
        out_path = resolve_output_path(base_out, doc_type, date_obj, filename)

        page_indices = [p["index"] for p in group]
        save_pages_as_pdf(pdf_path, page_indices, out_path)
        doc_count += 1

        rel = out_path.relative_to(base_out)
        pages_str = (
            str(page_indices[0] + 1)
            if len(page_indices) == 1
            else f"{page_indices[0]+1}-{page_indices[-1]+1}"
        )

        # Log row for each source page
        for p in group:
            log_rows.append({
                "page_number":       p["index"] + 1,
                "classified_as":     doc_type,
                "extracted_date":    date_obj.strftime("%d.%m.%Y") if date_obj else "",
                "extracted_company": company,
                "fatura_no":         fatura_no,
                "tutar":             tutar,
                "pages_in_document": len(page_indices),
                "output_filename":   str(out_path.relative_to(base_out.parent)),
                "status":            "success",
                "reason":            "" if date_obj else "no date found",
            })

        # Cost Excel row
        if doc_type == "cost":
            cost_rows.append({
                "Tarih":     date_obj.strftime("%d.%m.%Y") if date_obj else "",
                "Firma":     company,
                "Tutar":     tutar,
                "Fatura No": fatura_no,
                "Proje":     "",
            })

        print(
            f"  [Doc {doc_count:>4}] pages {pages_str:<9} | "
            f"{doc_type:<16} | "
            f"date: {(date_obj.strftime('%d.%m.%Y') if date_obj else 'no date'):<12} | "
            f"company: {company[:28]:<28} | "
            f"saved: results/{rel}"
        )

    # ── page loop ──────────────────────────────────────────────────────────────

    pending: list[dict] = []

    for page_idx in range(total_pages):
        page_num = page_idx + 1
        info = {
            "index":     page_idx,
            "doc_type":  "unknown",
            "date_obj":  None,
            "company":   "",
            "fatura_no": "",
            "tutar":     "",
            "iban":      "",
            "status":    "failed",
            "reason":    "",
        }

        try:
            text, method = get_page_text(pdf_path, page_idx)

            if not text.strip():
                info["reason"] = "blank page"
                print(f"[Page {page_num:>4}/{total_pages}] BLANK – skipping")
                # Treat blank as continuation so it stays with its group
                if pending:
                    pending.append(info)
                continue

            info["fatura_no"] = extract_fatura_no(text)   # extract first
            info["doc_type"]  = classify_page(text, fatura_no=info["fatura_no"])
            info["date_obj"], _ = extract_date(text)
            info["company"]   = extract_company(text)
            info["tutar"]     = extract_tutar(text)
            info["iban"]      = extract_iban(text)
            info["status"]    = "success"

            print(
                f"[Page {page_num:>4}/{total_pages}] "
                f"{info['doc_type']:<16} | "
                f"method: {method:<11} | "
                f"fatura: {(info['fatura_no'] or '-'):<22} | "
                f"date: {(info['date_obj'].strftime('%d.%m.%Y') if info['date_obj'] else 'none'):<12} | "
                f"company: {info['company'][:25]}"
            )

        except Exception as exc:
            info["status"] = "failed"
            info["reason"] = f"{type(exc).__name__}: {exc}"
            print(f"[Page {page_num:>4}/{total_pages}] ERROR – {info['reason']}")
            traceback.print_exc(file=sys.stderr)

        # ── grouping decision ─────────────────────────────────────────────────
        if not pending:
            pending.append(info)
        elif info["status"] == "failed":
            # Failed page: log it separately, don't break current group
            log_rows.append({
                "page_number":       page_num,
                "classified_as":     "unknown",
                "extracted_date":    "",
                "extracted_company": "",
                "fatura_no":         "",
                "tutar":             "",
                "pages_in_document": 1,
                "output_filename":   "",
                "status":            "failed",
                "reason":            info["reason"],
            })
        elif is_continuation(info, pending[-1]):
            pending.append(info)
        else:
            flush_group(pending)
            pending = [info]

    if pending:
        flush_group(pending)

    # ── Write CSV log ──────────────────────────────────────────────────────────
    log_path = base_out / "processing_log.csv"
    pd.DataFrame(log_rows).to_csv(log_path, index=False, encoding="utf-8-sig")

    # ── Write costs Excel ──────────────────────────────────────────────────────
    if cost_rows:
        xl_path = base_out / "costs.xlsx"
        df_cost = pd.DataFrame(cost_rows, columns=["Tarih", "Firma", "Tutar", "Fatura No", "Proje"])

        with pd.ExcelWriter(xl_path, engine="openpyxl") as writer:
            df_cost.to_excel(writer, index=False, sheet_name="Masraflar")
            ws = writer.sheets["Masraflar"]
            # Auto-fit column widths
            for col in ws.columns:
                max_len = max((len(str(c.value)) for c in col if c.value), default=10)
                ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 50)
        print(f"\n  Costs Excel: {xl_path}")

    # ── Summary ────────────────────────────────────────────────────────────────
    df_log = pd.DataFrame(log_rows)
    success = df_log[df_log.status == "success"].shape[0]
    failed  = df_log[df_log.status == "failed"].shape[0]
    counts  = (
        df_log[df_log.status == "success"]
        .classified_as.value_counts().to_dict()
    )

    print(f"\n{'='*60}")
    print(f"  Done!  {success}/{total_pages} pages processed | {doc_count} documents saved")
    print(f"  Cost:          {counts.get('cost', 0)}")
    print(f"  Income:        {counts.get('income', 0)}")
    print(f"  Bank stmts:    {counts.get('bank_statement', 0)}")
    print(f"  Unknown:       {counts.get('unknown', 0)}")
    if failed:
        print(f"  Failed pages:  {failed}")
    print(f"  Log:           {log_path}")
    print(f"{'='*60}\n")


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("PDF Financial Document Processor")
        print("-" * 40)
        path = input("Enter the full path to the PDF file: ").strip().strip('"')
    else:
        path = sys.argv[1].strip('"')

    process_pdf(path)
