#!/usr/bin/env python3
"""
PDF Financial Document Processor — Local (no API)
Uses pdfplumber text extraction + rule-based classification.
Same output structure as process_financials.py.

Classification rules (in priority order):
  1. e-FATURA stamp or fatura_no present → it's a bill
     → "Sayın NOVESYST" under Sayın → cost
     → "Sayın [other]" under Sayın   → income
  2. No fatura_no, no Sayın, transaction rows → bank_statement
  3. No clear signal → unknown

Usage:
    python process_financials_local.py <path_to_pdf>
"""

import sys
import os
import re
import io
import shutil
import traceback
from pathlib import Path
from datetime import datetime

if sys.platform == "win32":
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

import pdfplumber
import pypdfium2 as pdfium
import pytesseract
from PIL import Image
from pypdf import PdfReader, PdfWriter
import pandas as pd


# ── config ──────────────────────────────────────────────────────────────────────

OWN_COMPANY_NAME  = "NOVESYST"
OWN_COMPANY_LOWER = OWN_COMPANY_NAME.lower()
# OCR sometimes reads 0 instead of O — cover both
RE_OWN_COMPANY = re.compile(r"n[o0]vesyst", re.IGNORECASE)

# Minimum character count from pdfplumber to trust it; below this → use OCR
MIN_TEXT_LENGTH = 50
# OCR render scale (2.0 = 144 DPI — good balance of speed vs accuracy)
OCR_SCALE = 2.0
# Tesseract language: Turkish + English
TESS_LANG = "tur+eng"
# Turkish character normalisation map (OCR often mangles these)
TR_FIXES = str.maketrans({
    "þ": "ş", "Þ": "Ş",
    "ð": "ğ", "Ð": "Ğ",
    "ý": "ı", "Ý": "İ",
})

MONTH_FOLDER = {
    1: "01_Ocak",   2: "02_Subat",   3: "03_Mart",    4: "04_Nisan",
    5: "05_Mayis",  6: "06_Haziran", 7: "07_Temmuz",  8: "08_Agustos",
    9: "09_Eylul",  10: "10_Ekim",   11: "11_Kasim",  12: "12_Aralik",
}


# ── regex patterns ───────────────────────────────────────────────────────────────

# Turkish e-fatura number: 2-5 uppercase letters + 4-digit year + 7-9 digits
RE_FATURA_NO = re.compile(
    r"\b([A-Z]{2,5}20\d{2}\d{7,9})\b"
)

# Issue date — labelled dates only (never due dates)
RE_DATE_LABELLED = re.compile(
    r"(?:fatura\s*tarihi|düzenleme\s*tarihi|belge\s*tarihi|tarih)[^\d]*"
    r"(\d{2}[./]\d{2}[./]\d{4})",
    re.IGNORECASE,
)
# Fallback: first bare DD.MM.YYYY in text
RE_DATE_BARE = re.compile(r"\b(\d{2}[./]\d{2}[./]\d{4})\b")

# "Sayın" line — the buyer is named right after this word
# Covers OCR variants: Sayin, Say1n, Sayın, Sayin:
RE_SAYIN = re.compile(r"say[i1ı][nm][:\s]+(.+)", re.IGNORECASE)

# Total amount
RE_TUTAR = re.compile(
    r"(?:ödenecek\s*tutar|genel\s*toplam|toplam\s*tutar)[^\d]*([\d.,]+)",
    re.IGNORECASE,
)

# Bank statement indicators (only meaningful when NO fatura_no present)
RE_BANK = re.compile(
    r"hesap\s*(?:özeti|ekstresi|hareketi)|banka\s*ekstresi|ekstre",
    re.IGNORECASE,
)

# e-FATURA text indicator
RE_EFATURA = re.compile(r"e[-\s]?fatura", re.IGNORECASE)

# Company suffixes (to help extract clean company names)
RE_COMPANY_SUFFIX = re.compile(
    r"[A-ZÇĞİÖŞÜa-zçğışöşü0-9 &]+(?:Ltd|A\.Ş|AŞ|Şti|A\.S\.|Inc|GmbH|San\.|Tic\.)[A-ZÇĞİÖŞÜa-zçğışöşü0-9 .]*",
    re.IGNORECASE,
)


# ── text extraction ──────────────────────────────────────────────────────────────

def ocr_page(pdf_path: str, page_idx: int) -> str:
    """Render a PDF page to image and run Tesseract OCR on it."""
    doc  = pdfium.PdfDocument(pdf_path)
    page = doc[page_idx]
    bitmap = page.render(scale=OCR_SCALE)
    pil_img = bitmap.to_pil()
    text = pytesseract.image_to_string(pil_img, lang=TESS_LANG)
    # Fix common OCR mangling of Turkish characters
    return text.translate(TR_FIXES)


def extract_text(plumber_page, pdf_path: str, page_idx: int) -> tuple:
    """
    Try pdfplumber first (fast, perfect for digital PDFs).
    If extracted text is too short (scanned page), fall back to Tesseract OCR.
    Returns (text, method) where method is 'digital' or 'ocr'.
    """
    text = plumber_page.extract_text() or ""
    if len(text.strip()) >= MIN_TEXT_LENGTH:
        return text, "digital"
    text = ocr_page(pdf_path, page_idx)
    return text, "ocr"


# ── classification helpers ───────────────────────────────────────────────────────

def find_fatura_no(text: str) -> str:
    m = RE_FATURA_NO.search(text)
    return m.group(1) if m else ""


def find_date(text: str) -> str:
    # Prefer labelled date
    m = RE_DATE_LABELLED.search(text)
    if m:
        return m.group(1).replace("/", ".")
    # Fallback: first bare date
    m = RE_DATE_BARE.search(text)
    if m:
        return m.group(1).replace("/", ".")
    return ""


def find_tutar(text: str) -> str:
    m = RE_TUTAR.search(text)
    return m.group(1).strip() if m else ""


def find_company_via_sayin(text: str) -> tuple:
    """
    Returns (doc_type, company_name) using the "Sayın" rule.
    doc_type is "cost", "income", or "" (not determinable from Sayın).
    """
    for line in text.splitlines():
        m = RE_SAYIN.search(line)
        if not m:
            continue
        buyer_raw = m.group(1).strip()
        # Remove trailing junk (tax numbers, addresses that follow on same line)
        buyer = buyer_raw.split("  ")[0].strip()
        if RE_OWN_COMPANY.search(buyer):
            return "cost", _extract_other_company(text)
        elif len(buyer) > 3:
            return "income", buyer
    return "", ""


def _extract_other_company(text: str) -> str:
    """Find the non-Novesyst company name in the text."""
    for m in RE_COMPANY_SUFFIX.finditer(text):
        name = m.group(0).strip()
        if not RE_OWN_COMPANY.search(name) and len(name) > 5:
            return name
    # Fallback: find a non-Novesyst non-empty line that looks like a name
    for line in text.splitlines():
        line = line.strip()
        if (len(line) > 5
                and not RE_OWN_COMPANY.search(line)
                and not re.match(r"^\d", line)
                and any(c.isalpha() for c in line)):
            return line
    return ""


def is_bill_page(text: str, fatura_no: str) -> bool:
    """True if the page is definitely a bill (not a bank statement)."""
    if fatura_no:
        return True
    if RE_EFATURA.search(text):
        return True
    if re.search(r"\bfatura\b", text, re.IGNORECASE):
        return True
    return False


def is_bank_statement(text: str) -> bool:
    """True only if the page looks like a genuine bank statement."""
    if not RE_BANK.search(text):
        return False
    # Must also have transaction-row pattern (date + description + amount columns)
    has_rows = bool(re.search(
        r"\d{2}[./]\d{2}[./]\d{4}.{5,60}\d[\d.,]+",
        text,
    ))
    return has_rows


def is_continuation(text: str, fatura_no: str) -> bool:
    """True if this page is a continuation of the previous document."""
    if fatura_no:
        return False
    if RE_EFATURA.search(text):
        return False
    if RE_SAYIN.search(text):
        return False
    # Looks like item rows only — no document header signals
    return True


def classify_page(text: str) -> dict:
    fatura_no = find_fatura_no(text)
    date_str  = find_date(text)
    tutar     = find_tutar(text)

    # ── Step 1: is it a bill? ─────────────────────────────────────────────────
    if is_bill_page(text, fatura_no):
        doc_type, company = find_company_via_sayin(text)

        # If Sayın gave us nothing, try harder for company name
        if not company:
            company = _extract_other_company(text)

        # If Sayın gave us no direction, default to cost (safer — less likely
        # to miscount income)
        if not doc_type:
            doc_type = "cost"

        return {
            "doc_type":        doc_type,
            "fatura_no":       fatura_no,
            "date_str":        date_str,
            "company":         company,
            "tutar":           tutar,
            "is_continuation": False,
        }

    # ── Step 2: bank statement? ───────────────────────────────────────────────
    if is_bank_statement(text):
        return {
            "doc_type":        "bank_statement",
            "fatura_no":       "",
            "date_str":        date_str,
            "company":         "",
            "tutar":           "",
            "is_continuation": False,
        }

    # ── Step 3: continuation or unknown ──────────────────────────────────────
    cont = is_continuation(text, fatura_no)
    return {
        "doc_type":        "unknown",
        "fatura_no":       fatura_no,
        "date_str":        date_str,
        "company":         "",
        "tutar":           tutar,
        "is_continuation": cont,
    }


# ── output helpers ───────────────────────────────────────────────────────────────

def parse_date(date_str: str):
    if not date_str:
        return None
    for fmt in ("%d.%m.%Y", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def make_output_filename(date_obj, fatura_no: str) -> str:
    date_str = date_obj.strftime("%d.%m.%Y") if date_obj else "undated"
    if fatura_no:
        safe = re.sub(r"[^\w.\-]", "_", fatura_no)
        return f"{date_str}_{safe}.pdf"
    return f"{date_str}.pdf"


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


def save_pages_as_pdf(src_reader: PdfReader, page_indices: list, dest_path: Path) -> None:
    writer = PdfWriter()
    for idx in page_indices:
        writer.add_page(src_reader.pages[idx])
    with open(dest_path, "wb") as fh:
        writer.write(fh)


# ── main ─────────────────────────────────────────────────────────────────────────

def process_pdf(pdf_path: str) -> None:
    pdf_path = os.path.abspath(pdf_path)
    if not os.path.isfile(pdf_path):
        print(f"[ERROR] File not found: {pdf_path}")
        sys.exit(1)

    REPO_DIR = Path(__file__).resolve().parent
    base_out = REPO_DIR / "results_local"
    if base_out.exists():
        shutil.rmtree(base_out)
        print("  Cleared previous results_local folder.")
    base_out.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)

    print(f"\n{'='*60}")
    print(f"  Processing : {Path(pdf_path).name}")
    print(f"  Pages      : {total_pages}")
    print(f"  Mode       : local (no API)")
    print(f"  Output     : {base_out}")
    print(f"{'='*60}\n")

    log_rows  = []
    cost_rows = []
    doc_count = 0
    pending: list = []

    # ── flush helper ─────────────────────────────────────────────────────────

    def flush_group(group: list) -> None:
        nonlocal doc_count

        lead = next(
            (p for p in group if p["fatura_no"] or p["date_obj"]),
            group[0],
        )

        doc_type  = lead["doc_type"] if lead["doc_type"] != "unknown" else group[0]["doc_type"]
        date_obj  = lead["date_obj"]
        fatura_no = lead["fatura_no"]
        company   = lead["company"]
        tutar     = lead["tutar"] or next((p["tutar"] for p in group if p["tutar"]), "")

        filename = make_output_filename(date_obj, fatura_no)
        out_path = resolve_output_path(base_out, doc_type, date_obj, filename)

        page_indices = [p["index"] for p in group]
        save_pages_as_pdf(reader, page_indices, out_path)
        doc_count += 1

        rel = out_path.relative_to(base_out)
        pages_str = (
            str(page_indices[0] + 1)
            if len(page_indices) == 1
            else f"{page_indices[0]+1}-{page_indices[-1]+1}"
        )

        print(
            f"  [Doc {doc_count:>4}] pages {pages_str:<9} | "
            f"{doc_type:<16} | "
            f"date: {(date_obj.strftime('%d.%m.%Y') if date_obj else 'no date'):<12} | "
            f"fatura: {(fatura_no or '-'):<24} | "
            f"results_local/{rel}"
        )

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
            })

        if doc_type == "cost":
            cost_rows.append({
                "Tarih":     date_obj.strftime("%d.%m.%Y") if date_obj else "",
                "Firma":     company,
                "Tutar":     tutar,
                "Fatura No": fatura_no,
                "Proje":     "",
            })

    # ── page loop ─────────────────────────────────────────────────────────────

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx in range(total_pages):
            page_num = page_idx + 1
            info = {
                "index":           page_idx,
                "doc_type":        "unknown",
                "date_obj":        None,
                "company":         "",
                "fatura_no":       "",
                "tutar":           "",
                "is_continuation": False,
                "status":          "failed",
            }

            try:
                text, method = extract_text(pdf.pages[page_idx], pdf_path, page_idx)
                result = classify_page(text)

                info["doc_type"]        = result["doc_type"]
                info["fatura_no"]       = result["fatura_no"]
                info["date_obj"]        = parse_date(result["date_str"])
                info["company"]         = result["company"]
                info["tutar"]           = result["tutar"]
                info["is_continuation"] = result["is_continuation"]
                info["status"]          = "success"

                cont_tag  = " <CONT>" if result["is_continuation"] else ""
                ocr_tag   = " [OCR]"  if method == "ocr"            else ""
                print(
                    f"[Page {page_num:>4}/{total_pages}] "
                    f"{info['doc_type']:<16} | "
                    f"fatura={info['fatura_no'] or '-':<24} | "
                    f"date={info['date_obj'].strftime('%d.%m.%Y') if info['date_obj'] else 'none':<12}"
                    f"{cont_tag}{ocr_tag}"
                )

            except Exception as exc:
                info["status"] = "failed"
                print(f"[Page {page_num:>4}/{total_pages}] ERROR – {exc}")
                traceback.print_exc(file=sys.stderr)

            # ── grouping ──────────────────────────────────────────────────────
            if not pending:
                pending.append(info)

            elif info["status"] == "failed":
                log_rows.append({
                    "page_number": page_num, "classified_as": "unknown",
                    "extracted_date": "", "extracted_company": "",
                    "fatura_no": "", "tutar": "", "pages_in_document": 1,
                    "output_filename": "", "status": "failed",
                })

            elif info["is_continuation"]:
                pending.append(info)

            else:
                group_fatura = next((p["fatura_no"] for p in pending if p["fatura_no"]), "")
                if info["fatura_no"] and info["fatura_no"] == group_fatura:
                    pending.append(info)
                else:
                    flush_group(pending)
                    pending = [info]

    if pending:
        flush_group(pending)

    # ── write outputs ─────────────────────────────────────────────────────────

    log_path = base_out / "processing_log.csv"
    pd.DataFrame(log_rows).to_csv(log_path, index=False, encoding="utf-8-sig")

    if cost_rows:
        xl_path = base_out / "costs.xlsx"
        df_cost = pd.DataFrame(cost_rows, columns=["Tarih", "Firma", "Tutar", "Fatura No", "Proje"])
        with pd.ExcelWriter(xl_path, engine="openpyxl") as xw:
            df_cost.to_excel(xw, index=False, sheet_name="Masraflar")
            ws = xw.sheets["Masraflar"]
            for col in ws.columns:
                max_len = max((len(str(c.value)) for c in col if c.value), default=10)
                ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 50)
        print(f"\n  Costs Excel : {xl_path}")

    df_log  = pd.DataFrame(log_rows)
    success = (df_log.status == "success").sum() if "status" in df_log.columns else 0
    failed  = (df_log.status == "failed").sum()  if "status" in df_log.columns else 0
    counts  = (
        df_log[df_log.status == "success"].classified_as.value_counts().to_dict()
        if success > 0 else {}
    )

    print(f"\n{'='*60}")
    print(f"  Done!  {success}/{total_pages} pages | {doc_count} documents saved")
    print(f"  Cost:         {counts.get('cost', 0)}")
    print(f"  Income:       {counts.get('income', 0)}")
    print(f"  Bank stmts:   {counts.get('bank_statement', 0)}")
    print(f"  Unknown:      {counts.get('unknown', 0)}")
    if failed:
        print(f"  Failed:       {failed}")
    print(f"  Log:          {log_path}")
    print(f"{'='*60}\n")


# ── entry point ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("PDF Financial Document Processor (local)")
        print("-" * 40)
        path = input("Enter the full path to the PDF file: ").strip().strip('"')
    else:
        path = sys.argv[1].strip('"')

    process_pdf(path)
