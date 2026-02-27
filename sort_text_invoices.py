#!/usr/bin/env python3
"""
Text PDF Invoice Sorter
Sorts individual text-based PDF invoices (faturalar) into year/month folders
by Düzenleme Tarihi. Keeps original filenames.

Generates an Excel summary with: Firma, Tutar, Düzenleme Tarihi, Fatura No.

Usage:
    python sort_text_invoices.py <folder_path>
    python sort_text_invoices.py                   # interactive prompt
"""

import sys
import os
import re
import shutil
import traceback
from pathlib import Path
from datetime import datetime

if sys.platform == "win32":
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

import pdfplumber
import pandas as pd


# ── config ──────────────────────────────────────────────────────────────────────

RE_OWN_COMPANY = re.compile(r"n[o0]vesyst", re.IGNORECASE)

MONTH_FOLDER = {
    1: "01_Ocak",   2: "02_Subat",   3: "03_Mart",    4: "04_Nisan",
    5: "05_Mayis",  6: "06_Haziran", 7: "07_Temmuz",  8: "08_Agustos",
    9: "09_Eylul",  10: "10_Ekim",   11: "11_Kasim",  12: "12_Aralik",
}


# ── regex patterns ──────────────────────────────────────────────────────────────

# Turkish e-fatura number: 2-5 uppercase letters + optional digits + year + sequence
RE_FATURA_NO = re.compile(r"\b([A-Z]{2,5}\d{0,2}20\d{2}\d{7,9})\b")

# Düzenleme Tarihi / Fatura Tarihi — allows optional spaces around separators
# e.g. "25- 11- 2025", "25.11.2025", "25-11-2025"
RE_DATE_LABELLED = re.compile(
    r"(?:düzenleme\s*tarihi|fatura\s*tarihi|belge\s*tarihi|tarih)[^\d]*"
    r"(\d{2})\s*[./-]\s*(\d{2})\s*[./-]\s*(\d{4})",
    re.IGNORECASE,
)
# YYYY-MM-DD format (ISO) after a label
RE_DATE_LABELLED_ISO = re.compile(
    r"(?:düzenleme\s*tarihi|fatura\s*tarihi|belge\s*tarihi|tarih)[^\d]*"
    r"(\d{4})\s*[./-]\s*(\d{2})\s*[./-]\s*(\d{2})",
    re.IGNORECASE,
)
RE_DATE_BARE = re.compile(r"\b(\d{2})\s*[./-]\s*(\d{2})\s*[./-]\s*(\d{4})\b")

# Total amount (ödenecek tutar)
RE_TUTAR = re.compile(
    r"(?:ödenecek\s*tutar|genel\s*toplam|toplam\s*tutar)[^\d]*([\d.,]+)",
    re.IGNORECASE,
)

# Legal entity endings in Turkish e-fatura (word-boundary to avoid false matches)
RE_LEGAL_ENTITY = re.compile(
    r"\bŞİRKETİ\b|\bŞTİ\.?(?:\s|$)|\bLTD\.?(?:\s|$)|\bA\.Ş\.?(?:\s|$)|\bAŞ\b|\bANONİM\b|\bLİMİTED\b",
    re.IGNORECASE,
)


# ── extraction ──────────────────────────────────────────────────────────────────

# Unicode dashes to normalise (U+2010 hyphen, U+2011 non-breaking hyphen, U+2013 en-dash)
_DASH_MAP = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-"})


def extract_full_text(pdf_path: str) -> str:
    texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            texts.append(page.extract_text() or "")
    return "\n".join(texts).translate(_DASH_MAP)


def find_fatura_no(text: str) -> str:
    m = RE_FATURA_NO.search(text)
    return m.group(1) if m else ""


def _format_date_match(m) -> str:
    """Format a date regex match (3 groups: DD, MM, YYYY) and validate year."""
    dd, mm, yyyy = m.group(1), m.group(2), m.group(3)
    year = int(yyyy)
    if year < 2020 or year > 2030:
        return ""
    return f"{dd}.{mm}.{yyyy}"


def find_date(text: str) -> str:
    # Prefer labelled date DD.MM.YYYY
    for m in RE_DATE_LABELLED.finditer(text):
        result = _format_date_match(m)
        if result:
            return result
    # Try labelled ISO format YYYY-MM-DD
    for m in RE_DATE_LABELLED_ISO.finditer(text):
        yyyy, mm, dd = m.group(1), m.group(2), m.group(3)
        year = int(yyyy)
        if 2020 <= year <= 2030:
            return f"{dd}.{mm}.{yyyy}"
    # Fallback: first bare date with valid year
    for m in RE_DATE_BARE.finditer(text):
        result = _format_date_match(m)
        if result:
            return result
    return ""


def find_tutar(text: str) -> str:
    m = RE_TUTAR.search(text)
    return m.group(1).strip() if m else ""


def find_company(text: str) -> str:
    """
    Find the seller company name from the e-fatura header.
    The seller name appears in the first lines, before 'Sayın' / 'ALICI'.
    """
    lines = text.splitlines()

    # Find where the buyer section starts (everything before is the seller header)
    cutoff = len(lines)
    for i, line in enumerate(lines):
        lower = line.strip().lower()
        if lower.startswith("sayın") or lower.startswith("sayin") or lower == "alici" or lower == "alıcı":
            cutoff = i
            break

    # In the seller header, find the line with the legal entity name
    header_lines = lines[:cutoff]
    for i, line in enumerate(header_lines):
        stripped = line.strip()
        if not stripped:
            continue
        if RE_OWN_COMPANY.search(stripped):
            continue
        if RE_LEGAL_ENTITY.search(stripped):
            # Check if a previous non-empty line is part of the name
            # (e.g. "AÇI HİDROLİK MAKİNA İMALAT SANAYİ VE" + "TİCARET LİMİTED ŞİRKETİ")
            for prev_idx in range(i - 1, max(i - 4, -1), -1):
                prev = header_lines[prev_idx].strip()
                if not prev:
                    continue
                if RE_OWN_COMPANY.search(prev):
                    break
                # Skip noise lines and keep looking further back
                if re.match(r"^(e-?Fatura|e-?FATURA|Tel|Fax|Web|E-?Posta|Vergi|VKN|Bu Fatura|\d{5})", prev, re.IGNORECASE):
                    continue
                if "@" in prev or "www." in prev.lower():
                    continue
                if re.search(r"\bMAH\b|\bCAD\b|\bOSB\b", prev, re.IGNORECASE):
                    continue
                # Previous line looks like name continuation
                if prev[-1] not in ".:":
                    return f"{prev} {stripped}"
                break
            return stripped

    # Fallback: sole proprietors (no legal entity suffix) — first name-like line
    for line in header_lines:
        stripped = line.strip()
        if not stripped or len(stripped) < 3:
            continue
        if RE_OWN_COMPANY.search(stripped):
            continue
        if re.match(r"^(e-?Fatura|e-?FATURA|Tel|Fax|Web|E-?Posta|Vergi|VKN|Bu Fatura|\d{5})", stripped, re.IGNORECASE):
            continue
        if "@" in stripped or "www." in stripped.lower():
            continue
        if re.search(r"\bMAH\b|\bCAD\b|\bOSB\b|\bNO\b:", stripped, re.IGNORECASE):
            continue
        if re.match(r"^[\d/\-.()+\s]+$", stripped):
            continue
        # Looks like a name (has letters, not an address)
        if any(c.isalpha() for c in stripped):
            return stripped

    return ""


def parse_date(date_str: str):
    if not date_str:
        return None
    for fmt in ("%d.%m.%Y", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


# ── main ────────────────────────────────────────────────────────────────────────

def process_folder(folder_path: str) -> None:
    folder_path = os.path.abspath(folder_path)
    if not os.path.isdir(folder_path):
        print(f"[ERROR] Folder not found: {folder_path}")
        sys.exit(1)

    pdf_files = sorted(
        p for p in Path(folder_path).rglob("*.pdf")
        if not p.name.startswith(".")
    )

    if not pdf_files:
        print(f"[ERROR] No PDF files found in: {folder_path}")
        sys.exit(1)

    REPO_DIR = Path(__file__).resolve().parent
    base_out = REPO_DIR / "results"
    if base_out.exists():
        shutil.rmtree(base_out)
        print("  Cleared previous results folder.")
    base_out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Sorting text PDF invoices by date")
    print(f"  Source : {folder_path}")
    print(f"  Files  : {len(pdf_files)}")
    print(f"  Output : {base_out}")
    print(f"{'='*60}\n")

    excel_rows = []
    success_count = 0
    failed_count  = 0

    for i, pdf_file in enumerate(pdf_files, 1):
        try:
            text = extract_full_text(str(pdf_file))

            if len(text.strip()) < 30:
                print(f"[{i:>4}/{len(pdf_files)}] {pdf_file.name:<50} | SKIP — too little text")
                failed_count += 1
                continue

            date_str  = find_date(text)
            date_obj  = parse_date(date_str)
            fatura_no = find_fatura_no(text)
            tutar     = find_tutar(text)
            company   = find_company(text)

            # Determine output folder by date
            if date_obj:
                folder = base_out / str(date_obj.year) / MONTH_FOLDER[date_obj.month]
            else:
                folder = base_out / "undated"
            folder.mkdir(parents=True, exist_ok=True)

            # Keep original filename, handle collisions
            out_path = folder / pdf_file.name
            counter = 2
            while out_path.exists():
                out_path = folder / f"{pdf_file.stem}_{counter}{pdf_file.suffix}"
                counter += 1

            shutil.copy2(str(pdf_file), str(out_path))
            success_count += 1

            rel = out_path.relative_to(base_out)
            print(
                f"[{i:>4}/{len(pdf_files)}] "
                f"date: {(date_obj.strftime('%d.%m.%Y') if date_obj else 'no date'):<12} | "
                f"fatura: {(fatura_no or '-'):<24} | "
                f"→ results/{rel}"
            )

            excel_rows.append({
                "Düzenleme Tarihi": date_obj.strftime("%d.%m.%Y") if date_obj else "",
                "Firma":           company,
                "Tutar":           tutar,
                "Fatura No":       fatura_no,
                "Proje":           "",
            })

        except Exception as exc:
            failed_count += 1
            print(f"[{i:>4}/{len(pdf_files)}] {pdf_file.name:<50} | ERROR — {exc}")
            traceback.print_exc(file=sys.stderr)

    # ── write Excel ──────────────────────────────────────────────────────────────

    if excel_rows:
        xl_path = base_out / "faturalar.xlsx"
        df = pd.DataFrame(excel_rows, columns=["Düzenleme Tarihi", "Firma", "Tutar", "Fatura No", "Proje"])
        with pd.ExcelWriter(xl_path, engine="openpyxl") as xw:
            df.to_excel(xw, index=False, sheet_name="Faturalar")
            ws = xw.sheets["Faturalar"]
            for col in ws.columns:
                max_len = max((len(str(c.value)) for c in col if c.value), default=10)
                ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 50)
        print(f"\n  Excel : {xl_path}")

    # ── summary ──────────────────────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print(f"  Done!  {success_count}/{len(pdf_files)} files sorted")
    if failed_count:
        print(f"  Failed/Skip:  {failed_count}")
    print(f"{'='*60}\n")


# ── entry point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Text PDF Invoice Sorter")
        print("-" * 40)
        path = input("Enter the folder path containing PDF invoices: ").strip().strip('"')
    else:
        path = sys.argv[1].strip('"')

    process_folder(path)
