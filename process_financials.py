#!/usr/bin/env python3
"""
PDF Financial Document Processor
Uses Claude Haiku native PDF support for classification and data extraction.
No regex, no OCR — Claude reads each page directly.

Filename format: {DD.MM.YYYY}_{FATURA_NO}.pdf

Usage:
    python process_financials.py <path_to_pdf>
"""

import sys
import os
import re
import io
import base64
import json
import traceback
from pathlib import Path
from datetime import datetime

# Force UTF-8 + line-buffered output on Windows
if sys.platform == "win32":
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

import shutil
import anthropic
import pypdfium2 as pdfium
import pytesseract
from PIL import Image
from pypdf import PdfReader, PdfWriter
import pandas as pd


# ── config ─────────────────────────────────────────────────────────────────────

OWN_COMPANY_NAME = "NOVESYST"
CLAUDE_MODEL     = "claude-haiku-4-5-20251001"
MAX_TOKENS       = 512

# OCR render scale for Sayın override check
OCR_SCALE = 2.0

RE_SAYIN = re.compile(r"say[iı1l][nm]", re.IGNORECASE)

# Detect Tesseract availability once at startup
def _tesseract_available() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False

TESSERACT_OK = _tesseract_available()
if not TESSERACT_OK:
    print("[WARN] Tesseract not found — Sayın OCR override disabled, trusting Claude only.")


# ── constants ──────────────────────────────────────────────────────────────────

MONTH_FOLDER = {
    1: "01_Ocak",   2: "02_Subat",   3: "03_Mart",    4: "04_Nisan",
    5: "05_Mayis",  6: "06_Haziran", 7: "07_Temmuz",  8: "08_Agustos",
    9: "09_Eylul",  10: "10_Ekim",   11: "11_Kasim",  12: "12_Aralik",
}


# ── Claude client ───────────────────────────────────────────────────────────────

_client = None

def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env
    return _client


# ── system prompt ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""\
You are a financial document classifier for a Turkish company called {OWN_COMPANY_NAME}.

Given a page from a Turkish financial PDF, return ONLY a JSON object with these exact fields:

{{
  "doc_type": "cost" | "income" | "bank_statement" | "unknown",
  "fatura_no": string | null,
  "date": "DD.MM.YYYY" | null,
  "company": string | null,
  "tutar": string | null,
  "is_continuation": true | false
}}

══ STEP 1 — Is this page a bill, a continuation, or a bank statement? ══

A page is a BILL FIRST PAGE if ANY of the following are true:
  • It contains a fatura/invoice number (Fatura No field)
  • It contains the e-FATURA or e-Arşiv Fatura stamp/header
  → doc_type MUST be "cost" or "income". NEVER "bank_statement".

A page is a CONTINUATION (is_continuation: true) if:
  • No fatura_no on this page
  • No e-FATURA / e-Arşiv stamp
  • No "Sayın" field
  • Page contains ONLY items, totals, or payment info (e.g. "BANKA HESAP BİLGİLERİ" with IBANs)
  → is_continuation: true, doc_type: same as previous page (cost/income)

  !! IMPORTANT: A page showing "BANKA HESAP BİLGİLERİ" with IBAN numbers is the PAYMENT
  SECTION of an invoice (page 2/2), NOT a bank statement. Mark it as is_continuation: true.

A page is a BANK STATEMENT only if ALL are true:
  • No fatura_no, no e-FATURA stamp, no "Sayın" field
  • Has a heading like "Hesap Ekstresi" / "Hesap Özeti" / "Banka Ekstresi"
  • Shows transaction rows with columns: tarih / açıklama / borç / alacak / bakiye

══ STEP 2 — For bill first pages: cost or income? ══
CRITICAL: Both {OWN_COMPANY_NAME} and the other company ALWAYS appear on EVERY bill.
Do NOT use position, visual prominence, or "Fatura Tipi" to classify.

!! "Fatura Tipi: SATIS" appears on ALL invoices (both cost and income). IGNORE this field.

Use ONLY the "Sayın" field — it identifies the BUYER (who pays):
  • "Sayın {OWN_COMPANY_NAME}" → doc_type: "cost"
    The other company is at the top as seller. {OWN_COMPANY_NAME} is below as buyer. This is cost.
  • "Sayın [other company]" → doc_type: "income"
    {OWN_COMPANY_NAME} is at the top as seller. The other company is below as buyer.

If "Sayın {OWN_COMPANY_NAME}" appears ANYWHERE on the page → doc_type: "cost". No exceptions.

══ fatura_no ══
The invoice document number (e.g. EFA2026000000001, BYS2026000000378).
Turkish e-fatura format: 2-5 uppercase letters + 4-digit year + 9 digits (total 13-18 chars).
Every valid bill ALWAYS has a fatura_no on its first page — search carefully before returning null.

══ date ══
The document ISSUE date. Look for: "fatura tarihi", "düzenleme tarihi", "belge tarihi", "tarih".
Format as DD.MM.YYYY.
NEVER use vade tarihi, son ödeme tarihi, or any due/payment deadline date.
Every valid bill ALWAYS has an issue date — search carefully (header, stamp, table cell) before returning null.

══ company ══
The name of the OTHER company (not {OWN_COMPANY_NAME}).
Both companies appear on every bill — return only the one that is NOT {OWN_COMPANY_NAME}.
Return null only on continuation pages where no company name appears.

══ tutar ══
Final total amount payable: ödenecek tutar / genel toplam / toplam tutar (KDV dahil).
Format exactly as printed e.g. "1.234,56". Return null if not found.

══ is_continuation ══
true  → continuation page of a multi-page document:
        - no fatura_no on this page
        - no e-FATURA stamp
        - no "Sayın" field
        - contains only item rows, subtotals, or supplementary pages
false → first (or only) page of a new document.

Return ONLY the JSON object. No markdown fences, no explanation.\
"""


# ── page extraction ─────────────────────────────────────────────────────────────

def page_to_pdf_bytes(reader: PdfReader, page_index: int) -> bytes:
    """Extract a single page from an open PdfReader as raw PDF bytes."""
    writer = PdfWriter()
    writer.add_page(reader.pages[page_index])
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


# ── Claude classification ────────────────────────────────────────────────────────

def classify_page_claude(page_pdf_bytes: bytes) -> dict:
    """
    Send one page (as PDF bytes) to Claude Haiku.
    Returns a normalised dict with doc_type, fatura_no, date_str, company, tutar, is_continuation.
    """
    pdf_b64 = base64.standard_b64encode(page_pdf_bytes).decode("utf-8")

    response = get_client().beta.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        betas=["pdfs-2024-09-25"],
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": pdf_b64,
                    },
                },
                {
                    "type": "text",
                    "text": "Extract the information from this page.",
                },
            ],
        }],
    )

    raw = response.content[0].text.strip()
    # Strip accidental markdown code fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"    [WARN] JSON parse failed ({exc}) — raw: {raw[:120]}", file=sys.stderr)
        return _empty_result()

    return {
        "doc_type":        str(result.get("doc_type") or "unknown"),
        "fatura_no":       str(result.get("fatura_no") or "").strip().upper(),
        "date_str":        str(result.get("date") or "").strip(),
        "company":         str(result.get("company") or "").strip(),
        "tutar":           str(result.get("tutar") or "").strip(),
        "is_continuation": bool(result.get("is_continuation", False)),
    }


def _empty_result() -> dict:
    return {
        "doc_type": "unknown", "fatura_no": "", "date_str": "",
        "company": "", "tutar": "", "is_continuation": False,
    }


NOVESYST_SIGNALS = ["nove", "vesyst", "6321463"]  # covers OCR errors like sovesyst, n0vesyst

def _sayin_buyer(text: str) -> str:
    """
    Find who appears after 'Sayın' in OCR text.
    Returns 'novesyst', 'other', or 'unknown'.
    Looks at the 10 lines after each Sayın occurrence.
    Uses multiple signals to handle OCR errors (sovesyst, n0vesyst, etc.)
    Also checks Novesyst's unique VKN (6321463957) as a fallback.
    """
    lines = [l.strip() for l in text.splitlines()]
    for i, line in enumerate(lines):
        if RE_SAYIN.search(line):
            context = " ".join(lines[i + 1: i + 11]).lower()
            if any(sig in context for sig in NOVESYST_SIGNALS):
                return "novesyst"
            elif len(context.strip()) > 3:
                return "other"
    return "unknown"


def ocr_sayin_override(pdf_path: str, page_idx: int, result: dict) -> dict:
    """
    After Claude classifies a page, verify cost/income using OCR + Sayın rule.
    Finds the 'Sayın' line and checks the next 5 lines for 'nove' (Novesyst).
    Skips continuation pages (they have no Sayın by definition).
    """
    if result.get("is_continuation") or not TESSERACT_OK:
        return result

    try:
        doc    = pdfium.PdfDocument(pdf_path)
        page   = doc[page_idx]
        bitmap = page.render(scale=OCR_SCALE)
        text   = pytesseract.image_to_string(bitmap.to_pil(), lang="tur+eng")

        buyer = _sayin_buyer(text)

        if buyer == "novesyst":
            if result["doc_type"] != "cost":
                print(f"    [SAYIN OVERRIDE] {result['doc_type']} → cost")
            result["doc_type"] = "cost"
        elif buyer == "other":
            if result["doc_type"] != "income":
                print(f"    [SAYIN OVERRIDE] {result['doc_type']} → income")
            result["doc_type"] = "income"
        # buyer == 'unknown' → no Sayın found → trust Claude

    except Exception as e:
        print(f"    [SAYIN OVERRIDE] OCR failed: {e}", file=sys.stderr)

    return result


def parse_date(date_str: str):
    """Parse DD.MM.YYYY (or variants) from Claude's response into datetime."""
    if not date_str:
        return None
    for fmt in ("%d.%m.%Y", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


# ── file output helpers ─────────────────────────────────────────────────────────

def make_output_filename(date_obj, fatura_no: str) -> str:
    date_str = date_obj.strftime("%d.%m.%Y") if date_obj else "undated"
    if fatura_no:
        # Sanitise fatura_no for use in filenames
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


def save_pages_as_pdf(src_reader: PdfReader, page_indices: list[int], dest_path: Path) -> None:
    writer = PdfWriter()
    for idx in page_indices:
        writer.add_page(src_reader.pages[idx])
    with open(dest_path, "wb") as fh:
        writer.write(fh)


# ── main processing loop ────────────────────────────────────────────────────────

def process_pdf(pdf_path: str, start_page: int = 1, end_page: int = None) -> None:
    """
    Process a PDF file.
    start_page / end_page are 1-indexed and inclusive.
    If end_page is None, processes until the last page.
    """
    pdf_path = os.path.abspath(pdf_path)
    if not os.path.isfile(pdf_path):
        print(f"[ERROR] File not found: {pdf_path}")
        sys.exit(1)

    REPO_DIR = Path(__file__).resolve().parent
    base_out = REPO_DIR / "results"
    if base_out.exists():
        shutil.rmtree(base_out)
        print("  Cleared previous results folder.")
    base_out.mkdir(parents=True, exist_ok=True)

    reader      = PdfReader(pdf_path)
    total_pages = len(reader.pages)

    # Clamp and convert to 0-based indices
    start_idx = max(0, start_page - 1)
    end_idx   = min(total_pages - 1, (end_page - 1) if end_page else total_pages - 1)
    page_range = range(start_idx, end_idx + 1)

    print(f"\n{'='*60}")
    print(f"  Processing : {Path(pdf_path).name}")
    print(f"  Pages      : {start_page}–{end_idx + 1} of {total_pages}")
    print(f"  Model      : {CLAUDE_MODEL}")
    print(f"  Output     : {base_out}")
    print(f"{'='*60}\n")

    log_rows  = []
    cost_rows = []
    doc_count = 0
    pending: list[dict] = []

    # ── nested flush helper ─────────────────────────────────────────────────────

    def flush_group(group: list[dict]) -> None:
        nonlocal doc_count

        # Lead page = first page that has a fatura_no or date
        lead = next(
            (p for p in group if p["fatura_no"] or p["date_obj"]),
            group[0],
        )

        doc_type  = lead["doc_type"]
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
            f"results/{rel}"
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

    # ── page loop ───────────────────────────────────────────────────────────────

    for page_idx in page_range:
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
            "reason":          "",
        }

        try:
            page_bytes = page_to_pdf_bytes(reader, page_idx)
            result     = classify_page_claude(page_bytes)

            # OCR-based Sayın override — ground truth for cost/income
            result = ocr_sayin_override(pdf_path, page_idx, result)

            info["doc_type"]        = result["doc_type"]
            info["fatura_no"]       = result["fatura_no"]
            info["date_obj"]        = parse_date(result["date_str"])
            info["company"]         = result["company"]
            info["tutar"]           = result["tutar"]
            info["is_continuation"] = result["is_continuation"]
            info["status"]          = "success"

            # Hard override: a page with a fatura_no is ALWAYS an invoice.
            # Claude can misread IBAN/banka keywords as a bank statement.
            if info["fatura_no"] and info["doc_type"] == "bank_statement":
                info["doc_type"] = "cost"
                print(f"    [OVERRIDE] fatura_no present → changed bank_statement → cost")

            cont_tag = " <CONT>" if result["is_continuation"] else ""
            print(
                f"[Page {page_num:>4}/{total_pages}] "
                f"{info['doc_type']:<16} | "
                f"fatura={info['fatura_no'] or '-':<24} | "
                f"date={info['date_obj'].strftime('%d.%m.%Y') if info['date_obj'] else 'none':<12}"
                f"{cont_tag}"
            )

        except Exception as exc:
            info["status"] = "failed"
            info["reason"] = f"{type(exc).__name__}: {exc}"
            print(f"[Page {page_num:>4}/{total_pages}] ERROR – {info['reason']}")
            traceback.print_exc(file=sys.stderr)

        # ── grouping ────────────────────────────────────────────────────────────
        if not pending:
            pending.append(info)

        elif info["status"] == "failed":
            log_rows.append({
                "page_number": page_num, "classified_as": "unknown",
                "extracted_date": "", "extracted_company": "",
                "fatura_no": "", "tutar": "", "pages_in_document": 1,
                "output_filename": "", "status": "failed", "reason": info["reason"],
            })

        elif info["is_continuation"]:
            # Claude says this is a continuation page
            pending.append(info)

        else:
            # Check fatura_no match as extra safety (Claude might say is_continuation=false
            # on page 2 of a long invoice if the fatura_no is repeated there)
            group_fatura = next((p["fatura_no"] for p in pending if p["fatura_no"]), "")
            if info["fatura_no"] and info["fatura_no"] == group_fatura:
                pending.append(info)
            else:
                flush_group(pending)
                pending = [info]

    if pending:
        flush_group(pending)

    # ── write outputs ────────────────────────────────────────────────────────────

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
    print(f"  Done!  {success}/{len(page_range)} pages | {doc_count} documents saved")
    print(f"  Cost:         {counts.get('cost', 0)}")
    print(f"  Income:       {counts.get('income', 0)}")
    print(f"  Bank stmts:   {counts.get('bank_statement', 0)}")
    print(f"  Unknown:      {counts.get('unknown', 0)}")
    if failed:
        print(f"  Failed:       {failed}")
    print(f"  Log:          {log_path}")
    print(f"{'='*60}\n")


# ── entry point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("PDF Financial Document Processor")
        print("-" * 40)
        path = input("Enter the full path to the PDF file: ").strip().strip('"')
        process_pdf(path)
    else:
        path  = sys.argv[1].strip('"')
        start = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        end   = int(sys.argv[3]) if len(sys.argv) > 3 else None
        process_pdf(path, start_page=start, end_page=end)
