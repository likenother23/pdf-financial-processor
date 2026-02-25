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
from pypdf import PdfReader, PdfWriter
import pandas as pd


# ── config ─────────────────────────────────────────────────────────────────────

OWN_COMPANY_NAME = "NOVESYST"
CLAUDE_MODEL     = "claude-haiku-4-5-20251001"
MAX_TOKENS       = 512


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

══ MOST IMPORTANT RULE ══
If the page contains ANY invoice/fatura number (a document reference like EFA2026000000001,
BYS2026000000378, MTO2026000000002, or any code matching 2-5 letters + 4-digit year + digits),
then doc_type MUST be "cost" or "income" — NEVER "bank_statement".
Invoices always contain payment IBANs and bank names, but that does NOT make them bank statements.

══ doc_type rules ══
"cost"           → invoice/bill where {OWN_COMPANY_NAME} is the BUYER
                   (alıcı / müşteri / faturayı alan / sayın {OWN_COMPANY_NAME})
"income"         → invoice/bill where {OWN_COMPANY_NAME} is the SELLER
                   (satıcı / tedarikçi / faturayı kesen / düzenleyen {OWN_COMPANY_NAME})
"bank_statement" → ONLY for genuine bank account statements: hesap özeti, ekstre,
                   hesap hareketi listesi. These have NO fatura_no and show debit/credit rows.
"unknown"        → cannot determine

══ fatura_no ══
The invoice document number (e.g. EFA2026000000001, BYS2026000000378).
Turkish e-fatura format: 2-5 uppercase letters + 4-digit year + 9 digits (total 13-18 chars).
Return null only if truly absent from this page.

══ date ══
The document ISSUE date: look for "fatura tarihi", "düzenleme tarihi", "belge tarihi", "tarih".
Format as DD.MM.YYYY.
NEVER use vade tarihi, son ödeme tarihi, or any due/payment deadline date.
Search carefully — the date may appear in a header, stamp, or table cell.
Return null only if no issue date exists anywhere on the page.

══ company ══
The name of the OTHER company (not {OWN_COMPANY_NAME}).
Return null if not found or if this is a continuation page.

══ tutar ══
Final total amount payable: ödenecek tutar / genel toplam / toplam tutar (KDV dahil).
Format exactly as printed e.g. "1.234,56". Return null if not found.

══ is_continuation ══
true  → page 2+ of a multi-page document (no own header/fatura_no, just item rows or extra pages).
false → first or only page of a new document.

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
        "fatura_no":       str(result.get("fatura_no") or "").strip(),
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

def process_pdf(pdf_path: str) -> None:
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

    print(f"\n{'='*60}")
    print(f"  Processing : {Path(pdf_path).name}")
    print(f"  Pages      : {total_pages}")
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
            "reason":          "",
        }

        try:
            page_bytes = page_to_pdf_bytes(reader, page_idx)
            result     = classify_page_claude(page_bytes)

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
                info["doc_type"] = "cost"  # safest fallback; prompt already asked for perspective
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
    print(f"  Done!  {success}/{total_pages} pages | {doc_count} documents saved")
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
    else:
        path = sys.argv[1].strip('"')

    process_pdf(path)
