"""Extract benchmark tables from Chronos paper PDF.
Usage: uv run --with pdfplumber python3 extract_tables.py
"""
import pdfplumber
import os

base = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(base, "chronos_paper.pdf")

pdf = pdfplumber.open(pdf_path)
print(f"Total pages: {len(pdf.pages)}")

# Search all pages for tables with benchmark data
for i in range(len(pdf.pages)):
    page = pdf.pages[i]
    text = page.extract_text()
    if text and ("Agg" in text or "Relative" in text or "Table 9" in text or "Table 10" in text or "Table 8" in text):
        print(f"\n{'='*80}")
        print(f"PAGE {i+1}")
        print(f"{'='*80}")
        print(text[:4000])

    # Also try table extraction
    tables = page.extract_tables()
    if tables:
        for t_idx, table in enumerate(tables):
            # Check if this table has benchmark-like data
            flat = str(table)
            if any(kw in flat for kw in ["MASE", "WQL", "Agg", "Relative", "AutoETS", "Lag-Llama", "Moirai"]):
                print(f"\n--- TABLE on page {i+1}, table {t_idx} ---")
                for row in table:
                    print(row)
