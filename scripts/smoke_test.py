"""Smoke test of the PDF report and JSON export."""

from __future__ import annotations
from pathlib import Path
from src.report.pdf_report import build_pdf_report
from src.utils.export import json_bytes

def main():
    """Write a small report into outputs/."""
    Path("outputs").mkdir(exist_ok=True)
    pdf = build_pdf_report(
        "Smoke test - Unified XAI Interface",
        meta={"ok": True},
        sections=[
            {"heading":"Assets", "text":"placeholder files exist", "images":[]},
        ]
    )
    Path("outputs/smoke_test_report.pdf").write_bytes(pdf)
    Path("outputs/smoke_test.json").write_bytes(json_bytes({"ok": True}))
    print("Wrote outputs/smoke_test_report.pdf")

if __name__ == "__main__":
    main()
