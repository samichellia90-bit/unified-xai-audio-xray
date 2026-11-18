"""PDF report generation."""

from __future__ import annotations
import io
from typing import Dict, Any, List
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from PIL import Image as PILImage

def _img(bytes_png: bytes, w_cm: float = 15.5) -> RLImage:
    bio = io.BytesIO(bytes_png)
    try:
        im = PILImage.open(bio).convert("RGB")
        out = io.BytesIO()
        im.save(out, format="PNG")
        out.seek(0)
        img = RLImage(out)
    except Exception:
        bio.seek(0)
        img = RLImage(bio)
    img.drawWidth = w_cm * cm
    img.drawHeight = img.imageHeight * (img.drawWidth / img.imageWidth)
    return img

def build_pdf_report(title: str, meta: Dict[str, Any], sections: List[Dict[str, Any]]) -> bytes:
    """Build a PDF with a title, metadata and illustrated sections, and return its bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.6*cm, rightMargin=1.6*cm, topMargin=1.4*cm, bottomMargin=1.4*cm)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 0.25*cm))
    story.append(Paragraph("Interface XAI unifiee - Rapport auto", styles["Normal"]))
    story.append(Spacer(1, 0.35*cm))

    rows = [[str(k), str(v)] for k, v in meta.items()]
    tbl = Table(rows, colWidths=[5.0*cm, 11.5*cm])
    tbl.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.25, colors.Color(0.7,0.7,0.7)),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("TEXTCOLOR", (0,0), (-1,-1), colors.black),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [colors.whitesmoke, colors.white]),
    ]))
    story.append(tbl)

    for i, sec in enumerate(sections):
        if i > 0:
            story.append(PageBreak())
        story.append(Paragraph(sec.get("heading","Section"), styles["Heading1"]))
        story.append(Spacer(1, 0.2*cm))
        if sec.get("text"):
            story.append(Paragraph(sec["text"], styles["Normal"]))
            story.append(Spacer(1, 0.3*cm))
        images = [png for png in (sec.get("images", []) or []) if png is not None]
        if len(images) <= 1:
            for png in images:
                story.append(_img(png))
                story.append(Spacer(1, 0.35*cm))
        else:
            rows = []
            row = []
            for png in images:
                row.append(_img(png, w_cm=7.6))
                if len(row) == 2:
                    rows.append(row)
                    row = []
            if row:
                row.append("")
                rows.append(row)
            grid = Table(rows, colWidths=[8.0*cm, 8.0*cm])
            grid.setStyle(TableStyle([
                ("VALIGN", (0,0), (-1,-1), "TOP"),
                ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ]))
            story.append(grid)
            story.append(Spacer(1, 0.35*cm))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()
