from pathlib import Path
import markdown
from weasyprint import HTML

md_path = Path("AGENTIC_TRANSFORMATION_ANALYSIS.md")
html_path = Path("AGENTIC_TRANSFORMATION_ANALYSIS.html")
pdf_path = Path("AGENTIC_TRANSFORMATION_ANALYSIS.pdf")

# Convert Markdown to HTML
html_content = markdown.markdown(
    md_path.read_text(encoding="utf-8"),
    extensions=["tables", "fenced_code"]
)

html_path.write_text(html_content, encoding="utf-8")

# Convert HTML to PDF
HTML(html_path.as_uri()).write_pdf(pdf_path)

print("PDF created:", pdf_path)
