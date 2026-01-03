from pathlib import Path
import markdown

md_file = Path("GUI_ALTERNATIVES_ANALYSIS.md")
html_file = Path("GUI_ALTERNATIVES_ANALYSIS.html")

html = markdown.markdown(
    md_file.read_text(encoding="utf-8"),
    extensions=["tables", "fenced_code"]
)

html_file.write_text(html, encoding="utf-8")
print("Created GUI_ALTERNATIVES_ANALYSIS.html")
