from pathlib import Path

from bs4 import BeautifulSoup

html_dir = Path(
    r"C:\work\INTERVIEW_PREPARATION\CBC\data\extracted\extracted_guidelines_html"
)


def has_article_content(html_content, min_length=100):
    soup = BeautifulSoup(html_content, "html.parser")

    # Attempt to select the known main content container
    main_content = (
        soup.find("div", {"role": "main"}) or soup.find("article") or soup.find("main")
    )

    if main_content:
        for element in main_content(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        text = main_content.get_text(separator=" ", strip=True)
        text = " ".join(text.split())
        return len(text) >= min_length
    else:
        return False  # No recognized main content container found


empty_files = []
html_files = list(html_dir.glob("*.html"))

for html_file in html_files:
    with open(html_file, "r", encoding="utf-8") as f:
        content = f.read()

    if not has_article_content(content):
        empty_files.append(html_file.name)

print(f"\nTotal HTML files checked: {len(html_files)}")
print(f"Files without meaningful textual content: {len(empty_files)}")

if empty_files:
    print("The following HTML files lack meaningful textual content:")
    for file in empty_files:
        print(f"- {file}")
else:
    print("All HTML files contain sufficient textual content.")
