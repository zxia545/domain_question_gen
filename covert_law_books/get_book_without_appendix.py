#!/usr/bin/env python3
"""
Module: book_content_extractor.py

This module parses an HTML file representing a book’s content with CSS‐styled hierarchical
markers. It extracts the structure as follows:
  • A part is indicated by an H2 element whose inner span (class "larger") begins with "PART".
  • A chapter is indicated by an H3 element (with inner spans "x-large" and "smaller").
  • A section is indicated by an H4 element (with inner spans "larger" for the section number
    and "smaller" for the section title).
  
Within each section the module scans the content (all siblings following the H4 until the next
header) and extracts:
  – The bibliographic reference: the first paragraph (<p>) whose class starts with "indh".
  – One or more sidenote groups. Each sidenote group is started by a <div class="sidenote">
    (whose text is the sidenote heading) and followed by one or more paragraphs and footnote blocks.
    Footnote blocks (<div class="footnote">) found immediately after a sidenote block are appended
    (in a formatted manner) to that group’s content.

The final nested structure is exported as JSON.
"""

import json
from bs4 import BeautifulSoup

def extract_footnote_text(footnote_div) -> str:
    """
    Given a footnote div, extract and format its text.
    For example, if the div contains a <span class="label">[1]</span> and additional text,
    the result will be: "[Footnote 1: <rest of text>]"
    """
    paras = footnote_div.find_all("p")
    full_text = " ".join(p.get_text(" ", strip=True) for p in paras)
    label_tag = footnote_div.find("span", class_="label")
    label = label_tag.get_text(strip=True) if label_tag else ""
    if label.startswith("[") and label.endswith("]"):
        label = label[1:-1]
    prefix = f"[{label}]"
    if full_text.startswith(prefix):
        content_text = full_text[len(prefix):].strip()
    else:
        content_text = full_text
    return f"[Footnote {label}: {content_text}]"

def process_section(section_header) -> dict:
    """
    Given an H4 element (the section header), process its following siblings until the next header
    (h2, h3, or h4) is encountered. Returns a dictionary with:
      - "section_number": from the span.larger inside the h4,
      - "section_title": from the span.smaller (or the full text if absent),
      - "bibliographic": the first <p> with class starting with "indh",
      - "sidenote_groups": a list of groups, each an object with "sidenote" and "content".
      
    Footnote blocks (div.footnote) that appear are appended to the content of the current sidenote group.
    Any content is collected as text (joined by spaces).
    """
    section = {}
    # Extract section number and title from the h4 header.
    span_num = section_header.find("span", class_="larger")
    section["section_number"] = span_num.get_text(strip=True) if span_num else ""
    span_title = section_header.find("span", class_="smaller")
    section["section_title"] = span_title.get_text(" ", strip=True) if span_title else section_header.get_text(" ", strip=True)
    
    section["bibliographic"] = ""
    section["sidenote_groups"] = []
    # current_group will hold a dict with keys "sidenote" and "content" (a list of strings)
    current_group = None

    # Iterate over all siblings following the section header.
    for sib in section_header.next_siblings:
        # Skip strings that are only whitespace.
        if isinstance(sib, str):
            if sib.strip() == "":
                continue
            # If there's stray text, append to current group if exists.
            if current_group is not None:
                current_group["content"].append(sib.strip())
            continue
        # If we encounter a header element, end processing this section.
        if sib.name in ["h2", "h3", "h4"]:
            break

        # Process paragraph(s) with class starting with "indh".
        if sib.name == "p":
            classes = sib.get("class") or []
            if any(cls.startswith("indh") for cls in classes):
                p_text = sib.get_text(" ", strip=True)
                # If bibliographic content is not yet set, use the first such paragraph.
                if section["bibliographic"] == "":
                    section["bibliographic"] = p_text
                else:
                    # If we are in a sidenote group, append to its content.
                    if current_group is not None:
                        current_group["content"].append(p_text)
                    else:
                        # Otherwise, store as "orphan" content (if desired).
                        section.setdefault("orphan_content", []).append(p_text)
                continue
            else:
                # For other paragraphs, if a sidenote group is active, append.
                if current_group is not None:
                    current_group["content"].append(sib.get_text(" ", strip=True))
                continue

        # Process a sidenote block.
        if sib.name == "div" and "sidenote" in (sib.get("class") or []):
            # Start a new sidenote group.
            sidenote_text = sib.get_text(" ", strip=True)
            current_group = {"sidenote": sidenote_text, "content": []}
            section["sidenote_groups"].append(current_group)
            continue

        # Process a footnote block.
        if sib.name == "div" and "footnote" in (sib.get("class") or []):
            foot_text = extract_footnote_text(sib)
            if current_group is not None:
                # Append footnote text to current group's content.
                current_group["content"].append(foot_text)
            else:
                # If no sidenote group active, you might store it separately.
                section.setdefault("orphan_footnotes", []).append(foot_text)
            continue

        # For any other tag, if we wish, we could append its text to current group.
        # (You can customize this part as needed.)
        if current_group is not None:
            current_group["content"].append(sib.get_text(" ", strip=True))
    return section

def extract_book_content(html_content: str) -> dict:
    """
    Parses the HTML content and builds a nested dictionary representing the book structure.
    
    It iterates over header elements (h2, h3, h4) in document order:
      - An H2 with a span.larger beginning with "PART" starts a new part.
      - An H3 starts a new chapter.
      - An H4 triggers processing of a section via process_section().
      
    Returns a dictionary with a "parts" key (a list of parts).
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    result = {"parts": []}
    
    current_part = None
    current_chapter = None
    
    # Find all headers (h2, h3, h4) in document order.
    for header in soup.find_all(["h2", "h3", "h4"]):
        if header.name == "h2":
            # Look for a part header: the inner span.larger should start with "PART"
            span_larger = header.find("span", class_="larger")
            if span_larger and span_larger.get_text(strip=True).upper().startswith("PART"):
                part_title = span_larger.get_text(strip=True)
                span_smaller = header.find("span", class_="smaller")
                part_subtitle = span_smaller.get_text(strip=True) if span_smaller else ""
                current_part = {
                    "part_title": part_title,
                    "part_subtitle": part_subtitle,
                    "chapters": []
                }
                result["parts"].append(current_part)
                current_chapter = None
            continue

        if header.name == "h3":
            # Chapter header.
            span_xlarge = header.find("span", class_="x-large")
            chapter_title = span_xlarge.get_text(strip=True) if span_xlarge else header.get_text(" ", strip=True)
            span_smaller = header.find("span", class_="smaller")
            chapter_subtitle = span_smaller.get_text(strip=True) if span_smaller else ""
            current_chapter = {
                "chapter_title": chapter_title,
                "chapter_subtitle": chapter_subtitle,
                "sections": []
            }
            if current_part is None:
                # Create a dummy part if necessary.
                current_part = {"part_title": "", "part_subtitle": "", "chapters": []}
                result["parts"].append(current_part)
            current_part["chapters"].append(current_chapter)
            continue

        if header.name == "h4":
            # Section header. Process the section content using process_section.
            section_data = process_section(header)
            if current_chapter is None:
                # Create a dummy chapter if necessary.
                current_chapter = {"chapter_title": "", "chapter_subtitle": "", "sections": []}
                if current_part is None:
                    current_part = {"part_title": "", "part_subtitle": "", "chapters": []}
                    result["parts"].append(current_part)
                current_part["chapters"].append(current_chapter)
            current_chapter["sections"].append(section_data)
            continue

    return result

def main():
    # Replace 'content.html' with your actual file path.
    input_path = 'pg41046-images.html'
    with open(input_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    book_structure = extract_book_content(html_content)
    output_json = json.dumps(book_structure, indent=4, ensure_ascii=False)
    print(output_json)
    with open('pg41046.json', 'w', encoding='utf-8') as out_file:
        out_file.write(output_json)

if __name__ == '__main__':
    main()
