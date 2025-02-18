#!/usr/bin/env python3
"""
Module: book_text_extractor.py

This script converts a TEI‐style HTML book into a JSON structure that focuses solely on
the actual text content of each page. Pages are identified by a page marker:
   <span class="tei tei-pb" id="pageXXX">[pg XXX]</span>
For each page, the script extracts the text of paragraphs (<p> elements) and, if a list (<ul> or <ol>)
immediately follows a paragraph, its list items are attached as subparagraphs.
The resulting JSON has the following structure:

{
    "pages": [
         {
             "page_number": "001",
             "paragraphs": [
                 "First paragraph text",
                 {
                     "text": "Second paragraph text",
                     "subparagraphs": [
                         "First subparagraph",
                         "Second subparagraph"
                     ]
                 },
                 "Third paragraph text"
             ]
         },
         ...
    ]
}

Only the plain text is preserved; tag names, attributes, and styling are ignored.
"""

import json
from bs4 import BeautifulSoup, NavigableString, Tag

def extract_paragraphs_from_siblings(siblings):
    """
    Processes a sequence of sibling elements and extracts paragraphs.
    If a <p> is immediately followed by a list (<ul> or <ol>), that list’s <li> items
    are attached as subparagraphs to the preceding paragraph.
    Returns a list of paragraphs. Each paragraph is either a string or a dict
    with keys "text" and "subparagraphs".
    """
    paragraphs = []
    # Filter out empty strings
    sibling_list = [sib for sib in siblings if (isinstance(sib, Tag)) or (isinstance(sib, NavigableString) and sib.strip())]
    i = 0
    while i < len(sibling_list):
        sib = sibling_list[i]
        if isinstance(sib, Tag):
            if sib.name == 'p':
                para_text = sib.get_text(" ", strip=True)
                subparas = []
                # Look ahead: if next sibling is a list (<ul> or <ol>), attach its items as subparagraphs.
                if i + 1 < len(sibling_list):
                    next_sib = sibling_list[i + 1]
                    if isinstance(next_sib, Tag) and next_sib.name in ['ul', 'ol']:
                        subparas = [li.get_text(" ", strip=True) for li in next_sib.find_all('li', recursive=False)]
                        i += 1  # Skip the list since we processed it
                if subparas:
                    paragraphs.append({"text": para_text, "subparagraphs": subparas})
                elif para_text:
                    paragraphs.append(para_text)
            elif sib.name in ['ul', 'ol']:
                # If a list appears on its own, add its items as a paragraph (without a preceding para).
                li_texts = [li.get_text(" ", strip=True) for li in sib.find_all('li', recursive=False)]
                if li_texts:
                    paragraphs.append({"subparagraphs": li_texts})
            elif sib.name == 'div':
                # Process div children recursively.
                inner_paras = extract_paragraphs_from_siblings(sib.children)
                paragraphs.extend(inner_paras)
            else:
                # For other block tags, try to get their text.
                text = sib.get_text(" ", strip=True)
                if text:
                    paragraphs.append(text)
        elif isinstance(sib, NavigableString):
            text = sib.strip()
            if text:
                paragraphs.append(text)
        i += 1
    return paragraphs

def extract_pages(soup: BeautifulSoup) -> list:
    """
    Extracts page content from the HTML.
    It finds all <span> elements with class "tei tei-pb" (the page markers) and,
    for each, collects all following sibling elements until the next page marker.
    The collected sibling elements are then processed to extract paragraphs and subparagraphs.
    Returns a list of pages, each a dict with "page_number" and "paragraphs".
    """
    pages = []
    # Find all page marker spans (e.g., <span class="tei tei-pb" id="page001">[pg 001]</span>)
    page_markers = soup.find_all(lambda tag: tag.name == 'span' and tag.get('class') 
                                 and 'tei-pb' in tag.get('class'))
    for marker in page_markers:
        page_text = marker.get_text(strip=True)
        # Remove surrounding brackets and the "pg " prefix if present.
        if page_text.startswith('[') and page_text.endswith(']'):
            page_text = page_text[1:-1]
        if page_text.lower().startswith('pg '):
            page_num = page_text[3:]
        else:
            page_num = page_text
        # Gather all siblings until the next page marker.
        content_siblings = []
        for elem in marker.next_siblings:
            if isinstance(elem, Tag):
                # Stop if we hit another page marker.
                if elem.name == 'span' and elem.get('class') and 'tei-pb' in elem.get('class'):
                    break
                # Skip anchors with class "tei-anchor"
                if elem.name == 'a' and elem.get('class') and 'tei-anchor' in elem.get('class'):
                    continue
                content_siblings.append(elem)
            elif isinstance(elem, NavigableString) and elem.strip():
                content_siblings.append(elem)
        # Process the collected siblings to extract paragraphs.
        paragraphs = extract_paragraphs_from_siblings(content_siblings)
        pages.append({"page_number": page_num, "paragraphs": paragraphs})
    return pages

def main():
    input_path = 'start_v1_of_3.html'   # Replace with your HTML file path.
    output_path = 'start_v1_of_3.json'  # Output JSON file.
    
    with open(input_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    pages = extract_pages(soup)
    
    result = {"pages": pages}
    output_json = json.dumps(result, indent=4, ensure_ascii=False)
    print(output_json)
    
    with open(output_path, 'w', encoding='utf-8') as out_file:
        out_file.write(output_json)

if __name__ == '__main__':
    main()
