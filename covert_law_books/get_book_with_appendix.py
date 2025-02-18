#!/usr/bin/env python3

"""
Module: appendices_special_case.py

This module parses an HTML containing appendices, each introduced by
an H3 with span.larger for the main title. It organizes them so that
each is an item in a JSON list. For Appendix VI specifically, we parse
out sub-sections named as "I. Convention for the Pacific Settlement...,"
"II. Convention respecting...," etc. We also capture the text content
for each sub-section.

This approach is an illustration; you may need to adjust for your exact
HTML structure. Also, if your numbering or tagging is different, adapt
the code or the regex. The user’s example includes Roman numerals I..XV.
"""

import re
import json
from bs4 import BeautifulSoup

def extract_footnote_text(div_footnote) -> str:
    """
    Extracts footnote text as a bracketed string:
      [Footnote XX: actual text]
    """
    paras = div_footnote.find_all("p")
    full_text = " ".join(p.get_text(" ", strip=True) for p in paras)
    label_tag = div_footnote.find("span", class_="label")
    label = label_tag.get_text(strip=True) if label_tag else ""
    if label.startswith("[") and label.endswith("]"):
        label = label[1:-1]
    prefix = f"[{label}]"
    if full_text.startswith(prefix):
        content_text = full_text[len(prefix):].strip()
    else:
        content_text = full_text
    return f"[Footnote {label}: {content_text}]"


def parse_appendix_content(appendix_header, do_subsections=False):
    """
    Gathers all siblings after 'appendix_header' (H3) until
    the next H2/H3. If do_subsections=True, we look for lines that match
    a "Roman Numeral + dot + text" pattern to break them into sub-sections.

    Returns:
        {
          "appendix_title": "...",
          "content": "...",              # if do_subsections=False
          "sections": [ { "subsection_title": "...", "content": "..." }, ... ] # if do_subsections=True
        }
    """

    # Build a single title from all <span> or fallback to .get_text()
    spans = appendix_header.find_all("span")
    if spans:
        app_title = " ".join(sp.get_text(" ", strip=True) for sp in spans)
    else:
        app_title = appendix_header.get_text(" ", strip=True)

    if not do_subsections:
        # Just gather everything into a single "content" string
        content_parts = []
        pointer = appendix_header
        while True:
            pointer = pointer.next_sibling
            if pointer is None:
                break
            if isinstance(pointer, str):
                txt = pointer.strip()
                if txt:
                    content_parts.append(txt)
                continue
            if pointer.name in ["h2", "h3"]:
                break
            if pointer.name == "p":
                content_parts.append(pointer.get_text(" ", strip=True))
                continue
            if pointer.name == "div":
                classes = pointer.get("class") or []
                if "footnote" in classes:
                    foot_txt = extract_footnote_text(pointer)
                    content_parts.append(foot_txt)
                elif "sidenote" in classes:
                    note = " ".join(p.get_text(" ", strip=True) for p in pointer.find_all("p"))
                    content_parts.append(f"[Sidenote: {note}]")
                else:
                    content_parts.append(pointer.get_text(" ", strip=True))
                continue

            # default fallback
            txt2 = pointer.get_text(" ", strip=True)
            if txt2:
                content_parts.append(txt2)

        all_content = "\n\n".join(content_parts)
        return {
            "appendix_title": app_title,
            "content": all_content
        }
    else:
        # We want sub-sections, e.g. "I. Convention..." "II. Convention...", etc.
        # We'll gather text lines. If a line matches "^(I|II|III|X|V) ...", start a new subsection.
        # We'll store them in sections: each has a 'subsection_title' and 'content'.
        roman_pattern = re.compile(r'^([IVXLC]+)\.\s+(.*)$')
        # We'll keep a list of subsections. Each is a dict with
        # { "subsection_title": "I. Convention...", "content": "" }
        # We'll keep a pointer to the current subsection we are populating.
        subsections = []
        current_sub = None

        pointer = appendix_header
        while True:
            pointer = pointer.next_sibling
            if pointer is None:
                break
            if isinstance(pointer, str):
                txt = pointer.strip()
                if txt:
                    # check if it matches
                    match = roman_pattern.match(txt)
                    if match:
                        # new subsection
                        # close out old subsection
                        if current_sub:
                            subsections.append(current_sub)
                        # start new
                        sub_title = f"{match.group(1)}. {match.group(2)}"
                        current_sub = {"subsection_title": sub_title, "content": ""}
                    else:
                        # append to current if exist
                        if current_sub:
                            cur_text = current_sub["content"]
                            current_sub["content"] = cur_text + "\n" + txt if cur_text else txt
                continue

            if pointer.name in ["h2", "h3"]:
                break

            # If <p>
            if pointer.name == "p":
                txt = pointer.get_text(" ", strip=True)
                # check if it matches
                match = roman_pattern.match(txt)
                if match:
                    # new subsection
                    if current_sub:
                        subsections.append(current_sub)
                    sub_title = f"{match.group(1)}. {match.group(2)}"
                    current_sub = {"subsection_title": sub_title, "content": ""}
                else:
                    if current_sub:
                        if current_sub["content"]:
                            current_sub["content"] += "\n" + txt
                        else:
                            current_sub["content"] = txt
                    else:
                        # if no subsection yet, create a 'misc' subsection or something
                        # or just skip. We'll store in 'misc' maybe:
                        if not subsections or "subsection_title" in subsections[-1]:
                            # create a pseudo-subsection
                            subsections.append({"subsection_title": "Intro", "content": txt})
                        else:
                            # append
                            subsections[-1]["content"] += "\n" + txt
                continue

            if pointer.name == "div":
                classes = pointer.get("class") or []
                if "footnote" in classes:
                    foot_txt = extract_footnote_text(pointer)
                    if current_sub:
                        if current_sub["content"]:
                            current_sub["content"] += "\n" + foot_txt
                        else:
                            current_sub["content"] = foot_txt
                    else:
                        # store in a pseudo-subsection if none
                        if not subsections or "subsection_title" in subsections[-1]:
                            subsections.append({"subsection_title": "Intro", "content": foot_txt})
                        else:
                            subsections[-1]["content"] += "\n" + foot_txt
                elif "sidenote" in classes:
                    note = " ".join(p.get_text(" ", strip=True) for p in pointer.find_all("p"))
                    side_line = f"[Sidenote: {note}]"
                    if current_sub:
                        if current_sub["content"]:
                            current_sub["content"] += "\n" + side_line
                        else:
                            current_sub["content"] = side_line
                    else:
                        if not subsections or "subsection_title" in subsections[-1]:
                            subsections.append({"subsection_title": "Intro", "content": side_line})
                        else:
                            subsections[-1]["content"] += "\n" + side_line
                else:
                    # general <div>
                    block_txt = pointer.get_text(" ", strip=True)
                    if current_sub:
                        if current_sub["content"]:
                            current_sub["content"] += "\n" + block_txt
                        else:
                            current_sub["content"] = block_txt
                    else:
                        if not subsections or "subsection_title" in subsections[-1]:
                            subsections.append({"subsection_title": "Intro", "content": block_txt})
                        else:
                            subsections[-1]["content"] += "\n" + block_txt
                continue

            # default fallback
            fallback_txt = pointer.get_text(" ", strip=True)
            if fallback_txt:
                if current_sub:
                    if current_sub["content"]:
                        current_sub["content"] += "\n" + fallback_txt
                    else:
                        current_sub["content"] = fallback_txt
                else:
                    if not subsections or "subsection_title" in subsections[-1]:
                        subsections.append({"subsection_title": "Intro", "content": fallback_txt})
                    else:
                        subsections[-1]["content"] += "\n" + fallback_txt

        # after loop ends, close out last subsection
        if current_sub:
            subsections.append(current_sub)

        return {
            "appendix_title": app_title,
            "sections": subsections
        }

def extract_appendices(html_content: str):
    """
    1) Finds the H2 with "APPENDICES"
    2) For each H3 after that H2, we parse an appendix
       - if the H3 text is "APPENDIX VI" we parse with do_subsections=True,
         else false.
    Returns a list of appendices (dict).
    """

    soup = BeautifulSoup(html_content, "html.parser")
    # find H2 with "APPENDICES"
    h2_app = None
    for candidate in soup.find_all("h2"):
        span_larger = candidate.find("span", class_="larger")
        if span_larger and "APPENDICES" in span_larger.get_text(strip=True).upper():
            h2_app = candidate
            break

    if not h2_app:
        return []

    # gather appendices
    appendices = []
    for elem in h2_app.find_all_next():
        if elem.name == "h2" and elem != h2_app:
            break
        if elem.name == "h3":
            # check if this is "APPENDIX VI"
            # we can read the text from the <span class="larger"> or from entire H3
            spans = elem.find_all("span", class_="larger")
            if spans:
                app_larger_txt = " ".join(sp.get_text(" ", strip=True) for sp in spans)
            else:
                app_larger_txt = elem.get_text(" ", strip=True)

            # We see if the text includes "VI"
            # e.g. "APPENDIX VI"
            # we'll do something like:
            do_sub = False
            if "APPENDIX VI" in app_larger_txt.upper():
                do_sub = True

            app_dict = parse_appendix_content(elem, do_subsections=do_sub)
            appendices.append(app_dict)

    return appendices

def main():
    import sys

    input_file = "pg41047-images.html"
    with open(input_file, "r", encoding="utf-8") as f:
        data = f.read()

    app_list = extract_appendices(data)

    # print or write to JSON
    output_json = json.dumps(app_list, indent=4, ensure_ascii=False)
    print(output_json)
    with open("pg41047_appendices.json", "w", encoding="utf-8") as out_file:
        out_file.write(output_json)

if __name__ == "__main__":
    main()
