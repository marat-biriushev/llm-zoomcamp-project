import re

import pymupdf
import requests
from minsearch import Index

PDF_URL = 'https://www.middlebury.edu/sites/default/files/2025-01/PCI-DSS-v4_0_1.pdf'
PDF_PATH = 'data/pci-dss-v4_0_1.pdf'

# Requirement numbers look like "1.2.3", "12.10.7" or "A1.1.1"
REQ_ID_PATTERN = re.compile(r'^(\d{1,2}\.\d{1,2}(?:\.\d{1,2})*|A\d\.\d[\d.]*)\s', re.MULTILINE)


def download_pdf(url=PDF_URL, path=PDF_PATH):
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        return path

    response = requests.get(url, timeout=120)
    response.raise_for_status()

    with open(path, 'wb') as f_out:
        f_out.write(response.content)

    return path


def load_documents(path=PDF_PATH):
    pdf = pymupdf.open(path)

    documents = []

    for page_number, page in enumerate(pdf, start=1):
        text = page.get_text()

        # Keep only the pages of the "Requirements and Testing Procedures"
        # section. Every one of them has this header; the table of contents,
        # the intro and the closing appendices do not.
        if 'Defined Approach Requirements' not in text:
            continue

        req_ids = sorted(set(REQ_ID_PATTERN.findall(text)))

        documents.append({
            'page': page_number,
            'req_ids': ', '.join(req_ids),
            'requirement': req_ids[0].split('.')[0] if req_ids else '',
            'text': text.strip(),
        })

    return documents


def build_index(documents):
    index = Index(
        text_fields=['text', 'req_ids'],
        keyword_fields=['requirement']
    )
    index.fit(documents)
    return index
