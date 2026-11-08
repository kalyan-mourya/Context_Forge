import json
import mimetypes
import os
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None

try:
    import magic
except ImportError:  # pragma: no cover - optional dependency
    magic = None

try:
    import pdfplumber
except ImportError:  # pragma: no cover - optional dependency
    pdfplumber = None

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - optional dependency
    BeautifulSoup = None


extension_func_map = {
    ".txt": "read_txt",
    ".pdf": "read_pdf",
    ".docx": "read_docx",
    ".doc": "read_doc",
    ".csv": "read_csv",
    ".xlsx": "read_xlsx",
    ".json": "read_json",
    ".md": "read_md",
    ".html": "read_html",
    ".htm": "read_html",
    ".pptx": "read_pptx",
}

mime_func_map = {
    "application/pdf": "read_pdf",
    "application/msword": "read_doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "read_docx",
    "text/plain": "read_txt",
    "application/json": "read_json",
    "text/html": "read_html",
}


def read_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_pdf(path):
    if pdfplumber is None:
        raise RuntimeError("pdfplumber is required to read PDF files")
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


def read_doc(path):
    return read_txt(path) if Path(path).suffix.lower() == ".txt" else ""


def read_docx(path):
    with ZipFile(path) as zf:
        xml_data = zf.read("word/document.xml")
    root = ET.fromstring(xml_data)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []
    for node in root.iterfind(".//w:t", ns):
        text = node.text or ""
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


def read_csv(path):
    if pd is None:
        raise RuntimeError("pandas is required to read CSV files")
    df = pd.read_csv(path)
    return df.to_string(index=False)


def read_xlsx(path):
    if pd is None:
        raise RuntimeError("pandas is required to read Excel files")
    df = pd.read_excel(path)
    return df.to_string(index=False)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.dumps(json.load(f), indent=2)


def read_md(path):
    return read_txt(path)


def read_html(path):
    with open(path, "r", encoding="utf-8") as f:
        if BeautifulSoup is None:
            return f.read()
        soup = BeautifulSoup(f, "html.parser")
        return soup.get_text()


def read_pptx(path):
    return read_docx(path) if Path(path).suffix.lower() == ".docx" else ""


def detect_and_read(file_path):
    file_path = str(file_path)
    ext = Path(file_path).suffix.lower()

    if ext in extension_func_map:
        func_name = extension_func_map[ext]
    else:
        mime = None
        if magic is not None:
            try:
                mime = magic.from_file(file_path, mime=True)
            except Exception:
                mime = None
        if mime is None:
            mime = mimetypes.guess_type(file_path)[0]
        func_name = mime_func_map.get(mime, None)

    if func_name:
        reader_func = globals().get(func_name)
        if reader_func:
            return reader_func(file_path)
        print(f"Function not implemented: {func_name}")
    else:
        print(f"Unsupported file format: {file_path}")
    return ""


def get_file_contents(folder_path):
    """Return concatenated text from supported files in a folder tree."""
    file_contents = []
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    for root, dirs, files in os.walk(folder_path):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in {"__pycache__", ".cache", ".git"})
        for filename in sorted(files):
            if filename.startswith("."):
                continue
            full_path = os.path.join(root, filename)
            if ".cache" in full_path or ".git" in full_path:
                continue
            try:
                text = detect_and_read(full_path)
                if text:
                    file_contents.append(text)
            except Exception as exc:
                print(f"Error reading {filename}: {exc}")

    return file_contents






