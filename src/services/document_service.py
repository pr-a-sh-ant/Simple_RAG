from pypdf import PdfReader
import docx
import csv
from bs4 import BeautifulSoup
from pathlib import Path
import shutil
import uuid
import pytesseract
from PIL import Image
from src.utils.text_splitter import TextSplitter

def load_text(path):
    return Path(path).read_text(encoding="utf-8", errors="ignore")

def load_md(path):
    return Path(path).read_text(encoding="utf-8", errors="ignore")

def load_pdf(path):
    return "\n\n".join((p.extract_text() or "") for p in PdfReader(path).pages)

def load_docx(path):
    return "\n".join(p.text for p in docx.Document(path).paragraphs)

def load_csv(path):
    with open(path, newline='', encoding="utf-8", errors="ignore") as csvfile:
        reader = csv.reader(csvfile)
        return "\n".join([", ".join(row) for row in reader])
    
def load_html(path):
    html = Path(path).read_text(encoding="utf-8", errors="ignore")
    return BeautifulSoup(html, "html.parser").get_text(separator="\n")

def load_image(path):
    return pytesseract.image_to_string(Image.open(path))

LOADERS = {
    ".txt":  load_text,
    ".md":   load_md,
    ".pdf":  load_pdf,
    ".docx": load_docx,
    ".doc":  load_docx,
    ".csv":  load_csv,
    ".html": load_html,
    ".htm":  load_html,
    ".png":  load_image,
    ".jpg":  load_image,
    ".jpeg": load_image,
}

class DocumentService:
    def __init__(self, upload_dir="uploads"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)

        self.text_splitter = TextSplitter(chunk_size=500, chunk_overlap=50)


    def save_file(self, file):
        file_path = self.upload_dir / f"{file.filename}"

        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return file_path
    
    def load_file(self, file_path):
        ext = Path(file_path).suffix.lower()
        loader_cls = LOADERS.get(ext)
        if loader_cls is None:
            raise ValueError(f"Unsupported file type: {ext}")
        return loader_cls(str(file_path))

    def split_documents(self, documents, source:str):
        chunks = self.text_splitter.split_text(documents, metadata={"source": source})
        return chunks
    
    def format_docs(self, docs):
        return "\n\n".join([doc["content"] for doc in docs])
    
    def format_context(self, results) -> str:
        parts = []
        for i, (text, meta, _) in enumerate(results, 1):
            source = meta.get("source", "unknown")
            page = meta.get("page")
            tag = f"[{i}] {source}" + (f" p.{page}" if page else "")
            parts.append(f"{tag}\n{text}")
        return "\n\n".join(parts)