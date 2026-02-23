"""Steps 1-3: Content-addressable storage + deterministic extraction."""

import hashlib
import io
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

try:
    import pdfplumber
except ImportError:
    pdfplumber = None
try:
    import pytesseract
except ImportError:
    pytesseract = None
try:
    from PIL import Image
except ImportError:
    Image = None


class DocumentProcessor:
    """Steps 1-3: Content-addressable storage + deterministic extraction."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.raw_dir = Path("data/raw")

    def compute_hash(self, content: bytes) -> str:
        """Step 1: SHA-256 content addressing."""
        return hashlib.sha256(content).hexdigest()

    def hash_exists(self, file_hash: str) -> bool:
        """Idempotency check."""
        prefix = file_hash[:2]
        return (self.raw_dir / prefix / f"{file_hash}.json").exists()

    def process(
        self, file: Any, file_hash: str, ocr_enabled: bool = True
    ) -> Dict[str, Any]:
        """Step 2-3: Extract and chunk deterministically."""
        content = file.getvalue() if hasattr(file, "getvalue") else file.read()
        name = getattr(file, "name", "unknown")
        ext = name.split(".")[-1].lower() if "." in name else "txt"

        prefix = file_hash[:2]
        save_dir = self.raw_dir / prefix
        save_dir.mkdir(parents=True, exist_ok=True)

        if ext == "pdf":
            text = self._extract_pdf(content, ocr_enabled)
        elif ext == "csv":
            text = self._extract_csv(content)
        elif ext == "xlsx":
            text = self._extract_excel(content)
        else:
            text = content.decode("utf-8", errors="replace")

        chunks = self._chunk_text(text, file_hash)

        manifest = {
            "filename": name,
            "hash": file_hash,
            "type": ext,
            "chunks": len(chunks),
            "extracted_text_length": len(text),
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        with open(save_dir / f"{file_hash}.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        with open(save_dir / f"{file_hash}.bin", "wb") as f:
            f.write(content)

        return {
            "hash": file_hash,
            "filename": name,
            "text": text,
            "chunks": chunks,
            "type": ext,
        }

    def _extract_pdf(self, content: bytes, ocr: bool) -> str:
        """Deterministic PDF extraction."""
        if pdfplumber is None:
            raise ImportError("pdfplumber required. Run: pip install pdfplumber")
        text_parts = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"[Page {i+1}]\n{page_text}")
                elif ocr and pytesseract and Image:
                    try:
                        im = page.to_image(resolution=300).original
                        img_pil = im if hasattr(im, "save") else Image.fromarray(im)
                        ocr_text = pytesseract.image_to_string(
                            img_pil, config="--psm 6 --dpi 300"
                        )
                        text_parts.append(f"[Page {i+1} - OCR]\n{ocr_text}")
                    except Exception:
                        text_parts.append(f"[Page {i+1}]\n[OCR unavailable]")
        return "\n\n".join(text_parts) if text_parts else "[No text extracted]"

    def _extract_csv(self, content: bytes) -> str:
        """Deterministic CSV to text."""
        df = pd.read_csv(io.BytesIO(content))
        df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]
        return df.to_string(index=False)

    def _extract_excel(self, content: bytes) -> str:
        """Deterministic Excel extraction."""
        df = pd.read_excel(io.BytesIO(content))
        df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]
        return df.to_string(index=False)

    def _chunk_text(self, text: str, doc_hash: str) -> List[Dict[str, Any]]:
        """Step 3: Semantic chunking with overlap and metadata."""
        chars_per_chunk = self.chunk_size * 4
        overlap_chars = self.chunk_overlap * 4
        chunks = []
        start = 0
        chunk_idx = 0

        while start < len(text):
            end = start + chars_per_chunk
            if end < len(text):
                next_newline = text.find("\n\n", max(0, end - 50), end + 50)
                if next_newline != -1:
                    end = next_newline

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "doc_hash": doc_hash,
                    "chunk_index": chunk_idx,
                    "char_start": start,
                    "char_end": end,
                    "content_hash": hashlib.sha256(
                        chunk_text.encode()
                    ).hexdigest()[:16],
                })
                chunk_idx += 1
            start = end - overlap_chars

        return chunks
