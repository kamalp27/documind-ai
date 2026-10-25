"""Deterministic text chunking service with page-level metadata tracking."""

from typing import Any, Dict, List
from app.schemas.document import TextChunk


class ChunkingService:
    """Splits extracted page texts into overlapping chunks with citation metadata."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and strictly less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_page_text(self, text: str) -> List[str]:
        """Split a single text string into overlapping chunks, respecting word boundaries when possible."""
        cleaned = text.strip()
        if not cleaned:
            return []

        if len(cleaned) <= self.chunk_size:
            return [cleaned]

        chunks: List[str] = []
        start = 0
        text_len = len(cleaned)

        while start < text_len:
            end = start + self.chunk_size

            if end >= text_len:
                chunk = cleaned[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break

            # Attempt to snap to the nearest space or newline before the hard cut
            cutoff = cleaned.rfind(" ", start + self.chunk_size // 2, end)
            newline_cutoff = cleaned.rfind("\n", start + self.chunk_size // 2, end)
            best_cutoff = max(cutoff, newline_cutoff)

            if best_cutoff > start:
                chunk = cleaned[start:best_cutoff].strip()
                advance_to = best_cutoff
            else:
                chunk = cleaned[start:end].strip()
                advance_to = end

            if chunk:
                chunks.append(chunk)

            # Advance start with overlap
            start = max(start + 1, advance_to - self.chunk_overlap)

        return chunks

    def create_chunks(
        self,
        pages: List[Dict[str, Any]],
        doc_id: str,
        filename: str,
    ) -> List[TextChunk]:
        """Create structured TextChunk instances from extracted pages.

        Preserves 1-indexed page numbers and assigns sequential chunk indexes.
        """
        all_chunks: List[TextChunk] = []
        global_chunk_idx = 0

        for page in pages:
            page_num = page["page_number"]
            page_text = page.get("text", "")

            if not page_text or not page_text.strip():
                continue

            raw_chunks = self.split_page_text(page_text)
            for raw_chunk in raw_chunks:
                chunk_id = f"{doc_id}_p{page_num}_c{global_chunk_idx}"
                all_chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        filename=filename,
                        text=raw_chunk,
                        page_number=page_num,
                        chunk_index=global_chunk_idx,
                        char_count=len(raw_chunk),
                    )
                )
                global_chunk_idx += 1

        return all_chunks
