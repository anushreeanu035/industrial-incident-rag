"""BM25 keyword search engine using rank-bm25 with technical identifier preservation."""

import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
from rank_bm25 import BM25Okapi
from incident_rag.config import get_settings
from incident_rag.logging_config import get_logger
from incident_rag.schemas import DocumentChunk, RetrievalResultItem

logger = get_logger(__name__)

# Tokenization regex that keeps compound technical codes (e.g., HP-17, RX-404, V-204, 14.2, 850ppm)
TOKEN_REGEX = re.compile(r"[A-Za-z0-9]+(?:[.\-_/][A-Za-z0-9]+)*")


def tokenize_technical_text(text: str) -> List[str]:
    """Tokenizes text while preserving technical machine tags, error codes, and numbers."""
    if not text:
        return []
    tokens = TOKEN_REGEX.findall(text.lower())
    return tokens


class BM25SearchEngine:
    """Keyword search engine using BM25Okapi with persistent chunk corpus."""

    def __init__(self, index_path: Optional[Union[str, Path]] = None):
        settings = get_settings()
        self.index_path = Path(index_path) if index_path else settings.resolve_path(settings.BM25_INDEX_PATH)
        self.chunks: List[DocumentChunk] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self._load()

    def _load(self) -> None:
        """Loads indexed chunks and reinitializes BM25 from disk if present."""
        if self.index_path.exists():
            try:
                with open(self.index_path, "rb") as f:
                    data = pickle.load(f)
                    self.chunks = data.get("chunks", [])
                    self._build_index()
                logger.info(f"Loaded BM25 index with {len(self.chunks)} chunks from {self.index_path}")
            except Exception as e:
                logger.error(f"Failed to load BM25 index from {self.index_path}: {e}")
                self.chunks = []
                self.tokenized_corpus = []
                self.bm25 = None
        else:
            self.chunks = []
            self.tokenized_corpus = []
            self.bm25 = None

    def _save(self) -> None:
        """Persists indexed chunks to disk."""
        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.index_path, "wb") as f:
                pickle.dump({"chunks": self.chunks}, f)
        except Exception as e:
            logger.error(f"Failed to persist BM25 index: {e}")

    def _build_index(self) -> None:
        """Reconstructs BM25 index over the current chunk list."""
        if not self.chunks:
            self.tokenized_corpus = []
            self.bm25 = None
            return

        self.tokenized_corpus = [
            tokenize_technical_text(chunk.text) for chunk in self.chunks
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # Standard Lucene/Elasticsearch non-negative IDF: ln(1 + (N - n + 0.5) / (n + 0.5))
        # This prevents zero/negative IDF scores on small collections or high-frequency technical tags
        import math
        n_docs = len(self.tokenized_corpus)
        term_doc_freq = {}
        for doc in self.tokenized_corpus:
            for term in set(doc):
                term_doc_freq[term] = term_doc_freq.get(term, 0) + 1

        lucene_idf = {}
        for term, freq in term_doc_freq.items():
            lucene_idf[term] = math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
        self.bm25.idf = lucene_idf

    def add_chunks(self, new_chunks: List[DocumentChunk]) -> None:
        """Adds new chunks to the index, avoiding duplicates."""
        existing_ids = {c.chunk_id for c in self.chunks}
        to_add = [c for c in new_chunks if c.chunk_id not in existing_ids]

        if not to_add:
            return

        self.chunks.extend(to_add)
        self._build_index()
        self._save()
        logger.info(f"Added {len(to_add)} chunks to BM25 index. Total: {len(self.chunks)}")

    def remove_document_chunks(self, document_id: str) -> None:
        """Removes all chunks belonging to a document ID."""
        initial_count = len(self.chunks)
        self.chunks = [c for c in self.chunks if c.document_id != document_id]
        if len(self.chunks) < initial_count:
            self._build_index()
            self._save()
            logger.info(f"Removed chunks for doc {document_id} from BM25 index")

    def clear(self) -> None:
        """Clears all chunks from the BM25 index."""
        self.chunks = []
        self.tokenized_corpus = []
        self.bm25 = None
        self._save()

    def search(
        self,
        query: str,
        top_k: int = 10,
        machine_id_filter: Optional[str] = None,
        incident_type_filter: Optional[str] = None,
    ) -> List[RetrievalResultItem]:
        """Performs BM25 keyword scoring over candidate chunks with optional metadata filters."""
        if not self.bm25 or not self.chunks or not query.strip():
            return []

        tokenized_query = tokenize_technical_text(query)
        if not tokenized_query:
            return []

        doc_scores = self.bm25.get_scores(tokenized_query)

        # Pair scores with chunks and apply filters
        scored_candidates = []
        for idx, (chunk, score) in enumerate(zip(self.chunks, doc_scores)):
            if score <= 0.0:
                continue

            # Apply metadata filters
            if machine_id_filter and chunk.metadata.machine_id != machine_id_filter:
                continue
            if incident_type_filter and chunk.metadata.incident_type != incident_type_filter:
                continue

            scored_candidates.append((score, chunk))

        # Sort descending by BM25 score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        results: List[RetrievalResultItem] = []
        for rank_idx, (score, chunk) in enumerate(scored_candidates[:top_k], start=1):
            results.append(
                RetrievalResultItem(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    bm25_score=float(score),
                    bm25_rank=rank_idx,
                    final_rank=rank_idx,
                )
            )

        return results
