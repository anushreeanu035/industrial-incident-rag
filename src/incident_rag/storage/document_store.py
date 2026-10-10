"""Persistent metadata store for ingested documents."""

import json
from pathlib import Path
from typing import Dict, List, Optional
from incident_rag.config import get_settings
from incident_rag.logging_config import get_logger
from incident_rag.schemas import DocumentMetadata

logger = get_logger(__name__)


class DocumentStore:
    """Manages document metadata persistence on disk."""

    def __init__(self, store_path: Optional[Path] = None):
        settings = get_settings()
        self.store_path = store_path or settings.resolve_path(settings.DOCUMENT_STORE_PATH)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._documents: Dict[str, DocumentMetadata] = {}
        self._load()

    def _load(self) -> None:
        if self.store_path.exists():
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._documents = {
                        k: DocumentMetadata.model_validate(v) for k, v in data.items()
                    }
                logger.info(f"Loaded {len(self._documents)} documents from {self.store_path}")
            except Exception as e:
                logger.error(f"Error loading document store from {self.store_path}: {e}")
                self._documents = {}
        else:
            self._documents = {}

    def _save(self) -> None:
        try:
            data = {k: v.model_dump() for k, v in self._documents.items()}
            with open(self.store_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save document store: {e}")

    def add_document(self, metadata: DocumentMetadata) -> None:
        self._documents[metadata.document_id] = metadata
        self._save()

    def get_document(self, document_id: str) -> Optional[DocumentMetadata]:
        return self._documents.get(document_id)

    def list_documents(self) -> List[DocumentMetadata]:
        return list(self._documents.values())

    def find_by_checksum(self, checksum: str) -> Optional[DocumentMetadata]:
        for doc in self._documents.values():
            if doc.checksum_sha256 == checksum:
                return doc
        return None

    def delete_document(self, document_id: str) -> bool:
        if document_id in self._documents:
            del self._documents[document_id]
            self._save()
            return True
        return False
