"""In-memory tracking of active binary chunk transfer sessions."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional


@dataclass
class ChunkReceipt:
    sequence: int
    offset: int
    size: int
    checksum: str
    received_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TransferSession:
    job_id: str
    run_id: str
    image_id: str
    total_chunks: int = 0
    expected_total_size: int = 0
    file_checksum: str = ""
    received_chunks: Dict[int, ChunkReceipt] = field(default_factory=dict)
    is_assembled: bool = False
    is_verified: bool = False
    is_last_seen: bool = False

    def add_chunk(
        self,
        sequence: int,
        offset: int,
        size: int,
        checksum: str = "",
    ) -> bool:
        """Add receipt of a chunk. Returns True if this chunk was not seen before."""
        if sequence in self.received_chunks:
            return False
        self.received_chunks[sequence] = ChunkReceipt(
            sequence=sequence,
            offset=offset,
            size=size,
            checksum=checksum,
        )
        return True

    @property
    def received_count(self) -> int:
        return len(self.received_chunks)

    def is_complete(self) -> bool:
        if self.total_chunks <= 0:
            return False
        return self.received_count == self.total_chunks and all(
            i in self.received_chunks for i in range(self.total_chunks)
        )
