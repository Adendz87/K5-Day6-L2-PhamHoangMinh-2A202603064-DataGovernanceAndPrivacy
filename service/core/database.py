"""SQLite database manager for persistent job state, chunk tracking, and results."""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple


class JobDatabase:
    """Manages SQLite storage for job metadata, transfer state, and results."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.initialize_schema()

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Provide a configured SQLite connection context with WAL mode."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=10.0,
            check_same_thread=False,
            isolation_level=None,  # Autocommit mode; transactions managed via BEGIN/COMMIT
        )
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            conn.execute("PRAGMA busy_timeout=5000;")
            yield conn
        finally:
            conn.close()

    def initialize_schema(self) -> None:
        """Create tables and indexes if they do not exist."""
        with self._lock, self.get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    client_id TEXT DEFAULT '',
                    dataset_name TEXT DEFAULT '',
                    dataset_version TEXT DEFAULT '',
                    total_images INTEGER DEFAULT 0,
                    total_annotations INTEGER DEFAULT 0,
                    status TEXT NOT NULL,
                    reason TEXT DEFAULT '',
                    metadata_json TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (job_id, run_id)
                );

                CREATE TABLE IF NOT EXISTS images (
                    job_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    image_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    total_chunks INTEGER DEFAULT 0,
                    received_chunks INTEGER DEFAULT 0,
                    sha256 TEXT DEFAULT '',
                    file_path TEXT DEFAULT '',
                    error_message TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (job_id, run_id, image_id),
                    FOREIGN KEY (job_id, run_id) REFERENCES jobs (job_id, run_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS chunk_receipts (
                    job_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    image_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    chunk_size INTEGER NOT NULL,
                    sha256 TEXT DEFAULT '',
                    received_at TEXT NOT NULL,
                    PRIMARY KEY (job_id, run_id, image_id, chunk_index),
                    FOREIGN KEY (job_id, run_id, image_id) REFERENCES images (job_id, run_id, image_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS annotation_batches (
                    job_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    batch_id TEXT NOT NULL,
                    item_count INTEGER NOT NULL,
                    received_at TEXT NOT NULL,
                    PRIMARY KEY (job_id, run_id, batch_id),
                    FOREIGN KEY (job_id, run_id) REFERENCES jobs (job_id, run_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS annotations (
                    job_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    ann_id TEXT NOT NULL,
                    image_id TEXT NOT NULL,
                    class_name TEXT NOT NULL,
                    x REAL NOT NULL,
                    y REAL NOT NULL,
                    w REAL NOT NULL,
                    h REAL NOT NULL,
                    confidence REAL DEFAULT 0.0,
                    extra_json TEXT DEFAULT '{}',
                    PRIMARY KEY (job_id, run_id, ann_id),
                    FOREIGN KEY (job_id, run_id) REFERENCES jobs (job_id, run_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS result_sequences (
                    job_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    result_type TEXT NOT NULL,
                    ref_id TEXT NOT NULL,
                    delivery_state TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    persisted_at TEXT NOT NULL,
                    PRIMARY KEY (job_id, run_id, sequence),
                    FOREIGN KEY (job_id, run_id) REFERENCES jobs (job_id, run_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS idempotency_keys (
                    idempotency_key TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_images_status ON images (job_id, run_id, status);
                CREATE INDEX IF NOT EXISTS idx_images_image_id ON images (image_id);
                CREATE INDEX IF NOT EXISTS idx_annotations_image ON annotations (job_id, run_id, image_id);
                CREATE INDEX IF NOT EXISTS idx_results_delivery ON result_sequences (job_id, run_id, delivery_state);
                """
            )

    # ─────────────────────────────────────────────
    # Job operations
    # ─────────────────────────────────────────────
    def create_job(
        self,
        job_id: str,
        run_id: str,
        client_id: str = "",
        dataset_name: str = "",
        dataset_version: str = "",
        total_images: int = 0,
        total_annotations: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        initial_status: str = "CREATED",
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        meta_json = json.dumps(metadata or {})
        with self._lock, self.get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute(
                """
                INSERT INTO jobs (
                    job_id, run_id, client_id, dataset_name, dataset_version,
                    total_images, total_annotations, status, metadata_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id, run_id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    total_images = CASE WHEN excluded.total_images > 0 THEN excluded.total_images ELSE jobs.total_images END,
                    total_annotations = CASE WHEN excluded.total_annotations > 0 THEN excluded.total_annotations ELSE jobs.total_annotations END;
                """,
                (
                    job_id,
                    run_id,
                    client_id,
                    dataset_name,
                    dataset_version,
                    total_images,
                    total_annotations,
                    initial_status,
                    meta_json,
                    now,
                    now,
                ),
            )
            conn.execute("COMMIT;")

    def update_job_status(self, job_id: str, run_id: str, status: str, reason: str = "") -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self.get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, reason = ?, updated_at = ?
                WHERE job_id = ? AND run_id = ?;
                """,
                (status, reason, now, job_id, run_id),
            )
            conn.execute("COMMIT;")

    def get_job(self, job_id: str, run_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ? AND run_id = ?;",
                (job_id, run_id),
            ).fetchone()
            return dict(row) if row else None

    # ─────────────────────────────────────────────
    # Image & chunk operations
    # ─────────────────────────────────────────────
    def ensure_image(
        self,
        job_id: str,
        run_id: str,
        image_id: str,
        total_chunks: int = 0,
        status: str = "IMAGE_RECEIVING",
        expected_sha256: str = "",
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self.get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute(
                """
                INSERT INTO images (
                    job_id, run_id, image_id, status, total_chunks, sha256,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id, run_id, image_id) DO UPDATE SET
                    total_chunks = CASE WHEN excluded.total_chunks > 0 THEN excluded.total_chunks ELSE images.total_chunks END,
                    sha256 = CASE WHEN excluded.sha256 != '' THEN excluded.sha256 ELSE images.sha256 END,
                    updated_at = excluded.updated_at;
                """,
                (job_id, run_id, image_id, status, total_chunks, expected_sha256, now, now),
            )
            conn.execute("COMMIT;")

    def find_job_by_image_id(self, image_id: str) -> Optional[Dict[str, Any]]:
        """Look up active job_id and run_id for an image."""
        with self.get_connection() as conn:
            row = conn.execute(
                """
                SELECT job_id, run_id, status FROM images
                WHERE image_id = ?
                ORDER BY updated_at DESC LIMIT 1;
                """,
                (image_id,),
            ).fetchone()
            return dict(row) if row else None

    def is_image_complete(
        self, job_id: str, run_id: str, image_id: str
    ) -> Tuple[bool, int, str]:
        """Check if all chunks 0..total_chunks-1 are recorded for an image.
        
        Returns (is_complete, total_chunks, expected_sha256).
        """
        with self.get_connection() as conn:
            img = conn.execute(
                "SELECT total_chunks, received_chunks, sha256, status FROM images WHERE job_id = ? AND run_id = ? AND image_id = ?;",
                (job_id, run_id, image_id),
            ).fetchone()
            if not img or img["total_chunks"] <= 0 or img["received_chunks"] < img["total_chunks"]:
                return False, (img["total_chunks"] if img else 0), (img["sha256"] if img else "")

            tot = img["total_chunks"]
            rows = conn.execute(
                "SELECT chunk_index FROM chunk_receipts WHERE job_id = ? AND run_id = ? AND image_id = ?;",
                (job_id, run_id, image_id),
            ).fetchall()
            received_indices = set(r["chunk_index"] for r in rows)
            if set(range(tot)).issubset(received_indices):
                return True, tot, img["sha256"]
            return False, tot, img["sha256"]

    def get_chunk_receipt(
        self, job_id: str, run_id: str, image_id: str, chunk_index: int
    ) -> Optional[Dict[str, Any]]:
        """Retrieve chunk receipt record if present."""
        with self.get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM chunk_receipts
                WHERE job_id = ? AND run_id = ? AND image_id = ? AND chunk_index = ?;
                """,
                (job_id, run_id, image_id, chunk_index),
            ).fetchone()
            return dict(row) if row else None

    def check_chunk_status(
        self,
        job_id: str,
        run_id: str,
        image_id: str,
        chunk_index: int,
        sha256: str = "",
    ) -> Tuple[str, Optional[str]]:
        """Check status of chunk receipt.

        Returns:
            ("NEW", None) if not received yet.
            ("DUPLICATE_IDENTICAL", recorded_sha256) if already received with identical SHA-256.
            ("DUPLICATE_CONFLICT", recorded_sha256) if already received with differing SHA-256.
        """
        receipt = self.get_chunk_receipt(job_id, run_id, image_id, chunk_index)
        if not receipt:
            return "NEW", None

        recorded_sha = receipt.get("sha256", "")
        if sha256 and recorded_sha and sha256.lower() != recorded_sha.lower():
            return "DUPLICATE_CONFLICT", recorded_sha
        return "DUPLICATE_IDENTICAL", recorded_sha

    def delete_chunk_receipt(
        self, job_id: str, run_id: str, image_id: str, chunk_index: int
    ) -> bool:
        """Delete chunk receipt and decrement received_chunks if it existed.
        
        Returns True if deleted, False if no such receipt existed.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self.get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            existing = conn.execute(
                """
                SELECT 1 FROM chunk_receipts
                WHERE job_id = ? AND run_id = ? AND image_id = ? AND chunk_index = ?;
                """,
                (job_id, run_id, image_id, chunk_index),
            ).fetchone()
            if not existing:
                conn.execute("COMMIT;")
                return False

            conn.execute(
                """
                DELETE FROM chunk_receipts
                WHERE job_id = ? AND run_id = ? AND image_id = ? AND chunk_index = ?;
                """,
                (job_id, run_id, image_id, chunk_index),
            )
            conn.execute(
                """
                UPDATE images
                SET received_chunks = CASE WHEN received_chunks > 0 THEN received_chunks - 1 ELSE 0 END,
                    updated_at = ?
                WHERE job_id = ? AND run_id = ? AND image_id = ?;
                """,
                (now, job_id, run_id, image_id),
            )
            conn.execute("COMMIT;")
            return True

    def record_chunk(
        self,
        job_id: str,
        run_id: str,
        image_id: str,
        chunk_index: int,
        chunk_size: int,
        sha256: str = "",
    ) -> bool:
        """Record chunk receipt idempotently.
        
        Returns True if chunk was newly inserted, False if identical duplicate.
        Raises ValueError if chunk already exists with a different SHA-256.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self.get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            # Check existing chunk
            existing = conn.execute(
                """
                SELECT sha256 FROM chunk_receipts
                WHERE job_id = ? AND run_id = ? AND image_id = ? AND chunk_index = ?;
                """,
                (job_id, run_id, image_id, chunk_index),
            ).fetchone()
            if existing:
                recorded_sha = existing["sha256"]
                conn.execute("COMMIT;")
                if sha256 and recorded_sha and recorded_sha.lower() != sha256.lower():
                    raise ValueError(
                        f"Conflicting chunk content for chunk {chunk_index} of image {image_id}: "
                        f"recorded={recorded_sha}, incoming={sha256}"
                    )
                return False  # Already received identical chunk

            conn.execute(
                """
                INSERT INTO chunk_receipts (
                    job_id, run_id, image_id, chunk_index, chunk_size, sha256, received_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (job_id, run_id, image_id, chunk_index, chunk_size, sha256, now),
            )
            conn.execute(
                """
                UPDATE images
                SET received_chunks = received_chunks + 1, updated_at = ?
                WHERE job_id = ? AND run_id = ? AND image_id = ?;
                """,
                (now, job_id, run_id, image_id),
            )
            conn.execute("COMMIT;")
            return True

    def mark_image_received(
        self,
        job_id: str,
        run_id: str,
        image_id: str,
        sha256: str,
        file_path: str,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self.get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute(
                """
                UPDATE images
                SET status = 'IMAGE_RECEIVED', sha256 = ?, file_path = ?, updated_at = ?
                WHERE job_id = ? AND run_id = ? AND image_id = ?;
                """,
                (sha256, file_path, now, job_id, run_id, image_id),
            )
            conn.execute("COMMIT;")

    def update_image_status(
        self,
        job_id: str,
        run_id: str,
        image_id: str,
        status: str,
        error_msg: str = "",
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self.get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute(
                """
                UPDATE images
                SET status = ?, error_message = ?, updated_at = ?
                WHERE job_id = ? AND run_id = ? AND image_id = ?;
                """,
                (status, error_msg, now, job_id, run_id, image_id),
            )
            conn.execute("COMMIT;")

    def get_image(self, job_id: str, run_id: str, image_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM images WHERE job_id = ? AND run_id = ? AND image_id = ?;",
                (job_id, run_id, image_id),
            ).fetchone()
            return dict(row) if row else None

    def get_image_ids_by_status(
        self, job_id: str, run_id: str, status: str | List[str]
    ) -> List[str]:
        with self.get_connection() as conn:
            if isinstance(status, (list, tuple)):
                placeholders = ",".join(["?"] * len(status))
                rows = conn.execute(
                    f"SELECT image_id FROM images WHERE job_id = ? AND run_id = ? AND status IN ({placeholders});",
                    (job_id, run_id, *status),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT image_id FROM images WHERE job_id = ? AND run_id = ? AND status = ?;",
                    (job_id, run_id, status),
                ).fetchall()
            return [r["image_id"] for r in rows]

    def get_all_images(self, job_id: str, run_id: str) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM images WHERE job_id = ? AND run_id = ? ORDER BY image_id;",
                (job_id, run_id),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_last_chunk_indices(self, job_id: str, run_id: str) -> Dict[str, int]:
        with self.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT image_id, MAX(chunk_index) as last_idx
                FROM chunk_receipts
                WHERE job_id = ? AND run_id = ?
                GROUP BY image_id;
                """,
                (job_id, run_id),
            ).fetchall()
            return {r["image_id"]: r["last_idx"] for r in rows}

    # ─────────────────────────────────────────────
    # Annotations
    # ─────────────────────────────────────────────
    def save_annotation_batch(
        self,
        job_id: str,
        run_id: str,
        batch_id: str,
        items: List[Dict[str, Any]],
    ) -> bool:
        """Save a batch of annotations idempotently."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self.get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            existing = conn.execute(
                "SELECT batch_id FROM annotation_batches WHERE job_id = ? AND run_id = ? AND batch_id = ?;",
                (job_id, run_id, batch_id),
            ).fetchone()
            if existing:
                conn.execute("COMMIT;")
                return False  # Already processed

            conn.execute(
                "INSERT INTO annotation_batches (job_id, run_id, batch_id, item_count, received_at) VALUES (?, ?, ?, ?, ?);",
                (job_id, run_id, batch_id, len(items), now),
            )
            for it in items:
                extra_json = json.dumps(it.get("extra_fields", {}))
                bbox = it.get("bbox", {})
                conn.execute(
                    """
                    INSERT INTO annotations (
                        job_id, run_id, ann_id, image_id, class_name,
                        x, y, w, h, confidence, extra_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(job_id, run_id, ann_id) DO UPDATE SET
                        class_name = excluded.class_name,
                        x = excluded.x, y = excluded.y, w = excluded.w, h = excluded.h;
                    """,
                    (
                        job_id,
                        run_id,
                        str(it["ann_id"]),
                        str(it["image_id"]),
                        str(it["class_name"]),
                        float(bbox.get("x", 0.0)),
                        float(bbox.get("y", 0.0)),
                        float(bbox.get("w", 0.0)),
                        float(bbox.get("h", 0.0)),
                        float(it.get("confidence", 0.0)),
                        extra_json,
                    ),
                )
            conn.execute("COMMIT;")
            return True

    def get_annotations_for_image(
        self, job_id: str, run_id: str, image_id: str
    ) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM annotations WHERE job_id = ? AND run_id = ? AND image_id = ?;",
                (job_id, run_id, image_id),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_all_annotations(self, job_id: str, run_id: str) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM annotations WHERE job_id = ? AND run_id = ?;",
                (job_id, run_id),
            ).fetchall()
            return [dict(r) for r in rows]

    # ─────────────────────────────────────────────
    # Result sequences & persistence
    # ─────────────────────────────────────────────
    def next_result_sequence(self, job_id: str, run_id: str) -> int:
        with self._lock, self.get_connection() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 AS next_seq FROM result_sequences WHERE job_id = ? AND run_id = ?;",
                (job_id, run_id),
            ).fetchone()
            return int(row["next_seq"])

    def persist_result(
        self,
        job_id: str,
        run_id: str,
        sequence: int,
        result_type: str,
        ref_id: str,
        payload: Dict[str, Any],
        delivery_state: str = "PERSISTED",
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        payload_json = json.dumps(payload)
        with self._lock, self.get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute(
                """
                INSERT INTO result_sequences (
                    job_id, run_id, sequence, result_type, ref_id,
                    delivery_state, payload_json, persisted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id, run_id, sequence) DO UPDATE SET
                    delivery_state = excluded.delivery_state,
                    payload_json = excluded.payload_json;
                """,
                (
                    job_id,
                    run_id,
                    sequence,
                    result_type,
                    ref_id,
                    delivery_state,
                    payload_json,
                    now,
                ),
            )
            conn.execute("COMMIT;")

    def update_delivery_state(
        self,
        job_id: str,
        run_id: str,
        max_sequence: int,
        new_state: str = "ACKNOWLEDGED",
    ) -> None:
        with self._lock, self.get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute(
                """
                UPDATE result_sequences
                SET delivery_state = ?
                WHERE job_id = ? AND run_id = ? AND sequence <= ?;
                """,
                (new_state, job_id, run_id, max_sequence),
            )
            conn.execute("COMMIT;")

    def get_results_from_sequence(
        self, job_id: str, run_id: str, from_sequence: int = 1
    ) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM result_sequences
                WHERE job_id = ? AND run_id = ? AND sequence >= ?
                ORDER BY sequence ASC;
                """,
                (job_id, run_id, from_sequence),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_max_persisted_result_sequence(self, job_id: str, run_id: str) -> int:
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(sequence), 0) AS max_seq FROM result_sequences WHERE job_id = ? AND run_id = ?;",
                (job_id, run_id),
            ).fetchone()
            return int(row["max_seq"])

    # ─────────────────────────────────────────────
    # Idempotency
    # ─────────────────────────────────────────────
    def check_idempotency_key(self, key: str) -> Optional[Dict[str, str]]:
        if not key:
            return None
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM idempotency_keys WHERE idempotency_key = ?;", (key,)
            ).fetchone()
            return dict(row) if row else None

    def register_idempotency_key(self, key: str, job_id: str, run_id: str) -> None:
        if not key:
            return
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self.get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            conn.execute(
                "INSERT OR IGNORE INTO idempotency_keys (idempotency_key, job_id, run_id, created_at) VALUES (?, ?, ?, ?);",
                (key, job_id, run_id, now),
            )
            conn.execute("COMMIT;")
