"""Persistent storage for evaluation results, evidence, and run summaries."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from service.core.database import JobDatabase


class ResultStorage:
    """Stores evaluation results on disk and indexes sequences in SQLite."""

    def __init__(self, base_path: str | Path, db: JobDatabase) -> None:
        self.base_path = Path(base_path).resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.db = db

    def get_run_results_dir(self, job_id: str, run_id: str) -> Path:
        p = self.base_path / job_id / run_id / "results"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def save_annotation_result(
        self,
        job_id: str,
        run_id: str,
        result_dict: Dict[str, Any],
    ) -> int:
        seq = self.db.next_result_sequence(job_id, run_id)
        result_dict["result_sequence"] = seq
        self.db.persist_result(
            job_id=job_id,
            run_id=run_id,
            sequence=seq,
            result_type="ANNOTATION_RESULT",
            ref_id=str(result_dict.get("ann_id", "")),
            payload=result_dict,
        )
        return seq

    def save_image_result(
        self,
        job_id: str,
        run_id: str,
        result_dict: Dict[str, Any],
    ) -> int:
        seq = self.db.next_result_sequence(job_id, run_id)
        result_dict["result_sequence"] = seq
        self.db.persist_result(
            job_id=job_id,
            run_id=run_id,
            sequence=seq,
            result_type="IMAGE_RESULT",
            ref_id=str(result_dict.get("image_id", "")),
            payload=result_dict,
        )
        return seq

    def save_missing_candidate(
        self,
        job_id: str,
        run_id: str,
        result_dict: Dict[str, Any],
    ) -> int:
        seq = self.db.next_result_sequence(job_id, run_id)
        result_dict["result_sequence"] = seq
        self.db.persist_result(
            job_id=job_id,
            run_id=run_id,
            sequence=seq,
            result_type="MISSING_CANDIDATE",
            ref_id=str(result_dict.get("image_id", "")),
            payload=result_dict,
        )
        return seq

    def save_job_result(
        self,
        job_id: str,
        run_id: str,
        result_dict: Dict[str, Any],
    ) -> int:
        seq = self.db.next_result_sequence(job_id, run_id)
        result_dict["result_sequence"] = seq
        self.db.persist_result(
            job_id=job_id,
            run_id=run_id,
            sequence=seq,
            result_type="JOB_RESULT",
            ref_id=job_id,
            payload=result_dict,
        )

        # Write summary JSON to disk
        out_file = self.get_run_results_dir(job_id, run_id) / "job_result.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)

        return seq

    def save_job_completed(
        self,
        job_id: str,
        run_id: str,
        completed_dict: Dict[str, Any],
    ) -> int:
        seq = self.db.next_result_sequence(job_id, run_id)
        completed_dict["result_sequence"] = seq
        self.db.persist_result(
            job_id=job_id,
            run_id=run_id,
            sequence=seq,
            result_type="JOB_COMPLETED",
            ref_id=job_id,
            payload=completed_dict,
        )
        return seq

    def get_undelivered_results(
        self,
        job_id: str,
        run_id: str,
        from_sequence: int = 1,
    ) -> List[Dict[str, Any]]:
        rows = self.db.get_results_from_sequence(job_id, run_id, from_sequence)
        out = []
        for r in rows:
            payload = json.loads(r["payload_json"])
            payload["_result_type"] = r["result_type"]
            payload["_sequence"] = r["sequence"]
            out.append(payload)
        return out

    def mark_acknowledged(
        self,
        job_id: str,
        run_id: str,
        last_acknowledged_sequence: int,
    ) -> None:
        self.db.update_delivery_state(job_id, run_id, last_acknowledged_sequence, "ACKNOWLEDGED")

    def get_run_snapshot(self, job_id: str, run_id: str) -> Optional[Dict[str, Any]]:
        job = self.db.get_job(job_id, run_id)
        if not job:
            return None
        return {
            "job": job,
            "images": self.db.get_all_images(job_id, run_id),
            "max_sequence": self.db.get_max_persisted_result_sequence(job_id, run_id),
        }
