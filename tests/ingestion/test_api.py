from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import SecretStr

from src.ingestion import api
from src.shared.config import Settings


def job_row(**overrides):
    row = {
        "job_id": overrides.pop("job_id"),
        "doc_id": "upload-test",
        "original_filename": "guide.pdf",
        "source_path": "/tmp/guide.pdf",
        "file_hash": "a" * 64,
        "status": "queued",
        "attempts": 0,
        "lease_expires_at": None,
        "total_chunks": 0,
        "embedded_chunks": 0,
        "skipped_chunks": 0,
        "error_type": None,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "completed_at": None,
    }
    row.update(overrides)
    return row


def test_upload_requires_key(monkeypatch, tmp_path):
    settings = Settings(
        _env_file=None, ingestion_api_key=SecretStr("admin"), ingestion_upload_dir=str(tmp_path)
    )
    app_instance = api.create_app(settings)
    with TestClient(app_instance) as client:
        response = client.post(
            "/v1/ingestions", files={"file": ("guide.pdf", b"%PDF-1.7", "application/pdf")}
        )
    assert response.status_code == 401


def test_upload_persists_pdf_and_creates_queued_job(monkeypatch, tmp_path):
    settings = Settings(
        _env_file=None, ingestion_api_key=SecretStr("admin"), ingestion_upload_dir=str(tmp_path)
    )
    monkeypatch.setattr(api, "get_connection", lambda settings: nullcontext(object()))
    monkeypatch.setattr(api, "get_job_by_hash", lambda conn, file_hash: None)

    def create(conn, **values):
        return job_row(
            job_id=values["job_id"], doc_id=f"upload-{values['job_id']}", source_path=values["source_path"]
        )

    monkeypatch.setattr(api, "create_job", create)
    with TestClient(api.create_app(settings)) as client:
        response = client.post(
            "/v1/ingestions",
            headers={"X-Ingestion-Key": "admin"},
            files={"file": ("guide.pdf", b"%PDF-1.7\nbody", "application/pdf")},
        )
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert Path(tmp_path, f"{body['job_id']}.pdf").read_bytes().startswith(b"%PDF-")


def test_upload_rejects_non_pdf_and_reuses_identical_upload(monkeypatch, tmp_path):
    settings = Settings(
        _env_file=None, ingestion_api_key=SecretStr("admin"), ingestion_upload_dir=str(tmp_path)
    )
    monkeypatch.setattr(api, "get_connection", lambda settings: nullcontext(object()))
    existing = job_row(job_id=__import__("uuid").uuid4(), status="completed")
    monkeypatch.setattr(api, "get_job_by_hash", lambda conn, file_hash: existing)
    with TestClient(api.create_app(settings)) as client:
        rejected = client.post(
            "/v1/ingestions",
            headers={"X-Ingestion-Key": "admin"},
            files={"file": ("guide.txt", b"not pdf", "text/plain")},
        )
        reused = client.post(
            "/v1/ingestions",
            headers={"X-Ingestion-Key": "admin"},
            files={"file": ("guide.pdf", b"%PDF-1.7", "application/pdf")},
        )
    assert rejected.status_code == 422
    assert reused.status_code == 200
    assert reused.json()["job_id"] == str(existing["job_id"])
