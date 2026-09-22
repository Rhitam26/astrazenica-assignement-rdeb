"""Polling worker for durable PDF ingestion jobs."""

import argparse
import logging
from time import sleep

from src.ingestion.jobs import claim_next_job, fail_job
from src.ingestion.processor import process_uploaded_job
from src.shared.config import get_settings
from src.shared.database import get_connection
from src.shared.embedder import Embedder
from src.shared.observability import Telemetry, value_hash

logger = logging.getLogger(__name__)


def run_once() -> bool:
    settings = get_settings()
    with get_connection(settings) as conn:
        job = claim_next_job(
            conn,
            lease_seconds=settings.ingestion_job_lease_seconds,
            max_attempts=settings.ingestion_job_max_attempts,
        )
    if job is None:
        return False
    telemetry = Telemetry(settings)
    embedder = Embedder(settings, telemetry)
    try:
        with telemetry.trace(
            "ingestion-job",
            session_id=str(job["job_id"]),
            metadata={"job_id": str(job["job_id"]), "file_hash": value_hash(job["file_hash"])},
        ):
            process_uploaded_job(settings, job, embedder)
    except Exception as exc:
        logger.exception(
            "ingestion_job_failed", extra={"job_id": str(job["job_id"]), "error_type": type(exc).__name__}
        )
        with get_connection(settings) as conn:
            fail_job(conn, job["job_id"], type(exc).__name__)
    finally:
        embedder.close()
        telemetry.flush()
        telemetry.shutdown()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the asynchronous PDF ingestion worker")
    parser.add_argument("--once", action="store_true", help="Claim and process at most one job")
    args = parser.parse_args()
    settings = get_settings()
    while True:
        processed = run_once()
        if args.once:
            return
        if not processed:
            sleep(settings.ingestion_worker_poll_seconds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
