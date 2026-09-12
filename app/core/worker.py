"""A single background worker thread draining an in-process queue.

Why a worker thread rather than FastAPI's ``BackgroundTasks``
------------------------------------------------------------
``BackgroundTasks`` runs after the response is sent, but still inside the server's
request machinery: a synchronous task occupies a thread from the same limited pool
Starlette uses to run every ``def`` endpoint. Embedding is CPU-bound and takes
seconds per document, so a handful of concurrent uploads would starve that pool and
stall unrelated requests. It also gives the job no identity -- nothing to poll, no
state to report, and a crash disappears into the response cycle that already ended.

An explicit queue plus one daemon thread fixes both: uploads return ``202`` in
milliseconds, the job's state lives in SQLite where ``GET /documents/{id}`` can read
it, and the depth of the queue is observable. One worker is deliberate -- the
embedding model is the bottleneck and a second thread would contend for the same
CPU and the same model instance without improving throughput.

The scale-up path is Celery with a Redis or RabbitMQ broker, which moves the same
contract into separate processes so work survives a restart and can be spread over
machines. That is a deployment change, not a redesign: only ``enqueue_ingestion``
would be rewritten.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path

from app.core.ingest import ingest_document

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionJob:
    document_id: str
    file_path: str
    filename: str


_SHUTDOWN = object()

_queue: queue.Queue = queue.Queue()
_worker: threading.Thread | None = None
_lock = threading.Lock()


def enqueue_ingestion(document_id: str, file_path: str | Path, filename: str) -> None:
    """Hand a document to the worker and return immediately."""
    _queue.put(IngestionJob(document_id, str(file_path), filename))
    logger.info("queued %s (%s); depth now %d", document_id, filename, _queue.qsize())


def queue_depth() -> int:
    return _queue.qsize()


def start_worker() -> None:
    """Start the worker thread. Idempotent, so repeated startup is harmless."""
    global _worker
    with _lock:
        if _worker is not None and _worker.is_alive():
            return
        _worker = threading.Thread(target=_run, name="ingestion-worker", daemon=True)
        _worker.start()
        logger.info("ingestion worker started")


def stop_worker(timeout: float = 10.0) -> None:
    """Ask the worker to finish the current job and exit."""
    global _worker
    with _lock:
        worker, _worker = _worker, None
    if worker is None or not worker.is_alive():
        return
    _queue.put(_SHUTDOWN)
    worker.join(timeout=timeout)
    logger.info("ingestion worker stopped (alive=%s)", worker.is_alive())


def _run() -> None:
    while True:
        job = _queue.get()
        try:
            if job is _SHUTDOWN:
                return
            try:
                ingest_document(job.document_id, job.file_path, filename=job.filename)
            except Exception:
                # ingest_document has already recorded the failure in SQLite; the
                # worker must survive so later jobs still run.
                logger.exception("job %s failed", job.document_id)
        finally:
            _queue.task_done()


def drain(timeout: float | None = None) -> None:
    """Block until the queue is empty. Test and CLI helper, not used by the API."""
    _queue.join()
