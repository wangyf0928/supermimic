"""W&B model-artifact helpers: digest-cached download and versioned logging."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import wandb

_ARTIFACT_CACHE_DIR = Path.home() / ".cache" / "supermimic" / "artifacts"


def download_artifact_cached(artifact: Any) -> Path:
  """Download a wandb artifact with digest-based caching.

  Skips ``artifact.download()`` entirely when the local cache already holds
  files for the same content digest. Re-downloads automatically when the
  artifact content changes (different digest).

  Returns the path to the downloaded artifact directory.
  """
  # entity/project/name contains "/"; hash it to a flat cache filename.
  raw_key = f"{artifact.entity}/{artifact.project}/{artifact.name}"
  key = hashlib.sha256(raw_key.encode()).hexdigest()[:16]
  _ARTIFACT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
  meta_file = _ARTIFACT_CACHE_DIR / f"{key}.json"

  try:
    meta = json.loads(meta_file.read_text())
    if meta["digest"] == artifact.digest and Path(meta["path"]).exists():
      print(
        f"[INFO] Using cached artifact: {artifact.name} "
        f"(digest: {artifact.digest[:8]}...)"
      )
      return Path(meta["path"])
  except (OSError, json.JSONDecodeError, KeyError, TypeError):
    pass

  print(
    f"[INFO] Downloading artifact: {artifact.name} (digest: {artifact.digest[:8]}...)"
  )
  path = artifact.download()
  meta_file.write_text(json.dumps({"digest": artifact.digest, "path": path}))
  return Path(path)


def download_model_dir(run: Any) -> Path:
  """Local dir of a run's ``:latest`` model-checkpoint artifact (digest-cached)."""
  artifact = next(
    (a for a in run.logged_artifacts() if a.type == "model" and "latest" in a.aliases),
    None,
  )
  if artifact is None:
    raise RuntimeError(f"No 'model' artifact found for run {run.id}")
  return download_artifact_cached(artifact)


def _version_index(artifact: Any) -> int:
  return int(str(artifact.version).lstrip("v"))


def _publish_model_artifact(run: wandb.Run, artifact: wandb.Artifact) -> None:
  try:
    logged = run.log_artifact(artifact)
    if run.offline:
      return  # The local W&B run can be synced later.
    logged.wait()
    collection = f"{run.entity}/{run.project}/model-{run.id}"
    versions = sorted(wandb.Api().artifacts("model", collection), key=_version_index)
    for stale in versions[:-5]:
      if _version_index(stale) % 25 != 0 and not stale.aliases:
        stale.delete()
  except Exception as e:  # wandb failures must never kill training
    print(f"[WARN] Checkpoint artifact log/prune failed: {type(e).__name__}: {e}")


class CheckpointUploader:
  """Publish one snapshot at a time and flush the last local save on exit."""

  def __init__(self, run: wandb.Run) -> None:
    self._run = run
    self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="checkpoint")
    self._upload: Future[None] | None = None
    self._pending: tuple[list[str], int] | None = None

  def submit(self, files: list[str], step: int) -> None:
    if self._run.disabled:
      return
    # Only the most recent save is needed once the training loop stops writing.
    self._pending = (files, step)
    if self._upload is not None and not self._upload.done():
      return
    try:
      artifact = wandb.Artifact(f"model-{self._run.id}", type="model")
      for path in files:
        # Stage on this thread before the next local save can overwrite the files.
        artifact.add_file(path, policy="mutable")
      self._upload = self._pool.submit(_publish_model_artifact, self._run, artifact)
      self._pending = None
    except Exception as e:
      print(f"[WARN] Could not stage checkpoint at step {step} for W&B: {e}")

  def finish(self) -> None:
    if self._pending is not None:
      if self._upload is not None:
        self._upload.result()
      self.submit(*self._pending)
    self._pool.shutdown()
