"""Slurm job submission utilities for airgapped evaluation.

Provides helpers for downloading models and datasets on the login node,
submitting Slurm jobs for evaluation, waiting for completion, and
collecting results from the shared filesystem.
"""

from __future__ import annotations

import collections.abc as c
import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

from leaderboards.evaluation_common import run_euroeval

logger = logging.getLogger(__name__)


def download_for_airgapped_eval(
    model_id: str,
    languages: c.Sequence[str],
    cache_dir: Path,
    datasets: c.Sequence[str] | None = None,
    evaluate_test_split: bool = True,
    zero_shot: bool = False,
    gpu_memory_utilization: float | None = None,
) -> tuple[int, str]:
    """Download models and datasets for airgapped evaluation.

    Runs ``euroeval --download-only`` on the login node to pre-download
    models and datasets into a shared cache directory. Delegates to
    ``run_euroeval()`` with the ``download_only`` flag.

    Args:
        model_id:
            The model identifier to download.
        languages:
            ISO codes to pass via repeated ``--language`` flags.
        cache_dir:
            Directory to store downloaded models and datasets.
        datasets (optional):
            Dataset ids to pass via repeated ``--dataset`` flags. When
            None or empty, no ``--dataset`` flag is passed and the CLI
            uses its language-driven default. Defaults to None.
        evaluate_test_split (optional):
            When True pass ``--evaluate-test-split``; when False pass
            ``--evaluate-val-split``. Defaults to True.
        zero_shot (optional):
            When True pass ``--zero-shot``; otherwise omit (CLI default
            is few-shot). Defaults to False.
        gpu_memory_utilization (optional):
            When set, pass ``--gpu-memory-utilization VALUE``. When None,
            omit the flag so the euroeval CLI's default applies. Defaults
            to None.

    Returns:
        A ``(returncode, output)`` tuple. A returncode of 127 signals
        that the CLI was not found on PATH.
    """
    return run_euroeval(
        model_id=model_id,
        languages=languages,
        datasets=datasets,
        evaluate_test_split=evaluate_test_split,
        zero_shot=zero_shot,
        trust_remote_code=True,
        clear_model_cache=False,  # Don't clear cache for download
        gpu_memory_utilization=gpu_memory_utilization,
        download_only=True,
        cache_dir=cache_dir,
    )


def submit_slurm_eval_job(
    model_id: str,
    languages: list[str],
    cache_dir: Path,
    results_path: Path,
    gpu_memory_utilization: float | None = None,
) -> tuple[str, Path]:
    """Submit a Slurm job for running EuroEval evaluation.

    Creates a temporary Slurm script that runs the evaluation with the
    pre-populated cache directory and submits it via ``sbatch``.

    Args:
        model_id:
            The model identifier to evaluate.
        languages:
            ISO codes to pass via repeated ``--language`` flags.
        cache_dir:
            Path to the pre-populated model/dataset cache.
        results_path:
            Path to the shared results JSONL file. The job will write to
            a job-specific file adjacent to this path.
        gpu_memory_utilization (optional):
            GPU memory utilisation fraction (0.0–1.0). When None, the
            euroeval CLI default applies. Defaults to None.

    Returns:
        A ``(job_id, job_results_path)`` tuple. The job results path points
        to a job-specific JSONL file that the compute node will write to.

    Raises:
        RuntimeError:
            If ``sbatch`` fails to submit the job.
    """
    # Build the euroeval command
    cmd_parts: list[str] = [
        "euroeval",
        "--model",
        model_id,
        "--cache-dir",
        str(cache_dir),
        # Note: NOT using --clear-model-cache since we pre-downloaded
        "--trust-remote-code",
        "--evaluate-test-split",
    ]
    for lang in languages:
        cmd_parts += ["--language", lang]
    if gpu_memory_utilization is not None:
        cmd_parts += ["--gpu-memory-utilization", str(gpu_memory_utilization)]

    # Use a job-specific results file to avoid race conditions
    # The file will be alongside the main results file
    job_id_for_filename = "$(SLURM_JOB_ID)"
    job_results_filename = f".euroeval_job_{job_id_for_filename}.jsonl"
    job_results_path = results_path.parent / job_results_filename
    cmd_parts += ["--output-file", str(job_results_path)]

    euroeval_cmd = " ".join(cmd_parts)

    # Create the Slurm script
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=euroeval_{model_id.replace("/", "_")}
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# Set up environment variables
export HF_TOKEN="${{HF_TOKEN:-}}"
export HUGGINGFACE_API_KEY="${{HUGGINGFACE_API_KEY:-}}"
export FULL_LOG=1

# Run the evaluation
{euroeval_cmd}
"""

    # Write to a temporary file and submit
    with tempfile.NamedTemporaryFile(mode="w", suffix=".slurm", delete=False) as f:
        f.write(slurm_script)
        script_path = f.name

    try:
        logger.info(f"Submitting Slurm job: {euroeval_cmd}")
        result = subprocess.run(  # noqa: S603
            ["sbatch", script_path], capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            raise RuntimeError(f"sbatch failed: {result.stderr}")

        # Parse job ID from sbatch output (e.g. "Submitted batch job 12345")
        job_id: str | None = None
        for line in result.stdout.strip().split("\n"):
            parts = line.split()
            if "job" in parts:
                job_idx = parts.index("job")
                if job_idx + 1 < len(parts):
                    job_id = parts[job_idx + 1]
                    break

        if job_id is None:
            raise RuntimeError(
                f"Could not parse job ID from sbatch output: {result.stdout}"
            )

        # Return the actual path with the real job ID
        actual_job_results_path = results_path.parent / f".euroeval_job_{job_id}.jsonl"
        return job_id, actual_job_results_path
    finally:
        # Clean up the temporary script
        os.unlink(script_path)


def wait_for_slurm_job(job_id: str, poll_interval: float = 10.0) -> int:
    """Wait for a Slurm job to complete and return its exit code.

    Polls ``squeue -j <job_id>`` until the job exits, then uses ``sacct``
    to retrieve the exit code.

    Args:
        job_id:
            The Slurm job ID to wait for.
        poll_interval (optional):
            Time in seconds between polling attempts. Defaults to 10.0.

    Returns:
        The Slurm exit code (0 for success, non-zero for failure).
    """
    logger.info(f"Waiting for Slurm job {job_id} to complete")

    while True:
        result = subprocess.run(  # noqa: S603
            ["squeue", "-j", job_id, "--noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            # Job no longer in queue - it has finished
            break

        time.sleep(poll_interval)

    # Get the exit code from sacct
    logger.info(f"Job {job_id} completed, fetching exit code")
    sacct_result = subprocess.run(  # noqa: S603
        ["sacct", "-j", job_id, "--format=State,ExitCode", "--noheader", "-n"],
        capture_output=True,
        text=True,
        check=False,
    )

    if sacct_result.returncode != 0:
        logger.warning(f"sacct failed for job {job_id}: {sacct_result.stderr}")
        return -1

    # Parse exit code from sacct output
    # Format: "COMPLETED 0:0" or "FAILED 1:0" etc.
    for line in sacct_result.stdout.strip().split("\n"):
        parts = line.split()
        if len(parts) >= 2:
            # ExitCode format is "exit_code:signal"
            exit_code_str = parts[-1].split(":")[0]
            try:
                return int(exit_code_str)
            except ValueError:
                continue

    logger.warning(
        f"Could not parse exit code from sacct output: {sacct_result.stdout}"
    )
    return -1


def collect_slurm_results(results_path: Path, before_lines: set[str]) -> list[str]:
    """Collect new result lines from the Slurm job output.

    Reads the results JSONL file and returns only the lines that were
    not present before the job ran (i.e. not in ``before_lines``).

    Args:
        results_path:
            Path to the results JSONL file.
        before_lines:
            Set of lines that were present before the job ran.

    Returns:
        List of new result lines added by the job.
    """
    if not results_path.exists():
        logger.warning(f"Results file not found: {results_path}")
        return []

    new_lines: list[str] = []
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.rstrip("\n\r")
                if stripped and stripped not in before_lines:
                    new_lines.append(stripped)
    except (OSError, json.JSONDecodeError) as e:
        logger.error(f"Error reading results file {results_path}: {e}")
        return []

    logger.info(f"Collected {len(new_lines)} new result lines from {results_path}")
    return new_lines
