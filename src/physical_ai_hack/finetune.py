"""Fine-tune Dream-1 on a dataset with an embodiment config of our own (the kit's tools/finetune.py fixes its own).

    uv run --group convert finetune <dataset> --embodiment configs/embodiments/X.json --check     # local, free
    uv run --group convert finetune <dataset> --embodiment ... --name N --suffix S [--steps 1000] # PAID: queues a job
    uv run --group convert finetune --job ftj_...                                               # follow a job

Same steps as the kit: register the embodiment (reused when the same config is already registered under the name),
check the dataset, upload, wait for processing, queue the job, wait for the model. The account is the one in the
Dream credentials (DREAM_CREDENTIALS picks the file). A queued job cannot be cancelled and is charged by its steps.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time


def _embodiment(client, name: str, config: dict) -> str:
    import dream

    try:
        return client.add_embodiment(name, config).name
    except dream.DreamServerError as error:
        if error.code != "conflict":
            raise
    existing = next(e for e in client.get_embodiments() if e.name.endswith(f"/{name}"))
    if existing.embodiment_config != config:
        sys.exit(f"{existing.name} is registered with another config; pass --embodiment-name with a new name")
    return existing.name


def _wait_model(client, job_id: str) -> None:
    job = client.get_finetuning_status(job_id)
    while job["fine_tuned_model"] is None and job["status"] in ("queued", "running", "succeeded"):
        progress = job.get("progress") or {}
        detail = job.get("queue_position") if job["status"] == "queued" else progress.get("phase")
        print(f"job {job['id']}: {job['status']} {detail or ''} step {progress.get('step')}/"
              f"{progress.get('total_steps')} loss {progress.get('loss')}", flush=True)
        time.sleep(60)
        job = client.get_finetuning_status(job_id)
    if not job["fine_tuned_model"]:
        sys.exit(f"job {job['id']} ended {job['status']} without a model: {job.get('error')}")
    print(f"\nmodel: {job['fine_tuned_model']}\n"
          f"try it: dreamcli robot run --checkpoint {job['fine_tuned_model']} --task \"...\" --dry-run")


def main(argv=None) -> None:
    import dream

    p = argparse.ArgumentParser(prog="finetune", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", nargs="?", type=pathlib.Path)
    p.add_argument("--embodiment", type=pathlib.Path, help="embodiment config JSON (configs/embodiments/...)")
    p.add_argument("--embodiment-name", help="name to register it under (default: the JSON file's stem)")
    p.add_argument("--name", help="dataset name on the platform: lowercase letters, digits, hyphens; new each time")
    p.add_argument("--suffix", help="end of the model id: lowercase letters, digits, hyphens; new each time")
    p.add_argument("--steps", type=int, help="500-2000 in hundreds (default: the service's 1000); price grows with it")
    p.add_argument("--lr-schedule", choices=("constant", "cosine"),
                   help="cosine suits one task; constant if you may continue the model later (resume_from)")
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--base-share", type=float, choices=(0.0, 0.5), help="share of Dream's own data (default 0.5)")
    p.add_argument("--resume-from", help="a dream-1-ft:... model of yours to keep training on this dataset")
    p.add_argument("--no-eef", action="store_true", help="train joints only (default: end-effector, as the kit)")
    p.add_argument("--check", action="store_true", help="only run Dream's dataset checks, locally; send nothing")
    p.add_argument("--dataset-id", help="resume after the upload (ds_...)")
    p.add_argument("--job", help="only follow a queued job (ftj_...)")
    args = p.parse_args(argv)

    if args.job:
        _wait_model(dream.DreamClient(), args.job)
        return
    if args.embodiment is None:
        p.error("--embodiment is required")
    config = json.loads(args.embodiment.read_text())
    if args.dataset_id is None:
        if args.dataset is None:
            p.error("give the dataset folder")
        report = dream.datasets.validate_dataset(args.dataset.expanduser(), config)
        if not report.ok:
            sys.exit(f"Dream's checks refuse the dataset: {report.error}")
        print(f"checks pass: {report.episodes} episodes, {report.frames} frames at {report.fps} fps, "
              f"cameras {report.video_keys}")
        if args.check:
            return
    if args.suffix is None or (args.dataset_id is None and args.name is None):
        p.error("--name and --suffix are needed to upload and queue a job")
    client = dream.DreamClient()
    dataset_id = args.dataset_id
    if dataset_id is None:
        embodiment = _embodiment(client, args.embodiment_name or args.embodiment.stem.replace("_", "-"), config)
        dataset = client.upload_dataset(str(args.dataset.expanduser()), args.name, embodiment)
        dataset_id = dataset.id
        print(f"uploaded {dataset.name} as {dataset.id} (resume with --dataset-id {dataset.id})")
    dataset = client.get_dataset_status(dataset_id)
    while dataset.status not in ("processed", "failed"):
        print(f"dataset {dataset.id}: {dataset.status}", flush=True)
        time.sleep(30)
        dataset = client.get_dataset_status(dataset_id)
    if dataset.status == "failed":
        sys.exit(f"dataset {dataset.id} failed processing: {dataset.error}")
    job = client.finetune(dataset.name, args.suffix, steps=args.steps, learning_rate=args.learning_rate,
                          lr_schedule=args.lr_schedule, use_eef=not args.no_eef,
                          resume_from=args.resume_from, base_share=args.base_share)
    print(f"queued {job['id']} (follow with: finetune --job {job['id']})")
    _wait_model(client, job["id"])


def run() -> None:
    main()
