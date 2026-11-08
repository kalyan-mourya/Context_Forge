import json
import os
import warnings
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download


LECTURE_NOTES_REPO_ID = "hatakekksheeshh/CSE_course_RAG"
RESEARCH_DATASET_ID = "tegridydev/research-papers"
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[1] / "data" / "hf_datasets"
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def ensure_dataset_directory(root_dir: str | os.PathLike[str] = DEFAULT_DATASET_ROOT) -> Path:
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root


def download_lecture_notes_dataset(root_dir: str | os.PathLike[str] = DEFAULT_DATASET_ROOT) -> Path:
    target_dir = ensure_dataset_directory(root_dir) / "lecture_notes"
    target_dir.mkdir(parents=True, exist_ok=True)
    downloaded = snapshot_download(
        repo_id=LECTURE_NOTES_REPO_ID,
        repo_type="dataset",
        local_dir=str(target_dir),
        resume_download=True,
        token=HF_TOKEN,
    )
    return Path(downloaded)


def download_research_papers_dataset(root_dir: str | os.PathLike[str] = DEFAULT_DATASET_ROOT) -> Path:
    target_dir = ensure_dataset_directory(root_dir) / "research_papers"
    target_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(RESEARCH_DATASET_ID, token=HF_TOKEN)

    for split_name, split_data in ds.items():
        out_path = target_dir / f"{split_name}.jsonl"
        with out_path.open("w", encoding="utf-8") as handle:
            for row in split_data:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return target_dir


def ensure_datasets_available(root_dir: str | os.PathLike[str] = DEFAULT_DATASET_ROOT) -> Path:
    root = ensure_dataset_directory(root_dir)

    lecture_dir = root / "lecture_notes"
    research_dir = root / "research_papers"

    try:
        if not lecture_dir.exists() or not any(lecture_dir.iterdir()):
            download_lecture_notes_dataset(root)
    except Exception as exc:
        warnings.warn(f"Lecture notes dataset download skipped: {exc}")

    try:
        if not research_dir.exists() or not any(research_dir.iterdir()):
            download_research_papers_dataset(root)
    except Exception as exc:
        warnings.warn(f"Research dataset download skipped: {exc}")

    return root


if __name__ == "__main__":
    root = ensure_datasets_available()
    print(f"HF datasets available at: {root}")
