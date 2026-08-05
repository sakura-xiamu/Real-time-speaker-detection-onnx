#!/usr/bin/env python3
"""从 sherpa-onnx GitHub Release 下载说话人分离 / 声纹模型。"""
from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server import config  # noqa: E402

MODELS_DIR = config.MODELS_DIR
SEG_URL = (
    f"{config.SPEAKER_SEGMENTATION_RELEASE}/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
EMB_BASE = config.SPEAKER_RECOGNITION_RELEASE


def download(url: str, dest: Path, force: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"[skip] {dest.name}")
        return
    print(f"[download] {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)
        if dest.exists():
            dest.unlink()
        tmp.rename(dest)
        print(f"[saved] {dest}")
    except Exception as exc:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        print(f"[fail] {dest.name}: {exc}", file=sys.stderr)
        raise


def extract_tar_bz2(archive: Path, target_dir: Path, force: bool = False) -> None:
    if target_dir.exists() and (target_dir / "model.onnx").is_file() and not force:
        print(f"[skip extract] {target_dir.name}")
        return
    print(f"[extract] {archive.name} -> {target_dir}")
    with tarfile.open(archive, "r:bz2") as tar:
        try:
            tar.extractall(MODELS_DIR, filter="data")
        except TypeError:
            tar.extractall(MODELS_DIR)
    archive.unlink(missing_ok=True)


def download_segmentation(force: bool = False) -> None:
    archive = MODELS_DIR / "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
    target = config.SEGMENTATION_DIR
    download(SEG_URL, archive, force=force)
    extract_tar_bz2(archive, target, force=force)


def download_embedding(key: str | None = None, all_models: bool = False, force: bool = False) -> None:
    if all_models:
        keys = list(config.EMBEDDING_MODELS.keys())
    elif key:
        if key not in config.EMBEDDING_MODELS:
            raise SystemExit(f"未知模型 key: {key}")
        keys = [key]
    else:
        keys = [config.DEFAULT_EMBEDDING_MODEL_KEY]

    for k in keys:
        info = config.EMBEDDING_MODELS[k]
        if not info.get("downloadable", True):
            print(f"[skip undownloadable] {k}")
            continue
        filename = info["filename"]
        url = f"{EMB_BASE}/{filename}"
        dest = MODELS_DIR / filename
        try:
            download(url, dest, force=force)
        except Exception:
            print(
                f"[warn] 无法下载 {k} ({filename})，可稍后重试或手动放置到 models/",
                file=sys.stderr,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="下载说话人分离模型")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="下载指定 embedding key（默认仅默认模型）",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="下载注册表中全部可下载 embedding 模型（体积较大）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出注册表模型",
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.list:
        for item in config.list_embedding_models():
            flag = "OK" if item["available"] else "--"
            print(f"  [{flag}] {item['key']}: {item['display_name']}")
        seg_ok = config.SEGMENTATION_MODEL.is_file()
        print(f"  [{'OK' if seg_ok else '--'}] segmentation: {config.SEGMENTATION_MODEL}")
        return

    print("== 下载 segmentation ==")
    download_segmentation(force=args.force)

    print("== 下载 embedding ==")
    download_embedding(key=args.model, all_models=args.all, force=args.force)

    print("\nReady check:")
    ok = True
    if not config.SEGMENTATION_MODEL.is_file():
        print(f"  [MISS] {config.SEGMENTATION_MODEL}", file=sys.stderr)
        ok = False
    else:
        print(f"  [OK] {config.SEGMENTATION_MODEL}")

    default_path = config.embedding_model_path(config.DEFAULT_EMBEDDING_MODEL_KEY)
    if not default_path.is_file():
        print(f"  [MISS] default model {default_path}", file=sys.stderr)
        ok = False
    else:
        print(f"  [OK] {default_path}")

    if not ok:
        sys.exit(1)
    print("\nModels ready.")


if __name__ == "__main__":
    main()
