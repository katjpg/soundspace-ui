import logging
import zipfile
from pathlib import Path

from soundspace.config.dataset import AudioSource, DatasetConfig
from soundspace.dataset.sources.zenodo import ZenodoClient, ZenodoConfig

log = logging.getLogger(__name__)


def download(cfg: DatasetConfig) -> Path:
    source = cfg.source
    raw_dir = cfg.raw_dir

    if source.extract:
        target = raw_dir / cfg.active
        if target.exists():
            log.info("%s already present", target.name)
            return target

    archive = _download_source(source, raw_dir)

    if source.extract and zipfile.is_zipfile(archive):
        dataset = _extract_zip(archive, raw_dir, rename_to=cfg.active)
        archive.unlink(missing_ok=True)
        log.info("removed archive %s", archive.name)
        return dataset

    return archive


def _download_source(source: AudioSource, raw_dir: Path) -> Path:
    if source.provider != "zenodo":
        raise ValueError(f"unsupported dataset source provider: {source.provider!r}")
    dest = raw_dir / source.filename
    config = ZenodoConfig(base_url=source.base_url)
    with ZenodoClient(config) as zenodo:
        return zenodo.download_record_file(
            record_id=source.record_id,
            filename=source.filename,
            dest=dest,
            expected_md5=source.expected_md5,
        )


def _extract_zip(
    archive: Path,
    dest_dir: Path,
    *,
    rename_to: str | None = None,
) -> Path:
    root = _zip_root(archive)
    target = dest_dir / (rename_to or root)

    if target.exists():
        log.info("%s already extracted", target.name)
        return target

    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as file:
        _check_zip_paths(file.infolist(), dest_dir)
        log.info("extracting %s -> %s", archive.name, dest_dir)
        file.extractall(dest_dir)

    extracted = dest_dir / root
    if extracted != target:
        extracted.rename(target)
        log.info("renamed %s -> %s", root, target.name)

    return target


def _zip_root(archive: Path) -> str:
    with zipfile.ZipFile(archive) as file:
        roots = {
            Path(name).parts[0]
            for name in file.namelist()
            if name and not name.startswith("__MACOSX")
        }
    if len(roots) != 1:
        raise ValueError(
            f"expected one top-level folder in {archive.name}, found {sorted(roots)}"
        )
    return roots.pop()


def _check_zip_paths(members: list[zipfile.ZipInfo], dest_dir: Path) -> None:
    root = dest_dir.resolve()
    for member in members:
        target = (root / member.filename).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"zip member escapes destination: {member.filename!r}")
