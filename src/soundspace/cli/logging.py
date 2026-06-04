import logging
import os
import warnings

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def configure_runtime(level: int = logging.INFO) -> None:
    _export_hf_token()
    logging.basicConfig(level=level, format="%(message)s")
    logging.captureWarnings(True)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\..*")

    for name in (
        "httpx",
        "huggingface_hub",
        "transformers",
        "transformers.modeling_utils",
        "urllib3",
        "filelock",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)

    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()


def _export_hf_token() -> None:
    if os.environ.get("HF_TOKEN"):
        return
    from soundspace.config.llm import HuggingFaceSettings

    token = HuggingFaceSettings().token
    if token is not None:
        os.environ["HF_TOKEN"] = token.get_secret_value()
