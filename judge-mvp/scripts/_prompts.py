from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


@dataclass(frozen=True)
class PromptBundle:
    system: str
    user: str


def _read_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt file is empty: {path}")
    return text


def load_prompt_bundle(system_path: Path, user_path: Path) -> PromptBundle:
    return PromptBundle(system=_read_text(system_path), user=_read_text(user_path))


def load_named_prompt_bundle(name: str) -> PromptBundle:
    if name not in {"teacher", "main"}:
        raise ValueError(f"Unknown prompt bundle name: {name}")
    return load_prompt_bundle(
        PROMPTS_DIR / f"{name}_system.txt",
        PROMPTS_DIR / f"{name}_user.txt",
    )


def render_user_prompt(template: str, **values: str) -> str:
    try:
        rendered = template.format(**values)
    except KeyError as exc:
        missing = exc.args[0]
        raise ValueError(
            f"Missing placeholder value {{{missing}}} when rendering user prompt."
        ) from exc
    if not rendered.strip():
        raise ValueError("Rendered user prompt is empty.")
    return rendered
