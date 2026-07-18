"""One-shot helper: remove unused type: ignore comments reported by mypy."""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    proc = subprocess.run(
        ["python", "-m", "mypy", "src/histoweave", "--no-pretty", "--no-error-summary"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    unused: dict[Path, set[int]] = {}
    for line in (proc.stdout + proc.stderr).splitlines():
        if "unused-ignore" not in line:
            continue
        match = re.match(r"(.+?):(\d+): error: Unused", line)
        if not match:
            continue
        path = Path(match.group(1))
        if not path.is_absolute():
            path = ROOT / path
        unused.setdefault(path, set()).add(int(match.group(2)))

    for path, lines in unused.items():
        text = path.read_text(encoding="utf-8").splitlines(keepends=True)
        for ln in sorted(lines, reverse=True):
            i = ln - 1
            if 0 <= i < len(text):
                text[i] = re.sub(
                    r"[ \t]*#\s*type:\s*ignore(?:\[[^\]]*\])?[ \t]*$",
                    "",
                    text[i].rstrip("\n\r"),
                ) + ("\n" if text[i].endswith("\n") else "")
        path.write_text("".join(text), encoding="utf-8")
        logger.info("cleaned %s: %s", path.relative_to(ROOT), sorted(lines))
    logger.info("done %s files", len(unused))


if __name__ == "__main__":
    main()
