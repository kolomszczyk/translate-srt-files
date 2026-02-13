#!/usr/bin/env python3
"""
Translate an SRT file to Polish using OpenAI Chat Completions.

One subtitle block is translated per API request, while providing
surrounding subtitle context in the same request.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

from openai import APIStatusError, OpenAI


TIMING_RE = re.compile(
    r"^\d{2}:\d{2}:\d{2},\d{3}\s-->\s\d{2}:\d{2}:\d{2},\d{3}"
    r"(?:\s+.*)?$"
)


@dataclass
class SubtitleBlock:
    index: int
    timing: str
    lines: List[str]


def parse_srt(path: Path) -> List[SubtitleBlock]:
    raw = path.read_text(encoding="utf-8-sig")
    chunks = re.split(r"\r?\n\r?\n+", raw.strip())
    blocks: List[SubtitleBlock] = []

    for chunk in chunks:
        lines = chunk.splitlines()
        if len(lines) < 3:
            continue

        try:
            idx = int(lines[0].strip())
        except ValueError:
            continue

        timing = lines[1].strip()
        if not TIMING_RE.match(timing):
            continue

        text_lines = lines[2:]
        blocks.append(SubtitleBlock(index=idx, timing=timing, lines=text_lines))

    if not blocks:
        raise ValueError("Nie udało się sparsować żadnych bloków z pliku SRT.")
    return blocks


def format_block(block: SubtitleBlock) -> str:
    text = "\n".join(block.lines).strip()
    return f"[{block.index} | {block.timing}] {text}"


def build_context(blocks: List[SubtitleBlock], i: int, window: int) -> str:
    left = max(0, i - window)
    right = min(len(blocks), i + window + 1)
    context_items = []
    for j in range(left, right):
        if j == i:
            continue
        context_items.append(format_block(blocks[j]))
    return "\n".join(context_items)


def translate_block(
    client: OpenAI,
    model: str,
    block: SubtitleBlock,
    context_text: str,
    retries: int,
    backoff_seconds: float,
) -> List[str]:
    source_text = "\n".join(block.lines).strip()

    system_prompt = (
        "Jesteś profesjonalnym tłumaczem napisów filmowych.\n"
        "Tłumaczysz tekst na język polski.\n"
        "Zachowujesz sens, ton i naturalność wypowiedzi.\n"
        "Nie dodawaj komentarzy ani wyjaśnień.\n"
        "Wynik zwróć WYŁĄCZNIE jako sam przetłumaczony tekst."
    )

    user_prompt = (
        f"Kontekst sceny (sąsiednie dialogi):\n"
        f"{context_text if context_text else '(brak kontekstu)'}\n\n"
        f"Dialog do tłumaczenia:\n{source_text}\n\n"
        "Wymagania:\n"
        "- Zachowaj podziały linii możliwie blisko oryginału.\n"
        "- Nie tłumacz znaczników technicznych napisów.\n"
        "- Zwróć tylko tłumaczenie dialogu."
    )

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            translated = (response.choices[0].message.content or "").strip()
            if not translated:
                raise ValueError("Pusta odpowiedź modelu.")
            return translated.splitlines()
        except Exception as exc:  # noqa: BLE001
            if is_credit_exhausted_error(exc):
                raise RuntimeError(
                    "Wykryto brak kredytow/srodkow w OpenAI API "
                    "(insufficient_quota / billing limit)."
                ) from exc
            last_error = exc
            if attempt < retries:
                sleep_for = backoff_seconds * (2**attempt)
                time.sleep(sleep_for)
            else:
                break

    raise RuntimeError(f"Błąd tłumaczenia bloku {block.index}: {last_error}") from last_error


def write_srt(path: Path, blocks: List[SubtitleBlock]) -> None:
    out_chunks = []
    for i, block in enumerate(blocks, start=1):
        out_chunks.append(
            f"{i}\n{block.timing}\n" + "\n".join(block.lines).strip() + "\n"
        )
    payload = "\n".join(out_chunks).rstrip() + "\n"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    tmp_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tłumaczy plik SRT na polski przez OpenAI API "
            "(1 blok napisów = 1 zapytanie)."
        )
    )
    parser.add_argument("input_srt", type=Path, help="Ścieżka do wejściowego pliku .srt")
    parser.add_argument("output_srt", type=Path, help="Ścieżka do wynikowego pliku .srt")
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="Model OpenAI (domyślnie: gpt-4.1-mini)",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=2,
        help="Liczba bloków kontekstu przed i po dialogu (domyślnie: 2)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Liczba ponowień przy błędach API (domyślnie: 3)",
    )
    parser.add_argument(
        "--backoff",
        type=float,
        default=1.0,
        help="Początkowy backoff w sekundach (domyślnie: 1.0)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Wyłącz wznawianie i tłumacz od początku.",
    )
    return parser.parse_args()


def load_api_key() -> str | None:
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key.strip()

    def parse_key_file(content: str) -> str | None:
        raw = content.strip()
        if not raw:
            return None
        if "=" not in raw:
            return raw

        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            name = name.strip()
            if name.startswith("export "):
                name = name[len("export ") :].strip()
            if name not in {"OPENAI_API_KEY", "OPENAI_KEY"}:
                continue
            value = value.strip().strip("\"'")
            if value:
                return value
        return None

    secrets_dir = Path.home() / "secrets"
    candidates = [
        secrets_dir / "open_ai_api",
        secrets_dir / "openai_api",
        secrets_dir / "openai_api_key",
        secrets_dir / "chat_gpt_api_key",
    ]
    for path in candidates:
        if path.is_file():
            key = parse_key_file(path.read_text(encoding="utf-8"))
            if key:
                return key
    return None


def load_resume_blocks(
    output_srt: Path, input_blocks: List[SubtitleBlock], resume_enabled: bool
) -> List[SubtitleBlock]:
    if not resume_enabled or not output_srt.exists():
        return []

    existing = parse_srt(output_srt)
    if len(existing) > len(input_blocks):
        raise ValueError("Plik wyjściowy ma więcej bloków niż wejściowy.")

    for i, block in enumerate(existing):
        src = input_blocks[i]
        if block.timing != src.timing:
            raise ValueError(
                f"Niezgodny timing przy wznowieniu w bloku {i + 1}: "
                f"{block.timing} != {src.timing}"
            )

    return existing


def is_credit_exhausted_error(exc: Exception) -> bool:
    if isinstance(exc, APIStatusError):
        # OpenAI typically reports exhausted credits as 429 with
        # error code like "insufficient_quota" or billing limit notes.
        if exc.status_code == 429:
            body = str(getattr(exc, "response", "")) + " " + str(exc)
            text = body.lower()
            markers = (
                "insufficient_quota",
                "billing_hard_limit_reached",
                "exceeded your current quota",
                "quota",
                "billing",
            )
            return any(marker in text for marker in markers)
    else:
        text = str(exc).lower()
        markers = (
            "insufficient_quota",
            "billing_hard_limit_reached",
            "exceeded your current quota",
        )
        return any(marker in text for marker in markers)
    return False


def main() -> int:
    args = parse_args()
    api_key = load_api_key()
    if not api_key:
        print(
            "Brak klucza OpenAI. Ustaw OPENAI_API_KEY lub dodaj plik z kluczem "
            "w ~/secrets (np. open_ai_api / openai_api / chat_gpt_api_key).",
            file=sys.stderr,
        )
        return 1

    blocks = parse_srt(args.input_srt)
    client = OpenAI(api_key=api_key)

    total = len(blocks)
    try:
        translated_blocks = load_resume_blocks(
            output_srt=args.output_srt,
            input_blocks=blocks,
            resume_enabled=not args.no_resume,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Błąd wznawiania: {exc}", file=sys.stderr)
        return 1

    done = len(translated_blocks)
    if done > 0:
        print(f"Wznowienie: znaleziono {done}/{total} przetłumaczonych bloków.")
    else:
        print(f"Start tłumaczenia: {total} bloków...")

    for i in range(done, total):
        block = blocks[i]
        context_text = build_context(blocks, i, args.context_window)
        translated_lines = translate_block(
            client=client,
            model=args.model,
            block=block,
            context_text=context_text,
            retries=args.retries,
            backoff_seconds=args.backoff,
        )
        translated_blocks.append(
            SubtitleBlock(index=block.index, timing=block.timing, lines=translated_lines)
        )
        # Persist progress after every translated block, so interruption is safe.
        write_srt(args.output_srt, translated_blocks)
        print(f"[{i + 1}/{total}] przetłumaczono blok {block.index}")

    print(f"Zapisano plik: {args.output_srt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
