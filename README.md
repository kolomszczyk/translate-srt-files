# SRT Translator (PL)

Skrypt w `main.py` tłumaczy napisy `.srt` na język polski przez OpenAI API.

## Ważna informacja

Ten skrypt był tworzony przy wsparciu AI (OpenAI Codex/ChatGPT).

## Funkcje

- Tłumaczenie: 1 blok dialogu = 1 zapytanie do API.
- Kontekst: do zapytania dodawane są sąsiednie dialogi.
- Wznawianie: przerwany proces można uruchomić ponownie bez utraty postępu.
- Obsługa błędu limitu/kredytów API (`insufficient_quota`).

## Wymagania

- Python 3
- Pakiet `openai`
- Klucz OpenAI: `OPENAI_API_KEY` albo plik w `~/secrets/`

## Szybki start

```bash
python -m venv venv
./venv/bin/pip install openai
./venv/bin/python main.py napisy.srt napisy_pl.srt --model gpt-4.1-mini --context-window 2
```

## Wznawianie

Domyślnie skrypt wznawia tłumaczenie z istniejącego pliku wyjściowego.

Aby wymusić start od zera:

```bash
./venv/bin/python main.py napisy.srt napisy_pl.srt --no-resume
```
