# VOICE-002: wycinanie innych głosów (badanie)

Plan i kryteria: repo `voice`, plik
`implementation-plans/voice-002-personalized-voice-isolation-research.md`.

## Układ

- `texts/`: teksty do nagrań użytkownika (R1).
- `data/` (poza gitem): `public/` (LibriSpeech, szum), `user/` (nagrania użytkownika,
  tylko lokalnie), `mix/` (mieszanki R1).
- `results/` (poza gitem): wyjścia modeli i tabele metryk.

## Uruchamianie

```sh
uv sync
uv run python -c "import torch; print(torch.backends.mps.is_available())"
uv run pytest -q
```

Common Voice PL: klucz API w `.env` (`MDC_API_KEY=...`, poza gitem), potem
`scripts/download_common_voice.sh`.

Mieszanki (R1) i pomiar bazowy DFN3 (R2):

```sh
uv run python -m voice002.make_mixtures --target librispeech --count 5 --name ls-v1
uv run python -m voice002.run_baseline ls-v1          # results/ls-v1/{none,dfn3-a100}
uv run python -m voice002.evaluate data/mix/ls-v1 results/ls-v1/<system>
```

R4a, bramka głosu z odciskiem CAM++. Model ONNX (Apache 2.0) do `data/models/`:

```sh
gh release download speaker-recongition-models -R k2-fsa/sherpa-onnx \
  -p 3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx -D data/models
uv run python -m voice002.run_gate ls-dev --grid                     # strojenie
uv run python -m voice002.run_gate ls-v2 --window 1 --th-lo 0.35 --th-hi 0.45 --floor -20
```

Nagrania użytkownika z aplikacji Voice (`mic-raw.wav`, bez filtra):

```sh
uv run python -m voice002.import_user --list
uv run python -m voice002.import_user <rec-id> B      # A = próbka 10 s, B/C/D = mowa
uv run python -m voice002.make_mixtures --target user --count 10 --name user-v1
```

DFN3 do pomiaru bazowego (R2): `target/release/deep-filter` w katalogu głównym forka
(`cargo build --release -p deep_filter --features bin,tract,wav-utils,transforms --bin deep-filter`).

## Licencje danych

- LibriSpeech: CC BY 4.0.
- Common Voice PL: CC0 (pobranie wymaga konta Mozilla Data Collective).
- Nagrania użytkownika: wyłącznie lokalnie, nigdy w gicie ani w treningu.
