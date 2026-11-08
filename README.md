# HAAIE Dataset

Multilingual conversational dataset used to fine-tune **[HAAIE](https://github.com/code404nul/HAAIE)**, a local AI companion designed to help with social anxiety and loneliness.

The goal is a companion that sounds human: warm when things are heavy, funny when they're not, honest when a user describes something risky, and never clinical.

## Contents

| File | Language | Examples |
|---|---|---|
| `dataset_en.jsonl` | English (source) | 1 222 |
| `dataset_fra.jsonl` | French | 1 223 |
| `dataset_spa.jsonl` | Spanish | 1 223 |
| `dataset_por.jsonl` | Portuguese | 1 223 |
| `dataset_deu.jsonl` | German | 1 223 |
| `dataset_jpn.jsonl` | Japanese | 1 223 |

The English set is the original; the other languages were machine-translated with [NLLB-200 (1.3B distilled)](https://huggingface.co/facebook/nllb-200-distilled-1.3B), so expect some awkward phrasing.

## Format

Alpaca-style JSONL, one example per line:

```json
{"instruction": "Can you help me?", "input": "", "output": "Always. What's going on?"}
```

- `instruction` — what the user says
- `input` — unused (empty)
- `output` — the companion's reply

The data mixes emotional support (loneliness, self-hatred, feeling invisible), casual small talk, and light absurd humor, so the model doesn't become a sad-only bot.

## Scripts

| Script | What it does |
|---|---|
| `transform.py` | Translates the English dataset into other languages locally with NLLB-200 |
| `get_feeling.py` | Detects emotions in a message (28 GoEmotions labels) with a multilingual classifier |
| `prompter.py` | Builds a short system prompt adapted to the detected emotion |
| `add_prompt_to_dataset.py` | Adds that emotional system prompt to each example of a `.jsonl` file |

```bash
python add_prompt_to_dataset.py dataset_en.jsonl
# → dataset_en_enriched.jsonl
```

## ⚠️ Limitations

- This dataset is **experimental** and was not reviewed by mental-health professionals.
- A model trained on it is **not a substitute for therapy or crisis support**. If you are in danger, contact your local emergency number or a crisis line.
- Some replies use informal or rough language on purpose; filter them if your use case needs a neutral tone.

## License

[Apache 2.0](LICENSE)
