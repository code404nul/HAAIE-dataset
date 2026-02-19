"""
Traduction locale EN → FR d'un dataset JSONL
Modèle : facebook/nllb-200-distilled-1.3B

Installation :
    pip install transformers torch sentencepiece accelerate

Usage :
    python translate_dataset.py
    python translate_dataset.py --input dataset_clean.jsonl --output dataset_fr.jsonl
    python translate_dataset.py --input dataset_clean.jsonl --output dataset_fr.jsonl --device cpu
"""

import json
import argparse
import time
import sys
from pathlib import Path

# ─── CONFIG ────────────────────────────────────────────────────────────────────

MODEL_NAME   = "facebook/nllb-200-distilled-1.3B"
SRC_LANG     = "eng_Latn"
TGT_LANG     = ["tur_Latn"]
BATCH_SIZE   = 8          # Réduis à 4 si tu manques de VRAM
MAX_LENGTH   = 512        # Longueur max des tokens générés

# Champs du JSONL à traduire
FIELDS_TO_TRANSLATE = ["instruction", "input", "output"]

# ─── INIT ──────────────────────────────────────────────────────────────────────

def load_model(device: str):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch

    print(f"⏳ Chargement du modèle {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model = model.to(device)
    model.eval()
    print(f"✅ Modèle chargé sur {device}")
    return tokenizer, model


# ─── TRADUCTION ────────────────────────────────────────────────────────────────

def translate_batch(texts: list[str], tokenizer, model, language, device: str) -> list[str]:
    import torch

    # Filtre les textes vides
    non_empty = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
    if not non_empty:
        return [""] * len(texts)

    indices, valid_texts = zip(*non_empty)

    inputs = tokenizer(
        list(valid_texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    ).to(device)

    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(language),
            max_length=MAX_LENGTH,
            num_beams=4,
            early_stopping=True,
        )

    translations = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

    # Réinsère les textes vides à leur place
    result = [""] * len(texts)
    for idx, translation in zip(indices, translations):
        result[idx] = translation

    return result


# ─── PIPELINE ──────────────────────────────────────────────────────────────────

def translate_dataset(input_path: str, output_path: str, language: str, device: str):
    tokenizer, model = load_model(device)

    # Lecture
    entries = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    print(f"📂 {len(entries)} entrées chargées depuis {input_path}")

    # Collecte tous les textes à traduire avec leur position
    tasks = []  # (entry_idx, field, text)
    for i, entry in enumerate(entries):
        for field in FIELDS_TO_TRANSLATE:
            text = entry.get(field, "")
            if isinstance(text, str) and text.strip():
                tasks.append((i, field, text))

    print(f"🔤 {len(tasks)} textes à traduire...\n")

    # Traitement par batch
    total_batches = (len(tasks) + BATCH_SIZE - 1) // BATCH_SIZE
    t0 = time.time()

    for batch_idx in range(0, len(tasks), BATCH_SIZE):
        batch = tasks[batch_idx : batch_idx + BATCH_SIZE]
        texts = [t[2] for t in batch]

        translations = translate_batch(texts, tokenizer, model, language, device)

        for (entry_idx, field, _), translated in zip(batch, translations):
            entries[entry_idx][field] = translated

        # Progression
        done = min(batch_idx + BATCH_SIZE, len(tasks))
        elapsed = time.time() - t0
        speed = done / elapsed if elapsed > 0 else 0
        remaining = (len(tasks) - done) / speed if speed > 0 else 0
        batch_num = batch_idx // BATCH_SIZE + 1

        sys.stdout.write(
            f"\r  Batch {batch_num}/{total_batches} | "
            f"{done}/{len(tasks)} textes | "
            f"{speed:.1f} txt/s | "
            f"ETA {remaining:.0f}s   "
        )
        sys.stdout.flush()

    print(f"\n\n✅ Traduction terminée en {time.time() - t0:.1f}s")

    # Écriture
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"💾 Dataset sauvegardé → {output_path}")


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def detect_device():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"🎮 GPU détecté : {name}")
            return "cuda"
    except ImportError:
        pass
    print("💻 Pas de GPU détecté, utilisation du CPU (lent)")
    return "cpu"


if __name__ == "__main__":


    device = detect_device()
    for lang in TGT_LANG:
        print(f"\n=== Traduction vers {lang} ===")
        translate_dataset("dataset_clean.jsonl", f"dataset_{lang[0:3]}.jsonl", lang, device)