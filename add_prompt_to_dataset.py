import json
import os
import argparse
from pathlib import Path
from prompter import format_system_prompt


def enrich_jsonl(input_path: str, output_path: str = None, user_name: str = "buddy") -> None:
    """
    Lit un fichier .jsonl et enrichit chaque entrée en ajoutant un system prompt
    émotionnel au champ "instruction".

    Args:
        input_path  : Chemin vers le fichier .jsonl source
        output_path : Chemin de sortie (par défaut : <nom>_enriched.jsonl)
        user_name   : Nom utilisé dans le prompt généré
    """
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Fichier introuvable : {input_path}")

    if output_path is None:
        output_path = input_file.with_name(f"{input_file.stem}_enriched.jsonl")

    total = 0
    skipped = 0

    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line_number, raw_line in enumerate(fin, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue  # Ignore lignes vides

            # Charger l'entrée JSON
            try:
                entry = json.loads(raw_line)
            except json.JSONDecodeError as e:
                print(f"[SKIP] Ligne {line_number} invalide ({e})")
                skipped += 1
                continue

            # Récupérer l'instruction originale
            original_instruction = entry.get("instruction", "")
            if not original_instruction:
                print(f"[SKIP] Ligne {line_number} — champ 'instruction' vide ou absent")
                skipped += 1
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                continue

            # Générer le system prompt à partir de l'émotion détectée
            try:
                system_prompt = format_system_prompt(original_instruction, user_name=user_name)
            except Exception as e:
                print(f"[ERROR] Ligne {line_number} — format_system_prompt a échoué : {e}")
                skipped += 1
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                continue

            # Fusionner : system prompt + instruction originale
            entry["instruction"] = f"{system_prompt}\n{original_instruction}"

            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            total += 1

    print(f"\n✅ Terminé — {total} entrées enrichies, {skipped} ignorées")
    print(f"📄 Fichier de sortie : {output_path}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":

    enrich_jsonl("dataset_en.jsonl", "dataset.jsonl", "arch")