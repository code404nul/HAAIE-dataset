#!/usr/bin/env python3
"""
🧠 Qwen3-4B Creative Writing Tester
Prérequis :
  1. Installer Ollama        → https://ollama.com
  2. Installer la librairie  → pip install ollama
  3. Lancer ce script        → python3 qwen3_creative_test.py

Le modèle qwen3:4b sera téléchargé automatiquement si absent.
"""

import sys
import time

# ── Auto-install de la librairie ollama si absente ───────────────────────────
try:
    import ollama
except ImportError:
    import subprocess
    print("📦 Librairie 'ollama' absente — installation en cours...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ollama", "-q"])
    import ollama
    print("✅ Librairie installée.\n")

# ─────────────────────────────────────────────
# Couleurs terminal
# ─────────────────────────────────────────────
R  = "\033[0m"
B  = "\033[1m"
CY = "\033[96m"
GR = "\033[92m"
YL = "\033[93m"
MA = "\033[95m"
RD = "\033[91m"
DM = "\033[2m"

MODEL = "qwen3:4b"

# ─────────────────────────────────────────────
# Prompts Creative Writing
# ─────────────────────────────────────────────
PROMPTS = {
    "1": {
        "label": "🌊 Poème atmosphérique",
        "lang": "Français",
        "text": (
            "Écris un poème court et poignant sur une ville côtière "
            "abandonnée après une catastrophe naturelle. Utilise des "
            "images sensorielles fortes — sons, odeurs, lumières. "
            "Ton : mélancolique mais porteur d'espoir discret."
        ),
    },
    "2": {
        "label": "🗡️  Scène d'action narrative",
        "lang": "Français",
        "text": (
            "Écris une scène d'action de 200 mots maximum : une espionne "
            "doit traverser une place surveillée à minuit sans se faire "
            "repérer. Focale interne, présent de narration, tension maximale."
        ),
    },
    "3": {
        "label": "🤖 Sci-fi twist",
        "lang": "English",
        "text": (
            "Write a 150-word science fiction micro-story with an unexpected "
            "twist in the final sentence. Theme: an AI that discovers it has "
            "been running a simulation — but not the one it expected."
        ),
    },
    "4": {
        "label": "💌 Lettre historique (Van Gogh)",
        "lang": "Français",
        "text": (
            "Écris une lettre fictive de Vincent van Gogh à sa sœur Willemien, "
            "datée de la nuit où il a peint La Nuit étoilée. Il parle de ce "
            "qu'il ressent en peignant les étoiles, de sa solitude, et d'un "
            "espoir secret qu'il n'ose pas nommer."
        ),
    },
    "5": {
        "label": "🌙 Prose poétique",
        "lang": "Español",
        "text": (
            "Escribe un párrafo de prosa poética (100 palabras máximo) sobre "
            "el momento exacto entre el sueño y el despertar, cuando uno no "
            "sabe si todavía está en el sueño o ya en la realidad."
        ),
    },
    "6": {
        "label": "🔮 Dialogue fantastique",
        "lang": "Français",
        "text": (
            "Écris un dialogue de 15 répliques entre une vieille bibliothécaire "
            "et un livre qui refuse d'être lu. Le livre parle. La bibliothécaire "
            "est patiente mais ferme. Inclure de l'humour et une révélation finale "
            "sur pourquoi le livre résiste."
        ),
    },
    "7": {
        "label": "✏️  Prompt personnalisé",
        "lang": "Libre",
        "text": None,
    },
}

# ─────────────────────────────────────────────
def banner():
    print(f"""
{CY}{B}╔══════════════════════════════════════════════╗
║      🧠  Qwen3-4B Creative Writing Tester    ║
║      Propulsé par Ollama • Local & Privé     ║
╚══════════════════════════════════════════════╝{R}
""")


def ensure_model():
    """Vérifie si le modèle est présent ; le télécharge sinon."""
    client = ollama.Client()

    # Liste les modèles locaux
    try:
        local_models = [m.model for m in client.list().models]
    except Exception:
        print(f"{RD}❌ Impossible de joindre Ollama. Assure-toi qu'il est lancé.{R}")
        print(f"{YL}   → macOS/Linux : ollama serve{R}")
        print(f"{YL}   → ou installe Ollama : https://ollama.com{R}\n")
        sys.exit(1)

    # Normalise les noms (ex: "qwen3:4b" ou "qwen3:4b-...")
    model_present = any(MODEL in m for m in local_models)

    if model_present:
        print(f"{GR}✅ Modèle {B}{MODEL}{R}{GR} déjà disponible localement.{R}\n")
        return

    # Téléchargement avec barre de progression
    print(f"{YL}⬇️  Modèle {B}{MODEL}{R}{YL} non trouvé — téléchargement en cours...{R}")
    print(f"{DM}   (2–3 GB selon ta connexion, une seule fois){R}\n")

    downloaded = 0
    total = 0

    for progress in client.pull(MODEL, stream=True):
        status  = progress.status or ""
        completed = progress.completed or 0
        total_size = progress.total or 0

        if total_size:
            total = total_size
        if completed:
            downloaded = completed

        if total:
            pct = int(downloaded / total * 40)
            bar = "█" * pct + "░" * (40 - pct)
            mb_done = downloaded / 1024 / 1024
            mb_total = total / 1024 / 1024
            print(f"\r{CY}[{bar}] {mb_done:.0f}/{mb_total:.0f} MB{R}", end="", flush=True)
        elif status:
            print(f"\r{DM}{status:<60}{R}", end="", flush=True)

    print(f"\n{GR}✅ Modèle {B}{MODEL}{R}{GR} prêt !{R}\n")


def show_menu():
    print(f"{B}Choisis un prompt de test :{R}\n")
    for key, p in PROMPTS.items():
        print(f"  {CY}{B}{key}{R}. {p['label']}  {DM}[{p['lang']}]{R}")
    print(f"\n  {DM}q. Quitter{R}\n")


def stream_response(prompt: str):
    """Envoie le prompt via la librairie ollama et streame la réponse."""
    client = ollama.Client()

    print(f"\n{MA}{B}── Réponse de {MODEL} {'─'*30}{R}\n")

    full_response = ""
    start = time.time()

    try:
        stream = client.generate(
            model=MODEL,
            prompt=f"/no_think\n{prompt}",
            stream=True,
            options={
                "temperature": 0.85,
                "top_p": 0.92,
                "repeat_penalty": 1.1,
                "num_predict": 800,
            },
        )
        for chunk in stream:
            token = chunk.response or ""
            print(token, end="", flush=True)
            full_response += token

    except KeyboardInterrupt:
        print(f"\n{YL}⚡ Génération interrompue.{R}")
    except ollama.ResponseError as e:
        print(f"\n{RD}❌ Erreur Ollama : {e}{R}")

    elapsed = time.time() - start
    words = len(full_response.split())
    print(f"\n\n{DM}── {words} mots générés • {elapsed:.1f}s ──────────────────{R}\n")


def run():
    banner()
    ensure_model()

    while True:
        show_menu()
        choice = input(f"{B}Ton choix :{R} ").strip().lower()

        if choice == "q":
            print(f"\n{GR}À bientôt ! 👋{R}\n")
            break

        if choice not in PROMPTS:
            print(f"{RD}Choix invalide, réessaie.{R}\n")
            continue

        selected = PROMPTS[choice]

        if selected["text"] is None:
            print(f"\n{YL}Écris ton prompt (ligne vide pour valider) :{R}")
            lines = []
            while True:
                line = input()
                if line == "" and lines:
                    break
                lines.append(line)
            user_prompt = "\n".join(lines).strip()
            if not user_prompt:
                print(f"{RD}Prompt vide, retour au menu.{R}\n")
                continue
            prompt = user_prompt
        else:
            prompt = selected["text"]

        print(f"\n{YL}{B}── Prompt [{selected['lang']}] {'─'*33}{R}")
        print(f"{DM}{prompt}{R}")

        stream_response(prompt)

        input(f"{DM}[Entrée pour continuer]{R}")
        print()


if __name__ == "__main__":
    run()