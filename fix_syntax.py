# fix_syntax.py — Repare automatiquement les smart quotes et docstrings
# Usage : python fix_syntax.py main.py

import sys
import re
from pathlib import Path


def fix_file(filepath):
    p = Path(filepath)
    if not p.exists():
        print(f"ERREUR : fichier introuvable : {filepath}")
        return False

    # 1. Lire le fichier (plusieurs encodages possibles)
    raw = p.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            content = raw.decode(enc)
            print(f"Lu avec encodage : {enc}")
            break
        except UnicodeDecodeError:
            continue
    else:
        print("ERREUR : impossible de decoder le fichier")
        return False

    original = content

    # 2. Remplacer TOUS les smart quotes Unicode par des guillemets droits
    replacements = {
        "\u201c": '"',   # " ouverture double
        "\u201d": '"',   # " fermeture double
        "\u2018": "'",   # ' ouverture simple
        "\u2019": "'",   # ' fermeture simple
        "\u00ab": '"',   # «
        "\u00bb": '"',   # »
        "\u2039": "'",   # ‹
        "\u203a": "'",   # ›
        "\u201e": '"',   # „
        "\u201f": '"',   # ‟
        "\u2013": "-",   # – tiret en
        "\u2014": "-",   # — tiret em
        "\u2015": "-",   # ―
        "\u2026": "...", # …
        "\u00a0": " ",   # espace insecable
        "\u202f": " ",   # espace fine insecable
        "\u2009": " ",   # espace fine
        "\u200b": "",    # zero-width space
        "\ufeff": "",    # BOM
        "\u2264": "<=",  # <=
        "\u2265": ">=",  # >=
        "\u2192": "->",  # ->
        "\u2190": "<-",  # <-
        "\u00b7": "/",   # middot (utilise dans les tableaux markdown)
        "\u2022": "*",   # bullet
        "\u00d7": "x",   # multiplication
        "\u00b1": "+/-", # plus-moins
        "\u03c3": "sigma", # sigma grec
        "\u03bb": "lambda", # lambda grec
        "\u03b2": "beta", # beta grec
    }
    for old, new in replacements.items():
        content = content.replace(old, new)

    # 3. S'assurer que le fichier se termine par un newline
    if not content.endswith("\n"):
        content += "\n"

    # 4. Verifier si quelque chose a change
    if content == original:
        print("Aucun changement necessaire (fichier deja propre).")
        return True

    # 5. Ecrire en UTF-8 sans BOM
    p.write_text(content, encoding="utf-8")

    n_diff = sum(1 for a, b in zip(original, content) if a != b)
    print(f"OK : {n_diff} caracteres corriges dans {filepath}")
    print(f"Fichier ecrit en UTF-8 sans BOM.")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python fix_syntax.py main.py")
        sys.exit(1)

    target = sys.argv[1]
    ok = fix_file(target)

    if ok:
        # 6. Verifier que le fichier compile
        try:
            with open(target, "r", encoding="utf-8") as f:
                source = f.read()
            compile(source, target, "exec")
            print("\nVERIFICATION : le fichier compile SANS erreur !")
        except SyntaxError as e:
            print(f"\nAVERTISSEMENT : erreur de syntaxe restante :")
            print(f"  Ligne {e.lineno} : {e.msg}")
            print(f"  Texte : {e.text}")
            print("\nEnvoie cette info pour correction manuelle.")
        except Exception as e:
            print(f"\nErreur de compilation : {e}")
