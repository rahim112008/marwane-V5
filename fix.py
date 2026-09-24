# fix.py - Repare les smart quotes et caracteres speciaux
from pathlib import Path
import sys

def fix(path="main.py"):
    p = Path(path)
    if not p.exists():
        print(f"Fichier introuvable : {path}")
        return

    raw = p.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            text = raw.decode(enc)
            print(f"Lu avec : {enc}")
            break
        except UnicodeDecodeError:
            continue

    # TOUS les caracteres problematiques
    fixes = {
        "\u201c": '"', "\u201d": '"',
        "\u2018": "'", "\u2019": "'",
        "\u00ab": '"', "\u00bb": '"',
        "\u2013": "-", "\u2014": "-",
        "\u2026": "...", "\u00a0": " ",
        "\u202f": " ", "\u2009": " ",
        "\u200b": "", "\ufeff": "",
        "\u2264": "<=", "\u2265": ">=",
        "\u2192": "->", "\u2190": "<-",
        "\u00b7": "/", "\u2022": "*",
        "\u00d7": "x", "\u00b1": "+/-",
        "\u03c3": "sigma", "\u03bb": "lambda", "\u03b2": "beta",
    }
    n = 0
    for old, new in fixes.items():
        c = text.count(old)
        if c:
            text = text.replace(old, new)
            n += c

    if not text.endswith("\n"):
        text += "\n"

    p.write_text(text, encoding="utf-8")
    print(f"{n} caracteres corriges")

    # Verifier compilation
    try:
        compile(text, path, "exec")
        print("OK : le fichier compile !")
    except SyntaxError as e:
        print(f"ERREUR RESTANTE ligne {e.lineno} : {e.msg}")
        print(f"  >> {e.text}")

if __name__ == "__main__":
    fix(sys.argv[1] if len(sys.argv) > 1 else "main.py")
