# -*- coding: utf-8 -*-
"""Delo z zbirko Defects4J: delovne kopije programov, izvoz lastnosti, sledi sklada, izseki kode."""
import os
import re
import subprocess

from nastavitve import DEFECTS4J, DEFECTS4J_HOME, D4J_CHECKOUTS

CAS_UKAZA = 600          # največ sekund za en klic programa defects4j


def preveri_defects4j():
    if not os.path.isfile(DEFECTS4J):
        raise RuntimeError(f"ne najdem programa defects4j: {DEFECTS4J} (nastavite DEFECTS4J_HOME)")


def pozeni(ukaz, timeout=CAS_UKAZA):
    """Požene ukaz in vrne (izhodna koda, stdout, stderr). Ob preteku časa vrne kodo 124."""
    try:
        r = subprocess.run([str(a) for a in ukaz], capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return 124, "", "TIMEOUT"


def mapa_kopije(projekt, bug, razlicica):
    """Mapa delovne kopije, npr. ~/d4j_checkouts/lang_1b."""
    return os.path.join(D4J_CHECKOUTS, f"{projekt.lower()}_{bug}{razlicica}")


def checkout(projekt, bug, razlicica, mapa):
    """Pripravi delovno kopijo (razlicica 'b' je napačna, 'f' popravljena). Obstoječo kopijo pusti."""
    oznaka = os.path.join(mapa, ".defects4j.config")
    if os.path.isfile(oznaka):
        return True
    os.makedirs(os.path.dirname(mapa), exist_ok=True)
    rc, _, _ = pozeni([DEFECTS4J, "checkout", "-p", projekt, "-v", f"{bug}{razlicica}", "-w", mapa])
    return rc == 0 and os.path.isfile(oznaka)


def export(lastnost, mapa):
    """Ukaz defects4j export -p <lastnost>, vrne (izhodna koda, vrednost)."""
    rc, out, _ = pozeni([DEFECTS4J, "export", "-p", lastnost, "-w", mapa])
    return rc, out


# okvirji ogrodij JUnit in Ant ter refleksije so šum, okvirji programa in JDK ostanejo
_SUM = re.compile(r"^\s*at (org\.junit|junit\.|sun\.reflect|jdk\.internal\.reflect|"
                  r"java\.lang\.reflect\.Method|org\.apache\.tools\.ant)")
_OKVIR = re.compile(r"^\s*at ")


def sled_sklada(projekt, bug, max_blokov=4, max_sporocilo=6, max_okvirjev=25):
    """Očiščena in skrajšana sled sklada sprožilnih testov iz Defects4J.

    Za vsak sprožilni test obdrži glavo, največ max_sporocilo vrstic sporočila in največ max_okvirjev
    okvirjev. Vrne (besedilo, število surovih vrstic, število obdržanih vrstic).
    """
    pot = os.path.join(DEFECTS4J_HOME, "framework", "projects", projekt, "trigger_tests", str(bug))
    if not os.path.isfile(pot):
        return "", 0, 0
    with open(pot, encoding="utf-8", errors="replace") as f:
        surovo = f.read().splitlines()
    bloki, blok = [], []
    for ln in surovo:
        if ln.startswith("---"):
            if blok:
                bloki.append(blok)
            blok = [ln]
        else:
            blok.append(ln)
    if blok:
        bloki.append(blok)

    out = []
    for b in bloki[:max_blokov]:
        n_sporocilo = n_okvirjev = okvirjev_skupaj = 0
        for j, ln in enumerate(b):
            if j == 0 and ln.startswith("---"):
                out.append(ln)
                continue
            if _SUM.match(ln):
                continue
            if _OKVIR.match(ln):
                okvirjev_skupaj += 1
                if n_okvirjev < max_okvirjev:
                    out.append(ln)
                    n_okvirjev += 1
            else:
                if n_sporocilo < max_sporocilo:
                    out.append(ln)
                    n_sporocilo += 1
        if okvirjev_skupaj > n_okvirjev:
            out.append(f"\t... ({okvirjev_skupaj - n_okvirjev} more frames)")
    if len(bloki) > max_blokov:
        out.append(f"... ({len(bloki) - max_blokov} more triggering tests omitted)")
    return "\n".join(out), len(surovo), len(out)


def izsek_kode(pot, zacetek, konec):
    """Vrstice od zacetek do konec z izvirnimi številkami vrstic, ali None, če datoteke ni."""
    if not os.path.isfile(pot):
        return None
    with open(pot, encoding="utf-8", errors="replace") as f:
        vrstice = f.read().split("\n")
    zacetek = max(1, zacetek)
    konec = min(len(vrstice), konec)
    return "\n".join(f"{i}: {vrstice[i - 1]}" for i in range(zacetek, konec + 1))


def pot_razreda(mapa_kopije_, srcdir, classname):
    """Pot do datoteke .java zunanjega razreda (notranji razredi za '$' so v isti datoteki)."""
    zunanji = classname.split("$")[0]
    return os.path.join(mapa_kopije_, srcdir, zunanji.replace(".", "/") + ".java")
