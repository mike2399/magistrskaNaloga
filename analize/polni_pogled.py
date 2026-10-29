#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ali izbor 50 kandidatov omejuje jezikovni model? (brez klicev API)

Napake razdelimo v dve skupini. Pri napakah z največ 50 metodami je jezikovni model videl vse
metode (polni pogled), pri večjih programih pa le 50 najbolj sumljivih po klasičnih ocenah
(okrnjeni pogled). Na napakah s polnim pogledom primerjamo jezikovni model sam, kontrolo H9,
kombinacijo H9+LLM in kaskado brez učenja: če kombinacija tudi tu premaga model sam, prednost
ni le posledica tega, da klasične ocene pokrijejo napake, ki jih izbor zgreši.

Zagon:  python analize/polni_pogled.py
        (9. skripta v cevovodu, vrstni red je v README)
Vhod:   izhod/modeli/mlp_llm_napovedi.csv
Izhod:  izhod/analize/polni_pogled.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # koren repozitorija (nastavitve.py, skupno/)

import json
import os

import numpy as np
import pandas as pd

import nastavitve as N
from skupno.metrike import prvi_rangi, wilcoxon_parni
from skupno.preverbe import glava

MODELI = {"LLM sam": "llm", "H9": "mlp_h9", "H9+LLM": "mlp_h9llm", "kaskada": "kaskada"}
MEJA = N.K_KANDIDATOV


def main():
    glava("polni proti okrnjenemu pogledu jezikovnega modela")
    d = pd.read_csv(N.pot(N.MLP_LLM_NAPOVEDI))
    d["llm_scored"] = d["llm_scored"].astype(str).str.lower().isin(["true", "1"])

    info = d.groupby("fault_id").agg(n_methods=("faulty", "size"), n_cand=("llm_scored", "sum"), n_faulty=("faulty", "sum"))
    info["fullview"] = info["n_methods"] <= MEJA
    napake = list(info.index)
    polni = [f for f in napake if info.loc[f, "fullview"]]
    okrnjeni = [f for f in napake if not info.loc[f, "fullview"]]
    rangi = {ime: prvi_rangi(d, stolpec) for ime, stolpec in MODELI.items()}
    pokrita = {f: bool(d[(d["fault_id"] == f) & (d["faulty"] == 1)]["llm_scored"].any()) for f in napake}

    print(f"\nNapak skupaj: {len(napake)}")
    print(f"  polni pogled (do {MEJA} metod): {len(polni)}, mediana metod {int(info.loc[polni, 'n_methods'].median())}")
    print(f"  okrnjeni pogled (nad {MEJA} metod): {len(okrnjeni)}, mediana metod {int(info.loc[okrnjeni, 'n_methods'].median())}")

    tabele = {}

    def povzemi(skupina, kljuc, naslov):
        pokritost = float(np.mean([pokrita[f] for f in skupina]))
        print(f"\n[{naslov}: {len(skupina)} napak, kriva metoda med kandidati pri {pokritost:.1%}]")
        print(f"  {'model':16s} " + "  ".join(f"Top-{n}" for n in N.NS) + "     MFR")
        blok = {"n_faults": len(skupina), "coverage": pokritost, "models": {}}
        for ime in MODELI:
            r = np.array([rangi[ime][f] for f in skupina])
            top = [int((r <= n).sum()) for n in N.NS]
            print(f"  {ime:16s} " + "  ".join(f"{t:5d}" for t in top) + f"   {r.mean():6.2f}")
            blok["models"][ime] = {"top": {str(n): t for n, t in zip(N.NS, top)}, "mfr": float(r.mean())}
        tabele[kljuc] = blok

    povzemi(polni, "polni_pogled", "polni pogled")
    povzemi(okrnjeni, "okrnjeni_pogled", "okrnjeni pogled")

    print("\n[Parni Wilcoxonovi testi na napakah s polnim pogledom (enostranski)]")
    testi = {}
    for a, b in [("H9+LLM", "LLM sam"), ("H9+LLM", "H9"), ("LLM sam", "H9")]:
        w = wilcoxon_parni(rangi[a], rangi[b], polni, "less")
        znacilno = "značilno boljši" if (w["p"] == w["p"] and w["p"] < 0.05) else "ni značilno boljši"
        print(f"  {a} proti {b}: p = {w['p']:.3g}, {a} {znacilno} "
              f"(boljši {w['a_better']}, slabši {w['a_worse']}, enako {w['tie']})")
        testi[f"{a} vs {b}"] = w

    povzetek = {
        "opis": (f"Polni pogled = napake z največ {MEJA} metodami (jezikovni model je videl vse). Rangi po pravilu "
                 "povprečnega ranga, kot v mlp_llm.py. Wilcoxon je enostranski (alternative = less). Pri primerjavi "
                 "H9+LLM proti LLM samemu ima večina napak enak rang, zato ima test malo moči."),
        "n_faults_total": len(napake),
        "fullview": {"n": len(polni), "median_methods": int(info.loc[polni, "n_methods"].median())},
        "truncated": {"n": len(okrnjeni), "median_methods": int(info.loc[okrnjeni, "n_methods"].median())},
        "tabele": tabele,
        "testi_fullview": testi,
    }
    with open(N.pot(N.POLNI_POGLED_REZULTATI), "w", encoding="utf-8") as f:
        json.dump(povzetek, f, ensure_ascii=False, indent=2)
    print(f"\n[Shranjeno] {N.POLNI_POGLED_REZULTATI}")


if __name__ == "__main__":
    main()
