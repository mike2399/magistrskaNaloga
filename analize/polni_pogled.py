#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Polni proti okrnjenemu pogledu jezikovnega modela

Pri napakah z največ 50 metodami je jezikovni model videl vse metode (polni pogled), pri večjih
programih le 50 najbolj sumljivih po klasičnih ocenah (okrnjeni pogled). Na polnem pogledu
primerjamo LLM sam, H9, H9+LLM in kaskado brez učenja: če kombinacija zmaga tudi tu, prednost ni
le v tem, da klasične ocene pokrijejo napake, ki jih izbor zgreši.

Vhod:   izhod/modeli/mlp_llm_napovedi.csv
Izhod:  izhod/analize/polni_pogled.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # koren repozitorija

import json

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

    n_metod = d.groupby("fault_id").size()
    napake = list(n_metod.index)
    polni = [f for f in napake if n_metod[f] <= MEJA]
    okrnjeni = [f for f in napake if n_metod[f] > MEJA]
    rangi = {ime: prvi_rangi(d, stolpec) for ime, stolpec in MODELI.items()}
    pokrita = {f: bool(d[(d["fault_id"] == f) & (d["faulty"] == 1)]["llm_scored"].any()) for f in napake}

    print(f"\nNapak skupaj: {len(napake)}")
    print(f"  polni pogled (do {MEJA} metod): {len(polni)}, mediana metod {int(n_metod[polni].median())}")
    print(f"  okrnjeni pogled (nad {MEJA} metod): {len(okrnjeni)}, mediana metod {int(n_metod[okrnjeni].median())}")

    def povzemi(skupina, naslov):
        """Top-N in MFR vseh modelov na skupini napak, izpiše in vrne blok za JSON"""
        pokritost = float(np.mean([pokrita[f] for f in skupina]))
        print(f"\n[{naslov}: {len(skupina)} napak, kriva metoda med kandidati pri {pokritost:.1%}]")
        print(f"  {'model':16s} " + "  ".join(f"Top-{n}" for n in N.NS) + "     MFR")
        blok = {"n_faults": len(skupina), "coverage": pokritost, "models": {}}
        for ime in MODELI:
            r = np.array([rangi[ime][f] for f in skupina])
            top = [int((r <= n).sum()) for n in N.NS]
            print(f"  {ime:16s} " + "  ".join(f"{t:5d}" for t in top) + f"   {r.mean():6.2f}")
            blok["models"][ime] = {"top": {str(n): t for n, t in zip(N.NS, top)}, "mfr": float(r.mean())}
        return blok

    tabele = {"polni_pogled": povzemi(polni, "polni pogled"), "okrnjeni_pogled": povzemi(okrnjeni, "okrnjeni pogled")}

    print("\n[Parni Wilcoxonovi testi na napakah s polnim pogledom (enostranski)]")
    testi = {}
    for a, b in [("H9+LLM", "LLM sam"), ("H9+LLM", "H9"), ("LLM sam", "H9")]:
        w = wilcoxon_parni(rangi[a], rangi[b], polni, "less")
        znacilno = "značilno boljši" if w["p"] < 0.05 else "ni značilno boljši"
        print(f"  {a} proti {b}: p = {w['p']:.3g}, {a} {znacilno} "
              f"(boljši {w['a_better']}, slabši {w['a_worse']}, enako {w['tie']})")
        testi[f"{a} vs {b}"] = w

    povzetek = {
        "opis": (f"Polni pogled = napake z največ {MEJA} metodami (jezikovni model je videl vse). Rangi po pravilu "
                 "povprečnega ranga, kot v mlp_llm.py. Wilcoxon je enostranski (alternative = less). Pri primerjavi "
                 "H9+LLM proti LLM samemu ima večina napak enak rang, zato ima test malo moči."),
        "n_faults_total": len(napake),
        "fullview": {"n": len(polni), "median_methods": int(n_metod[polni].median())},
        "truncated": {"n": len(okrnjeni), "median_methods": int(n_metod[okrnjeni].median())},
        "tabele": tabele,
        "testi_fullview": testi,
    }
    with open(N.pot(N.POLNI_POGLED_REZULTATI), "w", encoding="utf-8") as f:
        json.dump(povzetek, f, ensure_ascii=False, indent=2)
    print(f"\n[Shranjeno] {N.POLNI_POGLED_REZULTATI}")


if __name__ == "__main__":
    main()
