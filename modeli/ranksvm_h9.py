#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Izhodišče iz literature, rankSVM (CombineFL) na matriki metod.

Linearni model rankSVM naučimo na istih devetih ocenah, z isto delitvijo v gube in isto metriko kot MLP.
Poročamo dve različici, surove ocene (kot v CombineFL) in znotraj napake normirane ocene (kot pri MLP).
Regularizacija C = 1 je privzeta vrednost iz CombineFL in je ne uglašujemo.

Zagon:  python modeli/ranksvm_h9.py
        (10. skripta v cevovodu, vrstni red je v README)
Vhod:   izhod/podatki/matrika_metod.parquet, modeli/mlp_llm_napovedi.csv, programa svm_rank iz CombineFL
Izhod:  izhod/modeli/ranksvm_h9_napovedi.csv, modeli/ranksvm_h9_rezultati.json, modeli/ranksvm_h9_rezultati.txt,
        modeli/ranksvm_h9_cache/ (napovedi in uteži), modeli/ranksvm_h9_tmp/ (datoteke LETOR in modeli)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # koren repozitorija (nastavitve.py, skupno/)

import json
import os

import numpy as np
import pandas as pd

import nastavitve as N
from skupno.metrike import metrike_po_napakah, wilcoxon_parni
from skupno.podatki import nalozi_matriko_metod, seznam_napak, normiraj_po_napaki, gube_po_napaki, v_intervalu_0_1
from skupno.preverbe import Preverbe, glava
from skupno.ranksvm import preveri_programa, ranksvm_cv, povzetek_utezi, C_REGULARIZACIJA

MODELI = [("Ochiai", "ochiai"), ("rankSVM (surove, Zou)", "ranksvm_raw"), ("rankSVM (min-max)", "ranksvm_norm"),
          ("MLP H9 (kontrola)", "mlp_h9"), ("MLP H9+LLM", "mlp_h9llm")]
PRIMERJAVE = [
    ("rankSVM (surove, Zou)", "Ochiai", "less", "rankSVM boljši od Ochiai (naučena kombinacija proti posamezni tehniki)"),
    ("MLP H9 (kontrola)", "rankSVM (surove, Zou)", "two-sided", "MLP H9 proti rankSVM (nevronski proti linearnemu kombinatorju)"),
    ("MLP H9+LLM", "rankSVM (surove, Zou)", "less", "MLP H9+LLM boljši od rankSVM"),
]


def main():
    glava("rankSVM (CombineFL) kot izhodišče na nivoju metod")
    preveri_programa()
    tmp, cache = N.pot(N.RANKSVM_TMP), N.pot(N.RANKSVM_CACHE)
    p = Preverbe()

    df = nalozi_matriko_metod(N.pot(N.MATRIKA_METOD))
    p("značilke so brez manjkajočih vrednosti", bool(df[N.TEHNIKE].notna().all().all()))
    napake = seznam_napak(df)
    qid = {fid: i + 1 for i, fid in enumerate(napake)}
    print(f"Matrika: {len(df):,} metod, {len(napake)} napak, {int(df['faulty'].sum())} krivih metod")
    gube = gube_po_napaki(napake)
    p("gube pokrijejo vsako napako natanko enkrat",
      sum(len(g) for g in gube) == len(napake) and set().union(*gube) == set(napake))
    napaka_vrstice = df["fault_id"].to_numpy()
    p("nobena testna napaka ni v učni množici svoje gube",
      all(not (set(napaka_vrstice[~np.isin(napaka_vrstice, list(t))]) & set(t)) for t in gube))

    print(f"\n[rankSVM na surovih ocenah (kot CombineFL, C = {C_REGULARIZACIJA})]")
    df["ranksvm_raw"], utezi_raw = ranksvm_cv(df, N.TEHNIKE, gube, qid, "raw", tmp, cache)
    print("\n[rankSVM na znotraj napake normiranih ocenah (kot MLP)]")
    dfn = normiraj_po_napaki(df, N.TEHNIKE)
    p("po normalizaciji so vse značilke v [0, 1]", v_intervalu_0_1(dfn, N.TEHNIKE))
    df["ranksvm_norm"], utezi_norm = ranksvm_cv(dfn, N.TEHNIKE, gube, qid, "norm", tmp, cache)
    p("napovedi rankSVM niso konstantne", float(np.std(df["ranksvm_raw"])) > 0 and float(np.std(df["ranksvm_norm"])) > 0)

    # napovedi MLP iz mlp_llm.py za skupno tabelo
    mlp = pd.read_csv(N.pot(N.MLP_LLM_NAPOVEDI))
    for c in ("m_start", "m_end"):
        mlp[c] = mlp[c].astype(int)
    pred = len(df)
    df = df.merge(mlp[N.KLJUC + ["mlp_h9", "mlp_h9llm"]], on=N.KLJUC, how="left")
    p("napovedi MLP so pripete brez izgube ali podvajanja vrstic",
      len(df) == pred and bool(df["mlp_h9"].notna().all()) and bool(df["mlp_h9llm"].notna().all()))

    print(f"\n[Rezultati, {len(napake)} napak, {N.K_GUB}-kratno prečno preverjanje po napakah]")
    print(f"  {'model':26s} " + "  ".join(f"Top-{n}" for n in N.NS) + "     MFR     MAR")
    rez = {}
    for ime, stolpec in MODELI:
        top, mfr, mar, rang = metrike_po_napakah(df, stolpec)
        rez[ime] = {"top": top, "mfr": mfr, "mar": mar, "_rang": rang}
        print(f"  {ime:26s} " + "  ".join(f"{top[n]:5d}" for n in N.NS) + f"   {mfr:6.2f}  {mar:6.2f}")

    print("\n[Parni Wilcoxonovi testi na rangu prve krive metode]")
    testi = {}
    for a, b, alt, opis in PRIMERJAVE:
        w = wilcoxon_parni(rez[a]["_rang"], rez[b]["_rang"], napake, alt)
        testi[f"{a} vs {b}"] = w
        print(f"  {opis}\n     W = {w['W']:.1f}, p = {w['p']:.3g}, A boljši {w['a_better']}, slabši {w['a_worse']}, "
              f"enako {w['tie']} ({alt})")

    utezi = {}
    for oznaka, u in (("raw", utezi_raw), ("norm", utezi_norm)):
        if u:
            utezi[oznaka] = povzetek_utezi(u, N.TEHNIKE)
    if utezi:
        print("\n[Uteži linearnega modela, povprečje čez gube]")
        for oznaka, u in utezi.items():
            print(f"  {oznaka:5s} " + ", ".join(f"{t} {u[t]['mean']:.3g}" for t in N.TEHNIKE))

    df[N.KLJUC + ["faulty", "ochiai", "ranksvm_raw", "ranksvm_norm", "mlp_h9", "mlp_h9llm"]].to_csv(
        N.pot(N.RANKSVM_NAPOVEDI), index=False)
    povzetek = {
        "config": {"model": "svm_rank (CombineFL)", "c": C_REGULARIZACIJA, "seed_split": N.SEME_DELITVE, "k": N.K_GUB,
                   "features": N.TEHNIKE, "n_faults": len(napake), "n_faulty": int(df["faulty"].sum()),
                   "preprocessing_variants": ["surove (CombineFL)", "znotraj napake min-max (kot MLP)"],
                   "metric": "rang prve krive metode po pravilu povprečnega ranga, kot v mlp_llm.py"},
        "results": {ime: {"top": rez[ime]["top"], "mfr": rez[ime]["mfr"], "mar": rez[ime]["mar"]} for ime, _ in MODELI},
        "wilcoxon": testi,
        "weights_mean_sd_per_fold": utezi or None,
    }
    with open(N.pot(N.RANKSVM_REZULTATI), "w", encoding="utf-8") as f:
        json.dump(povzetek, f, ensure_ascii=False, indent=2)
    with open(N.pot(N.RANKSVM_TXT), "w", encoding="utf-8") as f:
        f.write(f"rankSVM (CombineFL), nivo metod, {len(napake)} napak, {N.K_GUB}-kratno prečno preverjanje po napakah\n")
        f.write(f"svm_rank -c {C_REGULARIZACIJA}, metrika kot v mlp_llm.py\n\n")
        f.write(f"{'model':26s} " + "  ".join(f"Top-{n}" for n in N.NS) + "   MFR    MAR\n")
        for ime, _ in MODELI:
            t = rez[ime]["top"]
            f.write(f"{ime:26s} " + "  ".join(f"{t[n]:5d}" for n in N.NS) + f"   {rez[ime]['mfr']:.2f}  {rez[ime]['mar']:.2f}\n")
    print(f"\n[Shranjeno] {N.RANKSVM_NAPOVEDI}, {N.RANKSVM_REZULTATI}, {N.RANKSVM_TXT}")
    p.zakljuci()


if __name__ == "__main__":
    main()
