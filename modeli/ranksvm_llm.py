#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RankSVM z oceno jezikovnega modela kot deseto značilko.

Oceno jezikovnega modela dodamo še linearnemu modelu in oba kombinatorja primerjamo z isto deseto
značilko. Postopek je enak kot v ranksvm_h9.py (iste gube, C = 1, obe predobdelavi). Ker je surova
ocena jezikovnega modela na lestvici od 0 do 100, poročamo tudi naučene uteži. Top-1 poročamo pri
vseh treh obravnavah enakih ocen.

Zagon:  python modeli/ranksvm_llm.py
        (12. skripta v cevovodu, vrstni red je v README)
Vhod:   izhod/podatki/matrika_metod.parquet, jezikovni_model/ocene.csv, modeli/mlp_llm_napovedi.csv,
        modeli/ranksvm_h9_napovedi.csv, modeli/ranksvm_h9_rezultati.json
Izhod:  izhod/modeli/ranksvm_llm_napovedi.csv, modeli/ranksvm_llm_rezultati.json, modeli/ranksvm_llm_cache/,
        modeli/ranksvm_llm_tmp/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # koren repozitorija (nastavitve.py, skupno/)

import json
import os

import numpy as np
import pandas as pd

import nastavitve as N
from skupno.metrike import metrike_po_napakah, top1_po_konvencijah, wilcoxon_parni
from skupno.podatki import (nalozi_matriko_metod, seznam_napak, normiraj_po_napaki, gube_po_napaki,
                            zdruzi_llm_ocene, v_intervalu_0_1)
from skupno.preverbe import Preverbe, glava
from skupno.ranksvm import preveri_programa, ranksvm_cv, povzetek_utezi, C_REGULARIZACIJA

H10 = N.TEHNIKE + ["llm"]
MODELI = [("Ochiai", "ochiai"), ("rankSVM (surove, Zou)", "ranksvm_raw"), ("rankSVM (min-max)", "ranksvm_norm"),
          ("rankSVM H9+LLM (surove)", "ranksvm_llm_raw"), ("rankSVM H9+LLM (min-max)", "ranksvm_llm_norm"),
          ("MLP H9 (kontrola)", "mlp_h9"), ("MLP H9+LLM", "mlp_h9llm")]
PRIMERJAVE = [
    ("rankSVM H9+LLM (surove)", "rankSVM (surove, Zou)", "less", "ocena LLM pomaga linearnemu kombinatorju (surove značilke)"),
    ("rankSVM H9+LLM (min-max)", "rankSVM (min-max)", "less", "ocena LLM pomaga linearnemu kombinatorju (min-max)"),
    ("rankSVM H9+LLM (surove)", "MLP H9+LLM", "two-sided", "razlika med kombinatorjema z isto deseto značilko (surove proti MLP)"),
    ("rankSVM H9+LLM (min-max)", "MLP H9+LLM", "two-sided", "razlika med kombinatorjema pri isti predobdelavi (min-max)"),
    ("rankSVM H9+LLM (surove)", "Ochiai", "less", "rankSVM+LLM proti najboljši posamezni tehniki"),
    ("rankSVM H9+LLM (surove)", "rankSVM H9+LLM (min-max)", "two-sided", "vpliv lestvice ocene LLM (isti model, surove proti min-max)"),
]


def main():
    glava("rankSVM z oceno jezikovnega modela")
    preveri_programa()
    tmp, cache = N.pot(N.RANKSVM_LLM_TMP), N.pot(N.RANKSVM_LLM_CACHE)
    p = Preverbe()

    df = nalozi_matriko_metod(N.pot(N.MATRIKA_METOD))
    p("značilke so brez manjkajočih vrednosti", bool(df[N.TEHNIKE].notna().all().all()))
    napake = seznam_napak(df)
    qid = {fid: i + 1 for i, fid in enumerate(napake)}
    print(f"Matrika: {len(df):,} metod, {len(napake)} napak, {int(df['faulty'].sum())} krivih metod")
    gube = gube_po_napaki(napake)
    p("gube pokrijejo vsako napako natanko enkrat",
      sum(len(g) for g in gube) == len(napake) and set().union(*gube) == set(napake))
    p("velikosti gub so [35, 35, 35] in sedemkrat 34", sorted((len(g) for g in gube), reverse=True) == [35] * 3 + [34] * 7)

    df, n_ocenjenih = zdruzi_llm_ocene(df, N.pot(N.LLM_OCENE))
    p("ocena LLM je končna in v [0, 100]", bool(np.isfinite(df["llm"].to_numpy()).all())
      and float(df["llm"].min()) >= 0.0 and float(df["llm"].max()) <= 100.0,
      f"ocenjenih {n_ocenjenih:,} od {len(df):,} metod, ostale 0")

    print("\n[Lestvica surovih značilk]")
    lestvice = {}
    for c in H10:
        lestvice[c] = {"min": float(df[c].min()), "max": float(df[c].max())}
        print(f"  {c:22s} {lestvice[c]['min']:10.4g} .. {lestvice[c]['max']:10.4g}")

    print(f"\n[rankSVM H9+LLM na surovih značilkah (C = {C_REGULARIZACIJA})]")
    df["ranksvm_llm_raw"], utezi_raw = ranksvm_cv(df, H10, gube, qid, "rawllm", tmp, cache)
    print("\n[rankSVM H9+LLM na znotraj napake normiranih značilkah]")
    dfn = normiraj_po_napaki(df, H10)
    p("po normalizaciji je vseh deset značilk v [0, 1]", v_intervalu_0_1(dfn, H10))
    df["ranksvm_llm_norm"], utezi_norm = ranksvm_cv(dfn, H10, gube, qid, "normllm", tmp, cache)
    for c in ("ranksvm_llm_raw", "ranksvm_llm_norm"):
        p(f"{c}: vsaka metoda ima končno, nekonstantno napoved",
          bool(np.isfinite(df[c].to_numpy()).all()) and float(np.std(df[c])) > 0)

    # že izračunane napovedi iz ranksvm_h9.py in mlp_llm.py, ne računamo jih znova
    r10 = pd.read_csv(N.pot(N.RANKSVM_NAPOVEDI))
    r08 = pd.read_csv(N.pot(N.MLP_LLM_NAPOVEDI))
    for t in (r10, r08):
        for c in ("m_start", "m_end"):
            t[c] = t[c].astype(int)
    pred = len(df)
    df = df.merge(r10[N.KLJUC + ["ranksvm_raw", "ranksvm_norm"]], on=N.KLJUC, how="left")
    p("napovedi iz ranksvm_h9.py so pripete brez izgube", len(df) == pred and bool(df[["ranksvm_raw", "ranksvm_norm"]].notna().all().all()))
    r08 = r08[N.KLJUC + ["mlp_h9", "mlp_h9llm", "ochiai", "llm"]].rename(columns={"ochiai": "ochiai_08", "llm": "llm_08"})
    df = df.merge(r08, on=N.KLJUC, how="left")
    p("napovedi iz mlp_llm.py so pripete brez izgube", len(df) == pred and bool(df[["mlp_h9", "mlp_h9llm"]].notna().all().all()))
    # poravnava vrstic med skriptami, razlika sme biti le zaokrožitev pri zapisu v CSV
    razlika = float(np.abs(df["ochiai"].to_numpy(dtype=float) - df["ochiai_08"].to_numpy(dtype=float)).max())
    p("Ochiai iz mlp_llm.py se ujema z matriko (toleranca 1e-9)", razlika < 1e-9, f"največja razlika {razlika:.3e}")
    p("ocena LLM se ujema s stolpcem iz mlp_llm.py", bool(np.array_equal(df["llm"].to_numpy(), df["llm_08"].to_numpy())))
    df.drop(columns=["ochiai_08", "llm_08"], inplace=True)

    print(f"\n[Rezultati, {len(napake)} napak, {N.K_GUB}-kratno prečno preverjanje po napakah]")
    print(f"  {'model':26s} " + "  ".join(f"Top-{n}" for n in N.NS) + "     MFR     MAR")
    rez = {}
    for ime, stolpec in MODELI:
        top, mfr, mar, rang = metrike_po_napakah(df, stolpec)
        rez[ime] = {"top": top, "mfr": mfr, "mar": mar, "_rang": rang}
        print(f"  {ime:26s} " + "  ".join(f"{top[n]:5d}" for n in N.NS) + f"   {mfr:6.2f}  {mar:6.2f}")

    with open(N.pot(N.RANKSVM_REZULTATI), encoding="utf-8") as f:
        rez10 = json.load(f)["results"]
    odstopanja = [ime for ime in rez10 if not (all(rez[ime]["top"][n] == rez10[ime]["top"][str(n)] for n in N.NS)
                                               and abs(rez[ime]["mfr"] - rez10[ime]["mfr"]) < 1e-9
                                               and abs(rez[ime]["mar"] - rez10[ime]["mar"]) < 1e-9)]
    p("naložene napovedi reproducirajo rezultate ranksvm_h9.py", not odstopanja, ", ".join(odstopanja))

    print("\n[Top-1 po obravnavi enakih ocen: optimistična / povprečna / pesimistična]")
    for ime, stolpec in MODELI:
        t = top1_po_konvencijah(df, stolpec)
        rez[ime]["top1_tie"] = t
        print(f"  {ime:26s} {t['min']:5d} {t['avg']:5d} {t['max']:5d}")
    p("Top-1 pri povprečni obravnavi je enak kot v glavni tabeli", all(rez[ime]["top1_tie"]["avg"] == rez[ime]["top"][1] for ime, _ in MODELI))

    utezi = {}
    for oznaka, u in (("rawllm", utezi_raw), ("normllm", utezi_norm)):
        if u:
            utezi[oznaka] = povzetek_utezi(u, H10)
    if utezi:
        print("\n[Uteži linearnega modela, povprečje ± st. odklon čez gube]")
        print(f"  {'značilka':22s} " + "".join(f"{o:>24s}" for o in utezi))
        for c in H10:
            print(f"  {c:22s} " + "".join(f"{utezi[o][c]['mean']:15.5g} ±{utezi[o][c]['sd']:7.2g}" for o in utezi))
    p("uteži linearnega modela so shranjene", bool(utezi), "sicer poženite s praznim predpomnilnikom modeli/ranksvm_llm_cache")

    print("\n[Parni Wilcoxonovi testi na rangu prve krive metode]")
    testi = {}
    for a, b, alt, opis in PRIMERJAVE:
        w = wilcoxon_parni(rez[a]["_rang"], rez[b]["_rang"], napake, alt)
        w["opis"] = opis
        testi[f"{a} vs {b}"] = w
        print(f"  {opis}\n     W = {w['W']:.1f}, p = {w['p']:.3g}, A boljši {w['a_better']}, slabši {w['a_worse']}, enako {w['tie']} ({alt})")

    stolpci = N.KLJUC + ["faulty", "ochiai", "llm", "ranksvm_llm_raw", "ranksvm_llm_norm", "ranksvm_raw", "ranksvm_norm", "mlp_h9", "mlp_h9llm"]
    df[stolpci].to_csv(N.pot(N.RANKSVM_LLM_NAPOVEDI), index=False)
    povzetek = {
        "config": {"model": "svm_rank (CombineFL) z oceno LLM kot deseto značilko", "c": C_REGULARIZACIJA,
                   "seed_split": N.SEME_DELITVE, "k": N.K_GUB, "features": H10, "n_faults": len(napake),
                   "n_faulty": int(df["faulty"].sum()), "n_rows_with_llm_score": n_ocenjenih,
                   "llm_feature": f"{N.LLM_OCENE} (lestvica od 0 do 100), metoda brez ocene dobi 0",
                   "preprocessing_variants": ["surove (CombineFL)", "znotraj napake min-max (kot MLP)"],
                   "loaded_not_recomputed": {"ranksvm_raw/ranksvm_norm": N.RANKSVM_NAPOVEDI,
                                             "mlp_h9/mlp_h9llm": N.MLP_LLM_NAPOVEDI}},
        "feature_scales_raw": lestvice,
        "results": {ime: {"top": rez[ime]["top"], "mfr": rez[ime]["mfr"], "mar": rez[ime]["mar"], "top1_tie": rez[ime]["top1_tie"]}
                    for ime, _ in MODELI},
        "wilcoxon": testi,
        "weights_mean_sd_per_fold": utezi or None,
        "opombe": ["Metrika je rang prve krive metode po pravilu povprečnega ranga, ne mera E_inspect iz CombineFL.",
                   "Vrednost 0 ocene LLM združuje tri primere: metoda ni bila kandidat, model jo je ocenil z 0 "
                   "ali model zanjo ni vrnil ocene.",
                   "Napake niso neodvisne, ker so gnezdene v petih projektih, zato so vrednosti p optimistične."],
    }
    with open(N.pot(N.RANKSVM_LLM_REZULTATI), "w", encoding="utf-8") as f:
        json.dump(povzetek, f, ensure_ascii=False, indent=2)
    print(f"\n[Shranjeno] {N.RANKSVM_LLM_NAPOVEDI}, {N.RANKSVM_LLM_REZULTATI}")
    p.zakljuci()


if __name__ == "__main__":
    main()
