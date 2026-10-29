#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prečno preverjanje po projektih (leave-one-project-out, LOPO).

Pet gub je pet projektov. Za vsak projekt model naučimo na napakah ostalih štirih in ga preizkusimo na
izpuščenem, tako da je vsaka napaka napovedana z modelom, ki njenega projekta ni videl. Primerjamo
MLP H9 in MLP H9+LLM. Model, normalizacija, metrika in semena so enaki kot v mlp_llm.py, od tam je
prevzeto tudi število epoh. Ker so testne množice zelo neenake, poročamo rezultate po projektih in združeno.

Zagon:  python modeli/mlp_lopo.py
        (11. skripta v cevovodu, vrstni red je v README)
Vhod:   izhod/podatki/matrika_metod.parquet, jezikovni_model/ocene.csv, modeli/mlp_llm_izbrane_epohe.json,
        modeli/mlp_llm_rezultati.json
Izhod:  izhod/modeli/mlp_lopo_napovedi.csv, modeli/mlp_lopo_rezultati.json, modeli/mlp_lopo_cache/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # koren repozitorija (nastavitve.py, skupno/)

import json
import os
import time

import numpy as np
import torch

import nastavitve as N
from skupno.metrike import metrike_po_napakah, wilcoxon_parni
from skupno.mlp import uci_z_mejniki
from skupno.podatki import (nalozi_matriko_metod, seznam_napak, normiraj_po_napaki, zdruzi_llm_ocene,
                            v_intervalu_0_1, projekt_napake, kljuc_predpomnilnika)
from skupno.preverbe import Preverbe, glava

PRIMARNO_SEME = N.SEMENA[0]


def lopo(dfn, df, znacilke, projekti, projekt_vrstice, seme, epohe, oznaka, cache):
    """Za vsak projekt uči na ostalih in napove vse metode izpuščenega projekta. Guba k dobi seme (seme + k)."""
    kljuc = kljuc_predpomnilnika(dfn, znacilke, [epohe, seme, len(dfn)], predpona=b"LOPO|")
    pot = os.path.join(cache, f"lopo_{oznaka}_{seme}_{epohe}_{kljuc}.npy")
    if os.path.isfile(pot):
        print(f"  {oznaka:16s} seme {seme:5d} (epoh {epohe}): iz predpomnilnika", flush=True)
        return np.load(pot)
    X = dfn[znacilke].to_numpy(dtype=np.float32)
    y = df["faulty"].to_numpy(dtype=np.float32)
    napoved = np.full(len(df), np.nan, dtype=np.float64)
    zacetek = time.time()
    for k, pr in enumerate(projekti, 1):
        te = (projekt_vrstice == pr)
        out = uci_z_mejniki(torch.from_numpy(X[~te]), torch.from_numpy(y[~te]), torch.from_numpy(X[te]),
                            len(znacilke), seme + k, [epohe])
        napoved[te] = out[epohe]
        print(f"  {oznaka:16s} seme {seme:5d} guba {k}/5 (test {pr}, {int(te.sum()):6d} metod) [{time.time() - zacetek:5.0f} s]",
              flush=True)
    np.save(pot, napoved)
    return napoved


def main():
    glava("prečno preverjanje po projektih (LOPO)")
    cache = N.pot(N.LOPO_CACHE)
    os.makedirs(cache, exist_ok=True)
    p = Preverbe()

    df = nalozi_matriko_metod(N.pot(N.MATRIKA_METOD))
    df, n_ocenjenih = zdruzi_llm_ocene(df, N.pot(N.LLM_OCENE))
    napake = seznam_napak(df)
    znacilke_llm = N.TEHNIKE + ["llm"]
    p("značilke so brez manjkajočih vrednosti", bool(df[znacilke_llm].notna().all().all()))

    projekt_napak = {f: projekt_napake(f) for f in napake}
    projekt_vrstice = np.array([projekt_napake(f) for f in df["fault_id"]])
    p("projekt, izpeljan iz oznake napake, je enak stolpcu project v matriki",
      bool((projekt_vrstice == df["project"].to_numpy()).all()))
    projekti = sorted(set(projekt_napak.values()))
    p("natanko pet projektov", projekti == ["chart", "closure", "lang", "math", "time"], str(projekti))
    print(f"\nMatrika: {len(df):,} metod, {len(napake)} napak, {int(df['faulty'].sum())} krivih metod, "
          f"ocenjenih z LLM {n_ocenjenih:,}")

    print("\n[Gube: test je izpuščeni projekt, učenje ostali štirje]")
    gube, info_gub = {}, {}
    for pr in projekti:
        gube[pr] = {f for f in napake if projekt_napak[f] == pr}
        te = (projekt_vrstice == pr)
        info_gub[pr] = {"n_test_faults": len(gube[pr]), "n_test_methods": int(te.sum()),
                        "n_train_faults": len(napake) - len(gube[pr]), "n_train_methods": int((~te).sum()),
                        "n_train_faulty_methods": int(df.loc[~te, "faulty"].sum())}
        i = info_gub[pr]
        print(f"  {pr:8s} test {i['n_test_faults']:3d} napak / {i['n_test_methods']:6d} metod, "
              f"učenje {i['n_train_faults']:3d} napak / {i['n_train_methods']:6d} metod")
    p("gube razdelijo vse napake natanko enkrat",
      sum(len(g) for g in gube.values()) == len(napake) and set().union(*gube.values()) == set(napake))
    p("vsaka napaka pripada natanko enemu projektu", bool((df.groupby("fault_id")["project"].nunique() == 1).all()))
    p("vsaka učna množica vsebuje krive metode", all(i["n_train_faulty_methods"] >= 1 for i in info_gub.values()))

    dfn = normiraj_po_napaki(df, znacilke_llm)
    p("po normalizaciji so vse značilke v [0, 1]", v_intervalu_0_1(dfn, znacilke_llm))

    with open(N.pot(N.MLP_LLM_EPOHE), encoding="utf-8") as f:
        epohe = json.load(f)["chosen"]
    roke = {"MLP H9": (N.TEHNIKE, int(epohe["MLP H9"])), "MLP H9+LLM": (znacilke_llm, int(epohe["MLP H9+LLM"]))}
    print(f"\n[Število epoh iz mlp_llm.py: H9 {roke['MLP H9'][1]}, H9+LLM {roke['MLP H9+LLM'][1]}, semena {N.SEMENA}]")

    napovedi = {ime: {} for ime in roke}
    for ime, (znacilke, ep) in roke.items():
        oznaka = ime.replace(" ", "_").replace("+", "-")
        for s in N.SEMENA:
            napovedi[ime][s] = lopo(dfn, df, znacilke, projekti, projekt_vrstice, s, ep, oznaka, cache)
    p("vsaka metoda ima napoved pri obeh rokah in vseh semenih",
      all(len(v) == len(df) and not np.isnan(v).any() for roka in napovedi.values() for v in roka.values()))
    df["lopo_h9"] = napovedi["MLP H9"][PRIMARNO_SEME]
    df["lopo_h9llm"] = napovedi["MLP H9+LLM"][PRIMARNO_SEME]

    metrike = {ime: {} for ime in roke}
    for ime in roke:
        for s in N.SEMENA:
            tmp = df.copy()
            tmp["_p"] = napovedi[ime][s]
            metrike[ime][s] = metrike_po_napakah(tmp, "_p")

    # Ochiai ni naučen, zato mora biti enak kot v mlp_llm.py (preverba poravnave podatkov)
    och = metrike_po_napakah(df, "ochiai")
    with open(N.pot(N.MLP_LLM_REZULTATI), encoding="utf-8") as f:
        rez08 = json.load(f)
    o08 = rez08["all"]["Ochiai"]
    p("Ochiai je enak kot v mlp_llm.py", all(och[0][n] == o08["top"][str(n)] for n in N.NS)
      and abs(och[1] - o08["mfr"]) < 1e-9 and abs(och[2] - o08["mar"]) < 1e-9)

    print(f"\n[Rezultati po projektih, seme {PRIMARNO_SEME}]")
    print(f"  {'projekt':9s}{'n':>4s}  {'roka':8s} " + "  ".join(f"Top-{n}" for n in N.NS) + "      MFR      MAR")
    po_projektih = {}
    for pr in projekti:
        vrstica = {}
        for ime, stolpec in [("MLP H9", "lopo_h9"), ("MLP H9+LLM", "lopo_h9llm")]:
            top, mfr, mar, _ = metrike_po_napakah(df, stolpec, podmnozica=gube[pr])
            vrstica[ime] = {"n": len(gube[pr]), "top": top, "mfr": mfr, "mar": mar}
        po_projektih[pr] = vrstica
        a, b = vrstica["MLP H9+LLM"], vrstica["MLP H9"]
        print(f"  {pr:9s}{len(gube[pr]):>4d}  {'H9':8s} " + "  ".join(f"{b['top'][n]:5d}" for n in N.NS) + f"  {b['mfr']:7.2f}  {b['mar']:7.2f}")
        print(f"  {'':9s}{'':4s}  {'H9+LLM':8s} " + "  ".join(f"{a['top'][n]:5d}" for n in N.NS) + f"  {a['mfr']:7.2f}  {a['mar']:7.2f}")

    print(f"\n[Združeno čez vseh {len(napake)} napak, seme {PRIMARNO_SEME}]")
    print(f"  {'model':34s} " + "  ".join(f"Top-{n}" for n in N.NS) + "      MFR      MAR")
    zdruzeno = {}
    for ime, m in [("Ochiai (netrenirana referenca)", och), ("MLP H9 (LOPO)", metrike["MLP H9"][PRIMARNO_SEME]),
                   ("MLP H9+LLM (LOPO)", metrike["MLP H9+LLM"][PRIMARNO_SEME])]:
        zdruzeno[ime] = {"top": m[0], "mfr": m[1], "mar": m[2]}
        print(f"  {ime:34s} " + "  ".join(f"{m[0][n]:5d}" for n in N.NS) + f"  {m[1]:7.2f}  {m[2]:7.2f}")

    h9_10, llm_10 = rez08["all"]["MLP H9 (kontrola)"], rez08["all"]["MLP H9+LLM"]
    lo_h9, lo_llm = metrike["MLP H9"][PRIMARNO_SEME], metrike["MLP H9+LLM"][PRIMARNO_SEME]
    print(f"\n[Primerjava zasnov: LOPO proti {N.K_GUB}-kratnemu prečnemu preverjanju po napakah iz mlp_llm.py]")
    print(f"  {'roka':8s} {'zasnova':28s} {'Top-1':>6s} {'MFR':>8s}")
    for roka, zasnova, top1, mfr in [("H9", "LOPO", lo_h9[0][1], lo_h9[1]), ("H9", "po napakah", h9_10["top"]["1"], h9_10["mfr"]),
                                     ("H9+LLM", "LOPO", lo_llm[0][1], lo_llm[1]), ("H9+LLM", "po napakah", llm_10["top"]["1"], llm_10["mfr"])]:
        print(f"  {roka:8s} {zasnova:28s} {top1:6d} {mfr:8.2f}")

    print("\n[Robustnost na seme inicializacije uteži]")
    po_semenih = {}
    for ime in roke:
        t1 = [metrike[ime][s][0][1] for s in N.SEMENA]
        mfr = [metrike[ime][s][1] for s in N.SEMENA]
        po_semenih[ime] = (t1, mfr)
        print(f"  {ime:12s} Top-1 {np.mean(t1):6.1f} ± {np.std(t1, ddof=1):4.1f}   MFR {np.mean(mfr):6.2f} ± {np.std(mfr, ddof=1):.2f}   (Top-1 po semenih {t1})")
    d_top1 = np.array(po_semenih["MLP H9+LLM"][0]) - np.array(po_semenih["MLP H9"][0])
    d_mfr = np.array(po_semenih["MLP H9+LLM"][1]) - np.array(po_semenih["MLP H9"][1])
    print(f"  razlika Top-1 (H9+LLM minus H9): {d_top1.mean():+.1f} ± {d_top1.std(ddof=1):.1f}, izboljšanje pri {(d_top1 > 0).sum()}/{len(d_top1)} semenih")
    print(f"  razlika MFR   (H9+LLM minus H9): {d_mfr.mean():+.2f} ± {d_mfr.std(ddof=1):.2f}, izboljšanje pri {(d_mfr < 0).sum()}/{len(d_mfr)} semenih")
    print(f"  (pri prečnem preverjanju po napakah: razlika Top-1 {rez08['seed_delta_top1_mean']:+.1f} ± "
          f"{rez08['seed_delta_top1_sd']:.1f}, razlika MFR {rez08['seed_delta_mfr_mean']:+.2f} ± {rez08['seed_delta_mfr_sd']:.2f})")
    print("  Opomba: pri LOPO se poslabša predvsem kontrola H9, roka z LLM se premakne malo. Razlika MFR je zato "
          "velika, a nestabilna med semeni, razlika Top-1 je stabilna.")

    print("\n[Statistika: H9+LLM proti H9 pod LOPO]")
    w_napake, w_projekti, mfr_projektov = {}, {}, {}
    for s in N.SEMENA:
        w_napake[s] = wilcoxon_parni(metrike["MLP H9+LLM"][s][3], metrike["MLP H9"][s][3], napake, "less")
        mfr_projektov[s] = {pr: {"mfr_h9": float(np.mean([metrike["MLP H9"][s][3][f] for f in gube[pr]])),
                                 "mfr_h9llm": float(np.mean([metrike["MLP H9+LLM"][s][3][f] for f in gube[pr]]))}
                            for pr in projekti}
        w_projekti[s] = wilcoxon_parni({pr: mfr_projektov[s][pr]["mfr_h9llm"] for pr in projekti},
                                       {pr: mfr_projektov[s][pr]["mfr_h9"] for pr in projekti}, projekti, "less")
        w_projekti[s]["improved_projects"] = w_projekti[s]["a_better"]
    wf, wp = w_napake[PRIMARNO_SEME], w_projekti[PRIMARNO_SEME]
    print(f"  raven napake (n = {len(napake)}, seme {PRIMARNO_SEME}): W = {wf['W']:.1f}, p = {wf['p']:.3g}, "
          f"boljše {wf['a_better']}, slabše {wf['a_worse']}, enako {wf['tie']}")
    print("     p po semenih: " + ", ".join(f"{s}: {w_napake[s]['p']:.3g}" for s in N.SEMENA))
    print(f"  raven projekta (n = 5, seme {PRIMARNO_SEME}): p = {wp['p']:.3g}, izboljšanih projektov {wp['a_better']}/5 "
          f"(najmanjši možni p pri n = 5 je 1/32 = 0,03125)")
    print("     p po semenih: " + ", ".join(f"{s}: {w_projekti[s]['p']:.3g}" for s in N.SEMENA))
    for pr in projekti:
        m = mfr_projektov[PRIMARNO_SEME][pr]
        print(f"     {pr:9s} MFR H9 {m['mfr_h9']:7.2f} -> H9+LLM {m['mfr_h9llm']:7.2f}")

    df[N.KLJUC + ["project", "faulty", "lopo_h9", "lopo_h9llm"]].to_csv(N.pot(N.LOPO_NAPOVEDI), index=False)
    povzetek = {
        "config": {"design": "LOPO, pet gub = pet projektov (test je izpuščeni projekt)", "fold_order": projekti,
                   "fold_seed_rule": "uci_z_mejniki(seme + k), k = 1..5 po abecednem redu projektov",
                   "arms": {ime: {"features": z, "epochs": ep} for ime, (z, ep) in roke.items()},
                   "epochs_source": N.MLP_LLM_EPOHE, "seeds": N.SEMENA, "primary_seed": PRIMARNO_SEME,
                   "n_rows": len(df), "n_faults": len(napake), "n_faulty_methods": int(df["faulty"].sum()),
                   "llm_scored_methods": n_ocenjenih, "folds": info_gub},
        "per_project_primary_seed": po_projektih,
        "pooled_primary_seed": zdruzeno,
        "per_seed_pooled": {ime: {str(s): {"top": metrike[ime][s][0], "mfr": metrike[ime][s][1], "mar": metrike[ime][s][2]}
                                  for s in N.SEMENA} for ime in roke},
        "seed_stats": {
            "top1": {ime: {"values": po_semenih[ime][0], "mean": float(np.mean(po_semenih[ime][0])),
                           "sd": float(np.std(po_semenih[ime][0], ddof=1))} for ime in roke},
            "mfr": {ime: {"values": po_semenih[ime][1], "mean": float(np.mean(po_semenih[ime][1])),
                          "sd": float(np.std(po_semenih[ime][1], ddof=1))} for ime in roke},
            "delta_top1": {"values": [int(x) for x in d_top1], "mean": float(d_top1.mean()),
                           "sd": float(d_top1.std(ddof=1)), "improved_seeds": int((d_top1 > 0).sum())},
            "delta_mfr": {"values": [float(x) for x in d_mfr], "mean": float(d_mfr.mean()),
                          "sd": float(d_mfr.std(ddof=1)), "improved_seeds": int((d_mfr < 0).sum())},
        },
        "wilcoxon_fault_level": {str(s): w_napake[s] for s in N.SEMENA},
        "wilcoxon_project_level": {str(s): w_projekti[s] for s in N.SEMENA},
        "per_project_mfr_by_seed": {str(s): mfr_projektov[s] for s in N.SEMENA},
        "first_ranks_primary_seed": {f: {"project": projekt_napak[f], "h9": metrike["MLP H9"][PRIMARNO_SEME][3][f],
                                         "h9llm": metrike["MLP H9+LLM"][PRIMARNO_SEME][3][f]} for f in napake},
        "context_10fold_05c": {"source": N.MLP_LLM_REZULTATI, "MLP H9": h9_10, "MLP H9+LLM": llm_10,
                               "seed_delta_top1_mean": rez08["seed_delta_top1_mean"], "seed_delta_top1_sd": rez08["seed_delta_top1_sd"],
                               "seed_delta_mfr_mean": rez08["seed_delta_mfr_mean"], "seed_delta_mfr_sd": rez08["seed_delta_mfr_sd"],
                               "wilcoxon_fault_p": rez08["wilcoxon_fault_p"], "wilcoxon_project_p": rez08["wilcoxon_project_p"]},
    }
    with open(N.pot(N.LOPO_REZULTATI), "w", encoding="utf-8") as f:
        json.dump(povzetek, f, ensure_ascii=False, indent=2)
    print(f"\n[Shranjeno] {N.LOPO_NAPOVEDI}, {N.LOPO_REZULTATI}")
    p.zakljuci()


if __name__ == "__main__":
    main()
