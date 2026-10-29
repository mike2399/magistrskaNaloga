#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Glavni poskus, MLP z oceno jezikovnega modela kot deseto značilko.

Primerja tri različice istega modela z isto delitvijo v gube: MLP H9 (devet klasičnih ocen, kontrola),
MLP H9+LLM (še ocena jezikovnega modela) in MLP H9+premešan (kontrolni preizkus, ocene jezikovnega
modela so znotraj vsake napake naključno premešane med kandidati). Poleg tega poroča Ochiai, jezikovni
model sam in kaskado brez učenja. Število epoh izbere na notranji validacijski gubi, končne zagone
ponovi s petimi semeni, izračuna statistične teste iz naloge in rezultate po projektih.

Zagon:  python modeli/mlp_llm.py
        (8. skripta v cevovodu, vrstni red je v README)
Vhod:   izhod/podatki/matrika_metod.parquet, jezikovni_model/ocene.csv
Izhod:  izhod/modeli/mlp_llm_izbrane_epohe.json, modeli/mlp_llm_cache/, modeli/mlp_llm_napovedi.csv,
        modeli/mlp_llm_rezultati.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # koren repozitorija (nastavitve.py, skupno/)

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import nastavitve as N
from skupno.metrike import metrike_po_napakah, wilcoxon_parni
from skupno.mlp import izberi_epohe, napovedi_cv
from skupno.podatki import (nalozi_matriko_metod, seznam_napak, normiraj_po_napaki, gube_po_napaki,
                            zdruzi_llm_ocene, v_intervalu_0_1, projekt_napake)
from skupno.preverbe import Preverbe, glava

MREZA_EPOH = [60, 150, 300, 600]
DELEZ_VALIDACIJE = 0.2
ROKE = {"MLP H9": N.TEHNIKE, "MLP H9+LLM": N.TEHNIKE + ["llm"], "MLP H9+premešan": N.TEHNIKE + ["llm_shuf"]}
STOLPEC_ROKE = {"MLP H9": "mlp_h9", "MLP H9+LLM": "mlp_h9llm", "MLP H9+premešan": "mlp_h9shuf"}
MODELI = [("Ochiai", "ochiai"), ("LLM (izbor+preureditev)", "llm"), ("MLP H9 (kontrola)", "mlp_h9"),
          ("MLP H9+LLM", "mlp_h9llm"), ("MLP H9+premešan (ablacija)", "mlp_h9shuf"),
          ("KASKADA LLM→H9 (brez učenja)", "kaskada")]


def tabela(df, podmnozica, naslov):
    print(f"\n[Rezultati: {naslov}]")
    print(f"  {'model':30s} " + "  ".join(f"Top-{n}" for n in N.NS) + "     MFR     MAR")
    out = {}
    for ime, stolpec in MODELI:
        top, mfr, mar, rang = metrike_po_napakah(df, stolpec, podmnozica=podmnozica)
        out[ime] = (top, mfr, mar, rang)
        print(f"  {ime:30s} " + "  ".join(f"{top[n]:5d}" for n in N.NS) + f"   {mfr:6.2f}  {mar:6.2f}")
    return out


def main():
    glava("MLP z oceno jezikovnega modela (glavni poskus)")
    p = Preverbe()
    df = nalozi_matriko_metod(N.pot(N.MATRIKA_METOD))
    df, n_ocenjenih = zdruzi_llm_ocene(df, N.pot(N.LLM_OCENE))

    # premešane ocene znotraj kandidatov iste napake (kontrolni preizkus)
    rs = np.random.default_rng(12345)
    df["llm_shuf"] = df["llm"].to_numpy(dtype=float)
    for fid, g in df.groupby("fault_id"):
        idx = g.index[g["llm_scored"].to_numpy()]
        if len(idx) > 1:
            v = df.loc[idx, "llm"].to_numpy(dtype=float).copy()
            rs.shuffle(v)
            df.loc[idx, "llm_shuf"] = v
    p("premešana značilka ima isto vsoto kot izvirna", abs(df["llm"].sum() - df["llm_shuf"].sum()) < 1e-6)

    napake = seznam_napak(df)
    krive = df[df["faulty"] == 1]
    pokrite = set(df[(df["faulty"] == 1) & (df["llm"] > 0)]["fault_id"])          # kriva metoda z oceno > 0
    med_kandidati = set(df[(df["faulty"] == 1) & df["llm_scored"]]["fault_id"])   # kriva metoda med kandidati
    print(f"\nMatrika: {len(df):,} metod, {len(napake)} napak, {int(df['faulty'].sum())} krivih metod")
    print(f"Jezikovni model je videl {n_ocenjenih:,} metod ({n_ocenjenih / len(df):.1%}), "
          f"ocena > 0 pri {int((df['llm'] > 0).sum()):,}")
    print(f"Kriva metoda med kandidati pri {len(med_kandidati)}/{len(napake)} napakah "
          f"({len(med_kandidati) / len(napake):.1%}), z oceno > 0 pri {len(pokrite)}/{len(napake)}")
    print(f"Krivih metod med kandidati: {int(krive['llm_scored'].sum())}/{len(krive)} ({krive['llm_scored'].mean():.1%})")

    vse_znacilke = N.TEHNIKE + ["llm", "llm_shuf"]
    dfn = normiraj_po_napaki(df, vse_znacilke)
    p("po normalizaciji so vse značilke v [0, 1]", v_intervalu_0_1(dfn, vse_znacilke))
    gube = gube_po_napaki(napake)
    p("gube pokrijejo vsako napako natanko enkrat",
      sum(len(g) for g in gube) == len(napake) and set().union(*gube) == set(napake))
    napaka_vrstice = df["fault_id"].to_numpy()

    # izbira števila epoh na notranji validaciji, shranjena, da je ponovljiva
    print("\n[Izbira števila epoh na notranji validacijski gubi]")
    pot_epoh = N.pot(N.MLP_LLM_EPOHE)
    if os.path.isfile(pot_epoh):
        with open(pot_epoh, encoding="utf-8") as f:
            shranjeno = json.load(f)
        izbrane, krivulje = shranjeno["chosen"], shranjeno["val_curves"]
        print(f"  naloženo iz {N.MLP_LLM_EPOHE}: {izbrane}")
    else:
        izbrane, krivulje = {}, {}
        for ime, znacilke in ROKE.items():
            izbrane[ime], krivulje[ime] = izberi_epohe(dfn, df, znacilke, gube, napaka_vrstice, N.SEME_DELITVE,
                                                       MREZA_EPOH, DELEZ_VALIDACIJE, ime)
        with open(pot_epoh, "w", encoding="utf-8") as f:
            json.dump({"chosen": izbrane, "val_curves": krivulje}, f, ensure_ascii=False, indent=2)

    # končni zagoni: tri roke, pet semen, napovedi gredo v predpomnilnik
    print(f"\n[Končni zagoni: {len(ROKE)} roke x {len(N.SEMENA)} semen]")
    cache = N.pot(N.MLP_LLM_CACHE)
    os.makedirs(cache, exist_ok=True)
    napovedi = {ime: {} for ime in ROKE}
    for ime, znacilke in ROKE.items():
        oznaka = ime.replace(" ", "_").replace("+", "-")
        for s in N.SEMENA:
            pot = os.path.join(cache, f"pred_{oznaka}_{s}_{izbrane[ime]}.npy")
            if os.path.isfile(pot):
                napovedi[ime][s] = np.load(pot)
                print(f"  {ime:16s} seme {s:5d} (epoh {izbrane[ime]}): iz predpomnilnika", flush=True)
            else:
                napovedi[ime][s] = napovedi_cv(dfn, df, znacilke, gube, napaka_vrstice, s, izbrane[ime])
                np.save(pot, napovedi[ime][s])
                print(f"  {ime:16s} seme {s:5d} (epoh {izbrane[ime]}): izračunano", flush=True)
    for ime in ROKE:
        df[STOLPEC_ROKE[ime]] = napovedi[ime][N.SEMENA[0]]

    # kaskada brez učenja: ocena jezikovnega modela, kjer obstaja, sicer normirana kontrola H9
    r = df.groupby("fault_id")["mlp_h9"].transform(lambda v: (v - v.min()) / (v.max() - v.min() + 1e-12))
    df["kaskada"] = np.where(df["llm_scored"], 2.0 + df["llm"] / 100.0, r)
    p("kaskada uvrsti ocenjene metode pred neocenjene",
      float(df.loc[df["llm_scored"], "kaskada"].min()) > float(df.loc[~df["llm_scored"], "kaskada"].max()))

    rez_vse = tabela(df, None, f"vse napake ({len(napake)})")
    rez_pokrite = tabela(df, pokrite, f"samo napake s krivo metodo z oceno > 0 ({len(pokrite)})")

    print("\n[Statistična značilnost, H9+LLM proti H9]")
    rang_h9, rang_llm = rez_vse["MLP H9 (kontrola)"][3], rez_vse["MLP H9+LLM"][3]
    w = wilcoxon_parni(rang_llm, rang_h9, napake, "less")
    print(f"  raven napake (n = {len(napake)}): W = {w['W']:.1f}, p = {w['p']:.3g}, "
          f"boljše {w['a_better']}, slabše {w['a_worse']}, enako {w['tie']}")
    print("  (napake niso neodvisne, ker so gnezdene v petih projektih, zato še test na ravni projektov)")
    projekt = {f: projekt_napake(f) for f in napake}
    projekti = sorted(set(projekt.values()))
    po_projektih = {}
    for pr in projekti:
        fs = [f for f in napake if projekt[f] == pr]
        po_projektih[pr] = {"n_faults": len(fs),
                            "mfr_h9": float(np.mean([rang_h9[f] for f in fs])),
                            "mfr_h9llm": float(np.mean([rang_llm[f] for f in fs])),
                            "top1_h9": sum(1 for f in fs if rang_h9[f] <= 1),
                            "top1_h9llm": sum(1 for f in fs if rang_llm[f] <= 1)}
    wp = wilcoxon_parni({pr: v["mfr_h9llm"] for pr, v in po_projektih.items()},
                        {pr: v["mfr_h9"] for pr, v in po_projektih.items()}, projekti, "less")
    print(f"  raven projekta (n = {len(projekti)}): p = {wp['p']:.3g}, izboljšanih projektov {wp['a_better']}/{len(projekti)}")
    for pr, v in po_projektih.items():
        print(f"     {pr:9s} n = {v['n_faults']:3d}  MFR H9 {v['mfr_h9']:7.2f} -> H9+LLM {v['mfr_h9llm']:7.2f}   "
              f"Top-1 {v['top1_h9']:3d} -> {v['top1_h9llm']:3d}")

    print("\n[Dodatni parni testi, ki jih navaja naloga (enostranski, A boljši od B)]")
    rang = {ime: rez_vse[ime][3] for ime, _ in MODELI}
    testi = {
        "shuffle_ablation_H9shuf_vs_H9": wilcoxon_parni(rang["MLP H9+premešan (ablacija)"], rang["MLP H9 (kontrola)"], napake, "less", "asymptotic"),
        "H9_vs_H9shuf": wilcoxon_parni(rang["MLP H9 (kontrola)"], rang["MLP H9+premešan (ablacija)"], napake, "less", "asymptotic"),
        "H9LLM_vs_LLM_alone": wilcoxon_parni(rang["MLP H9+LLM"], rang["LLM (izbor+preureditev)"], napake, "less", "asymptotic"),
        "H9LLM_vs_cascade": wilcoxon_parni(rang["MLP H9+LLM"], rang["KASKADA LLM→H9 (brez učenja)"], napake, "less", "asymptotic"),
    }
    for k, v in testi.items():
        print(f"  {k:32s} p = {v['p']:.4g}  (boljši {v['a_better']}, slabši {v['a_worse']}, enako {v['tie']})")

    print("\n[Robustnost na seme inicializacije uteži (delitev v gube je fiksna)]")
    po_semenih = {}
    for ime in ROKE:
        t1, mfr = [], []
        for s in N.SEMENA:
            tmp = df.copy()
            tmp["_p"] = napovedi[ime][s]
            top, m, _, _ = metrike_po_napakah(tmp, "_p")
            t1.append(top[1])
            mfr.append(m)
        po_semenih[ime] = (t1, mfr)
        print(f"  {ime:16s} Top-1 {np.mean(t1):6.1f} ± {np.std(t1, ddof=1):4.1f}   MFR {np.mean(mfr):6.2f} ± "
              f"{np.std(mfr, ddof=1):.2f}   (Top-1 po semenih {t1})")
    d_top1 = np.array(po_semenih["MLP H9+LLM"][0]) - np.array(po_semenih["MLP H9"][0])
    d_mfr = np.array(po_semenih["MLP H9+LLM"][1]) - np.array(po_semenih["MLP H9"][1])
    print(f"  razlika Top-1 (H9+LLM minus H9): {d_top1.mean():+.1f} ± {d_top1.std(ddof=1):.1f}, "
          f"izboljšanje pri {(d_top1 > 0).sum()}/{len(d_top1)} semenih")
    print(f"  razlika MFR   (H9+LLM minus H9): {d_mfr.mean():+.2f} ± {d_mfr.std(ddof=1):.2f}, "
          f"izboljšanje pri {(d_mfr < 0).sum()}/{len(d_mfr)} semenih")

    print("\n[Ocene jezikovnega modela: krive proti nekrivim metodam]")
    oc = df[df["llm_scored"]]
    print(f"  samo ocenjene metode: krive {oc[oc.faulty == 1]['llm'].mean():6.2f}, nekrive {oc[oc.faulty == 0]['llm'].mean():5.2f}")
    print(f"  delež ocen > 50 med ocenjenimi: krive {(oc[oc.faulty == 1]['llm'] > 50).mean():.1%}, "
          f"nekrive {(oc[oc.faulty == 0]['llm'] > 50).mean():.2%}")
    rho, _ = spearmanr(oc["llm"], oc["ochiai"])
    print(f"  Spearmanova korelacija med oceno LLM in Ochiai na kandidatih (n = {len(oc):,}): {rho:.3f} "
          f"(kandidati so že izbrani po klasičnih ocenah, zato je razpon omejen)")

    print("\n[Jezikovni model glede na težavnost napake za Ochiai]")
    _, _, _, rang_och = metrike_po_napakah(df, "ochiai")
    _, _, _, rang_l = metrike_po_napakah(df, "llm")
    print(f"  {'rang krive po Ochiai':>20s} | {'napak':>5s} | {'med kandidati':>13s} | {'LLM Top-1 (vse)':>15s} | {'LLM Top-1 (med kandidati)':>25s}")
    for lo, hi, oznaka in [(1, 1, "1"), (2, 3, "2 do 3"), (4, 20, "4 do 20"), (21, 10**9, "nad 20")]:
        izbor = [f for f in napake if lo <= rang_och[f] <= hi]
        if not izbor:
            continue
        kand = [f for f in izbor if f in med_kandidati]
        t_vse = np.mean([rang_l[f] <= 1 for f in izbor])
        t_kand = np.mean([rang_l[f] <= 1 for f in kand]) if kand else float("nan")
        print(f"  {oznaka:>20s} | {len(izbor):5d} | {len(kand) / len(izbor):12.1%} | {t_vse:14.1%} | {t_kand:24.1%}")

    stolpci = N.KLJUC + ["faulty", "ochiai", "llm", "llm_shuf", "llm_scored",
                         "mlp_h9", "mlp_h9llm", "mlp_h9shuf", "kaskada"]
    df[stolpci].to_csv(N.pot(N.MLP_LLM_NAPOVEDI), index=False)
    povzetek = {
        "chosen_epochs": izbrane, "val_curves": krivulje,
        "seeds": N.SEMENA, "n_faults": len(napake),
        "covered_faults": len(pokrite), "in_shortlist_faults": len(med_kandidati),
        "wilcoxon_fault_p": float(w["p"]), "wilcoxon_project_p": float(wp["p"]),
        "wilcoxon_fault": w, "wilcoxon_project": wp, "wilcoxon_tests": testi,
        "seed_delta_top1_mean": float(d_top1.mean()), "seed_delta_top1_sd": float(d_top1.std(ddof=1)),
        "seed_delta_mfr_mean": float(d_mfr.mean()), "seed_delta_mfr_sd": float(d_mfr.std(ddof=1)),
        "per_project": po_projektih,
        "all": {k: {"top": v[0], "mfr": v[1], "mar": v[2]} for k, v in rez_vse.items()},
        "covered_only": {k: {"top": v[0], "mfr": v[1], "mar": v[2]} for k, v in rez_pokrite.items()},
    }
    with open(N.pot(N.MLP_LLM_REZULTATI), "w", encoding="utf-8") as f:
        json.dump(povzetek, f, ensure_ascii=False, indent=2)
    print(f"\n[Shranjeno] {N.MLP_LLM_NAPOVEDI}, {N.MLP_LLM_REZULTATI}")
    p.zakljuci()


if __name__ == "__main__":
    main()
