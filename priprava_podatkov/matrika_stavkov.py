#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Matrika ocen na nivoju stavkov iz zbirke CombineFL.

Iz datoteke release.json sestavi tabelo z eno vrstico na mesto v programu (razred:vrstica) in napako.
Stolpci so ocene devetih tehnik in oznaka faulty (ali je mesto del razvijalčevega popravka).

Zagon:  python priprava_podatkov/matrika_stavkov.py
        (1. skripta v cevovodu, vrstni red je v README)
Vhod:   combinefl/data/release.json
Izhod:  izhod/podatki/matrika_stavkov.parquet, podatki/matrika_stavkov_vzorec.csv (prvih 200 vrstic),
        podatki/napake_brez_oznak_stavkov.txt (napake brez oznake faulty na nivoju stavka)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # koren repozitorija (nastavitve.py, skupno/)

import json
import os
from collections import Counter

import numpy as np
import pandas as pd

import nastavitve as N
from skupno.podatki import razcleni_fault_id
from skupno.preverbe import Preverbe, glava


def main():
    glava("matrika ocen na nivoju stavkov (CombineFL)")
    if not os.path.isfile(N.RELEASE_JSON):
        sys.exit(f"NAPAKA: ne najdem {N.RELEASE_JSON} (glej README, razdelek o podatkih CombineFL)")
    print(f"Vir: {N.RELEASE_JSON} ({os.path.getsize(N.RELEASE_JSON) / 2**20:.1f} MB)")

    print("\n[1] Nalagam release.json ...")
    with open(N.RELEASE_JSON, encoding="utf-8") as f:
        podatki = json.load(f)
    napake = list(podatki.keys())
    print(f"    napak: {len(napake)}")

    print("[2] Gradim tabelo (vrstica = mesto v programu) ...")
    vrstice = []
    napacni_kljuci, nestevilske_vrstice, brez_lokacije = [], [], []
    vrednosti_faulty = set()
    mest_na_napako = {}
    pricakovani_kljuci = set(N.TEHNIKE) | {"faulty"}
    for fid in napake:
        projekt, stevilka = razcleni_fault_id(fid)
        mesta = podatki[fid]
        mest_na_napako[fid] = len(mesta)
        for mesto, ocene in mesta.items():
            if set(ocene.keys()) != pricakovani_kljuci:
                napacni_kljuci.append((fid, mesto, sorted(ocene.keys())))
            if ":" in mesto:
                razred, vrstica = mesto.rsplit(":", 1)
            else:
                razred, vrstica = mesto, ""
                brez_lokacije.append((fid, mesto))
            if not vrstica.isdigit():
                nestevilske_vrstice.append((fid, mesto))
            fv = ocene.get("faulty")
            vrednosti_faulty.add(fv)
            v = {"fault_id": fid, "project": projekt, "bug": stevilka,
                 "classname": razred, "line": vrstica, "stmt": mesto}
            for t in N.TEHNIKE:
                v[t] = ocene.get(t, np.nan)
            v["faulty"] = fv
            vrstice.append(v)
    stolpci = ["fault_id", "project", "bug", "classname", "line", "stmt"] + N.TEHNIKE + ["faulty"]
    df = pd.DataFrame(vrstice, columns=stolpci)
    print(f"    tabela: {len(df):,} vrstic x {df.shape[1]} stolpcev")

    # preverbe oblike podatkov. Števili 357 in 9 veljata za to različico zbirke CombineFL
    print("\n[3] Preverbe:")
    p = Preverbe()
    skupaj_mest = sum(mest_na_napako.values())
    p("število vrstic je enako številu mest v viru (nič izgubljenega)", len(df) == skupaj_mest,
      f"{len(df)} / {skupaj_mest}")
    p("število napak je 357", len(napake) == sum(N.PROJEKTI_COMBINEFL.values()), f"{len(napake)}")
    po_projektih = dict(Counter(razcleni_fault_id(fid)[0] for fid in napake))
    p("projekti in števila napak se ujemajo s pričakovano sestavo", po_projektih == N.PROJEKTI_COMBINEFL,
      f"{po_projektih}")
    p("vsako mesto ima natanko 9 tehnik in oznako faulty", len(napacni_kljuci) == 0,
      f"odstopanj: {len(napacni_kljuci)}")
    p("faulty je 0 ali 1", vrednosti_faulty.issubset({0, 1}), f"{sorted(v for v in vrednosti_faulty if v is not None)}")
    nan_teh = int(df[N.TEHNIKE].isna().sum().sum())
    nan_f = int(df["faulty"].isna().sum())
    p("ni manjkajočih vrednosti", nan_teh == 0 and nan_f == 0, f"NaN v tehnikah {nan_teh}, v faulty {nan_f}")
    p("vsa mesta imajo številsko vrstico", len(nestevilske_vrstice) == 0, f"odstopanj: {len(nestevilske_vrstice)}")
    brez_dstar = [t for t in N.TEHNIKE if t != "dstar"]
    p("vse ocene so nenegativne", float(df[N.TEHNIKE].min().min()) >= 0.0)
    p("osem tehnik (brez DStar) je v [0, 1]",
      float(df[brez_dstar].min().min()) >= 0.0 and float(df[brez_dstar].max().max()) <= 1.0)

    # napake brez oznake faulty na nivoju stavka (popravek doda novo kodo, ki je v matriki ni)
    faulty_po_napaki = df[df["faulty"] == 1].groupby("fault_id").size()
    brez_oznake = sorted(set(napake) - set(faulty_po_napaki.index))
    p("natanko 9 napak nima nobene oznake faulty na nivoju stavka", len(brez_oznake) == 9, f"{len(brez_oznake)}")
    with open(N.pot(N.NAPAKE_BREZ_OZNAK), "w", encoding="utf-8") as f:
        f.write("\n".join(brez_oznake) + "\n")
    print(f"    napake brez oznake: {brez_oznake}")

    print("\n[4] Opisna statistika:")
    spf = pd.Series(mest_na_napako)
    n_faulty = int((df["faulty"] == 1).sum())
    print(f"    mest skupaj: {len(df):,}, od tega krivih (faulty = 1): {n_faulty:,} ({100 * n_faulty / len(df):.3f} %)")
    print(f"    mest na napako: min {spf.min()}, mediana {int(spf.median())}, max {spf.max()}, povprečje {spf.mean():.1f}")
    print(f"    krivih mest na napako: min {int(faulty_po_napaki.min())}, mediana {int(faulty_po_napaki.median())}, "
          f"max {int(faulty_po_napaki.max())}")
    print("    po projektih:")
    for pr in N.PROJEKTI_COMBINEFL:
        sub = df[df["project"] == pr]
        print(f"      {pr:8s} napak {sub['fault_id'].nunique():3d}, mest {len(sub):8,d}, krivih {int((sub['faulty'] == 1).sum()):5d}")
    print("    razpon ocen po tehnikah (DStar je po formuli navzgor neomejen, zato ga modeli normirajo):")
    for t in N.TEHNIKE:
        s = df[t]
        print(f"      {t:22s} min {s.min():.4g}  max {s.max():.4g}  neničelnih {int((s > 0).sum()):,}")

    print("\n[5] Shranjujem ...")
    pot_parquet = N.pot(N.MATRIKA_STAVKOV)
    df.to_parquet(pot_parquet, index=False)
    pot_vzorec = N.pot(N.MATRIKA_STAVKOV_VZOREC)
    df.head(200).to_csv(pot_vzorec, index=False)
    for pot in (pot_parquet, pot_vzorec):
        print(f"    {pot} ({os.path.getsize(pot) / 2**20:.2f} MB)")
    p("matrika je zapisana", os.path.isfile(pot_parquet))
    p.zakljuci()


if __name__ == "__main__":
    main()
