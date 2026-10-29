#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Izhodišče Ochiai in DStar na nivoju metod.

Ponovi izračun iz ochiai_stavki.py na matriki metod in ga primerja z nivojem stavkov. Pričakujemo ožji
razpon med optimistično in pesimistično obravnavo enakih ocen in višji Top-N.

Zagon:  python izhodisca/ochiai_metode.py
        (4. skripta v cevovodu, vrstni red je v README)
Vhod:   izhod/podatki/matrika_metod.parquet, izhodisca/ochiai_stavki_rangi.csv (za primerjavo)
Izhod:  izhod/izhodisca/ochiai_metode_rangi.csv, izhodisca/ochiai_metode.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # koren repozitorija (nastavitve.py, skupno/)

import os

import numpy as np
import pandas as pd

import nastavitve as N
from skupno.metrike import meji_ranga
from skupno.podatki import nalozi_matriko_metod
from skupno.preverbe import Preverbe, glava

TEHNIKI = ["ochiai", "dstar"]


def main():
    glava("Ochiai in DStar na nivoju metod")
    df = nalozi_matriko_metod(N.pot(N.MATRIKA_METOD))
    print(f"Matrika metod: {len(df):,} metod, {df['fault_id'].nunique()} napak")
    oznak = df.groupby("fault_id")["faulty"].sum()
    ocenljive = sorted(oznak[oznak > 0].index)
    print(f"Ocenljivih napak (vsaj ena kriva metoda): {len(ocenljive)}, brez krive metode: {int((oznak == 0).sum())}")

    skupine = {fid: g for fid, g in df.groupby("fault_id")}
    rezultati, po_napakah = {}, []
    for t in TEHNIKI:
        spodnje, zgornje = {}, {}
        for fid in ocenljive:
            g = skupine[fid]
            s, z = meji_ranga(g[t].to_numpy(float), g["faulty"].to_numpy() == 1)
            spodnje[fid], zgornje[fid] = s, z
            if t == "ochiai":
                po_napakah.append({"fault_id": fid, "n_methods": len(g), "n_faulty": int(g["faulty"].sum()),
                                   "ochiai_best": s, "ochiai_worst": z})
        rezultati[t] = {n: (sum(spodnje[f] <= n for f in ocenljive), sum(zgornje[f] <= n for f in ocenljive))
                        for n in N.NS}

    print(f"\n[Top-N na nivoju metod, od {len(ocenljive)} napak, optimistična..pesimistična meja]")
    for t in TEHNIKI:
        print(f"  {t:8s} " + "   ".join(f"Top-{n}: {rezultati[t][n][0]}..{rezultati[t][n][1]}" for n in N.NS))

    # primerjava z nivojem stavkov
    stavki = None
    pot_stavki = N.pot(N.OCHIAI_STAVKI_RANGI)
    if os.path.isfile(pot_stavki):
        s = pd.read_csv(pot_stavki)
        s = s[s["tehnika"] == "ochiai"]
        stavki = {"n": s["fault_id"].nunique(),
                  "top": {n: (int((s["best_rank"] <= n).sum()), int((s["worst_rank"] <= n).sum())) for n in N.NS},
                  "razpon": float((s["worst_rank"] - s["best_rank"]).mean())}
        print(f"\n[Ochiai: stavki ({stavki['n']} napak) proti metodam]")
        for n in N.NS:
            print(f"  Top-{n:<2d} stavki {stavki['top'][n][0]}..{stavki['top'][n][1]}   metode "
                  f"{rezultati['ochiai'][n][0]}..{rezultati['ochiai'][n][1]}")

    print("\n[Preverbe]")
    p = Preverbe()
    p("spodnja meja ranga <= zgornja meja za vse napake",
      all(v["ochiai_best"] <= v["ochiai_worst"] for v in po_napakah))
    p("Top-N narašča z n", all(rezultati[t][N.NS[i]][k] <= rezultati[t][N.NS[i + 1]][k]
                                for t in TEHNIKI for k in (0, 1) for i in range(len(N.NS) - 1)))
    p("za vsak n velja 0 <= pesimistično <= optimistično <= število napak",
      all(0 <= rezultati[t][n][1] <= rezultati[t][n][0] <= len(ocenljive) for t in TEHNIKI for n in N.NS))
    o10, d10 = rezultati["ochiai"][10][0], rezultati["dstar"][10][0]
    p("Ochiai in DStar sta pri Top-10 blizu", abs(o10 - d10) <= max(20, int(0.25 * max(o10, d10))),
      f"ochiai {o10}, dstar {d10}")
    razpon_metode = float(np.mean([v["ochiai_worst"] - v["ochiai_best"] for v in po_napakah]))
    print(f"  povprečni razpon ranga na napako (pesimistična minus optimistična meja), Ochiai: metode {razpon_metode:.2f}"
          + (f", stavki {stavki['razpon']:.2f}" if stavki else ""))
    if stavki:
        p("delež Top-1 (optimistično) je na nivoju metod vsaj tolikšen kot na nivoju stavkov",
          rezultati["ochiai"][1][0] / len(ocenljive) >= stavki["top"][1][0] / stavki["n"])

    pd.DataFrame(po_napakah).to_csv(N.pot(N.OCHIAI_METODE_RANGI), index=False)
    with open(N.pot(N.OCHIAI_METODE_TXT), "w", encoding="utf-8") as f:
        f.write(f"Ochiai in DStar na nivoju metod, {len(ocenljive)} ocenljivih napak\n")
        for t in TEHNIKI:
            f.write(f"{t}: " + "  ".join(f"Top-{n} {rezultati[t][n][0]}..{rezultati[t][n][1]}" for n in N.NS) + "\n")
    print(f"\n[Shranjeno] {N.OCHIAI_METODE_RANGI}, {N.OCHIAI_METODE_TXT}")
    p.zakljuci()


if __name__ == "__main__":
    main()
