#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Izhodišče Ochiai in DStar na nivoju stavkov.

Za vsako napako uredi mesta po oceni in prešteje, pri koliko napakah je kriva vrstica med prvimi n (Top-N).
Ker je na nivoju stavkov veliko enakih ocen, rang poroča kot interval med optimistično in pesimistično
obravnavo enakih ocen. Devet napak brez oznake faulty na nivoju stavka je izločenih, ostane 348 napak.

Zagon:  python izhodisca/ochiai_stavki.py
        (2. skripta v cevovodu, vrstni red je v README)
Vhod:   izhod/podatki/matrika_stavkov.parquet, podatki/napake_brez_oznak_stavkov.txt
Izhod:  izhod/izhodisca/ochiai_stavki_rangi.csv (meji ranga za vsako napako), izhodisca/ochiai_stavki.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # koren repozitorija (nastavitve.py, skupno/)

import os

import pandas as pd

import nastavitve as N
from skupno.metrike import meji_ranga
from skupno.preverbe import Preverbe, glava

TEHNIKI = ["ochiai", "dstar"]


def main():
    glava("Ochiai in DStar na nivoju stavkov")
    df = pd.read_parquet(N.pot(N.MATRIKA_STAVKOV))
    print(f"Matrika: {len(df):,} mest x {df.shape[1]} stolpcev")

    oznak = df.groupby("fault_id")["faulty"].sum()
    brez_oznake = sorted(oznak[oznak == 0].index)
    ocenljive = sorted(oznak[oznak > 0].index)
    print(f"Napak: {len(oznak)}, brez oznake: {len(brez_oznake)}, ocenljivih: {len(ocenljive)}")

    p = Preverbe()
    pot_brez = N.pot(N.NAPAKE_BREZ_OZNAK)
    iz_datoteke = []
    if os.path.isfile(pot_brez):
        with open(pot_brez, encoding="utf-8") as f:
            iz_datoteke = sorted(x.strip() for x in f if x.strip())
    print("\n[Preverbe vhoda]")
    p("ocenljivih napak je 348", len(ocenljive) == 348, f"{len(ocenljive)}")
    p("seznam napak brez oznake se ujema s skripto matrika_stavkov.py", brez_oznake == iz_datoteke)

    skupine = {fid: g for fid, g in df.groupby("fault_id")}
    rezultati, vrstice = {}, []
    for t in TEHNIKI:
        spodnje, zgornje = {}, {}
        for fid in ocenljive:
            g = skupine[fid]
            s, z = meji_ranga(g[t].to_numpy(dtype=float), g["faulty"].to_numpy() == 1)
            spodnje[fid], zgornje[fid] = s, z
            vrstice.append({"fault_id": fid, "tehnika": t, "best_rank": s, "worst_rank": z})
        rezultati[t] = {n: (sum(1 for f in ocenljive if spodnje[f] <= n),
                            sum(1 for f in ocenljive if zgornje[f] <= n)) for n in N.NS}

    print(f"\n[Top-N na nivoju stavkov, od {len(ocenljive)} napak, optimistična..pesimistična meja]")
    for t in TEHNIKI:
        print(f"  {t:8s} " + "   ".join(f"Top-{n}: {rezultati[t][n][0]}..{rezultati[t][n][1]}" for n in N.NS))

    print("\n[Preverbe rezultatov]")
    p("spodnja meja ranga <= zgornja meja za vse napake",
      all(v["best_rank"] <= v["worst_rank"] for v in vrstice))
    p("Top-N narašča z n", all(rezultati[t][N.NS[i]][k] <= rezultati[t][N.NS[i + 1]][k]
                                for t in TEHNIKI for k in (0, 1) for i in range(len(N.NS) - 1)))
    p("za vsak n velja 0 <= pesimistično <= optimistično <= število napak",
      all(0 <= rezultati[t][n][1] <= rezultati[t][n][0] <= len(ocenljive) for t in TEHNIKI for n in N.NS))
    o10, d10 = rezultati["ochiai"][10][0], rezultati["dstar"][10][0]
    p("Ochiai in DStar sta pri Top-10 blizu (tehniki sta močno korelirani)",
      abs(o10 - d10) <= max(20, int(0.25 * max(o10, d10))), f"ochiai {o10}, dstar {d10}")

    pot_rangi = N.pot(N.OCHIAI_STAVKI_RANGI)
    pd.DataFrame(vrstice).to_csv(pot_rangi, index=False)
    pot_txt = N.pot(N.OCHIAI_STAVKI_TXT)
    with open(pot_txt, "w", encoding="utf-8") as f:
        f.write(f"Ochiai in DStar na nivoju stavkov, {len(ocenljive)} ocenljivih napak\n")
        f.write("Top-N kot optimistična..pesimistična meja (obravnava enakih ocen)\n\n")
        for t in TEHNIKI:
            f.write(f"{t:8s} " + "  ".join(f"Top-{n} {rezultati[t][n][0]}..{rezultati[t][n][1]}" for n in N.NS) + "\n")
    print(f"\n[Shranjeno] {pot_rangi}\n            {pot_txt}")
    p.zakljuci()


if __name__ == "__main__":
    main()
