#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prvi naučeni model, MLP nad devetimi klasičnimi ocenami (H9).

Preveri, ali naučena kombinacija devetih tehnik rangira bolje kot Ochiai. Model, normalizacija in
delitev v gube so enaki kot v glavnem poskusu (mlp_llm.py), le število epoh je tu fiksno (60).

Vhod:   izhod/podatki/matrika_metod.parquet
Izhod:  izhod/modeli/mlp_h9_napovedi.csv, modeli/mlp_h9.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # koren repozitorija

import numpy as np

import nastavitve as N
from skupno.metrike import metrike_po_napakah, top1_po_konvencijah, wilcoxon_parni
from skupno.mlp import napovedi_cv
from skupno.podatki import nalozi_matriko_metod, seznam_napak, normiraj_po_napaki, gube_po_napaki, v_intervalu_0_1
from skupno.preverbe import Preverbe, glava

EPOHE = 60


def main():
    glava("MLP nad devetimi klasičnimi ocenami (H9)")
    df = nalozi_matriko_metod(N.pot(N.MATRIKA_METOD))
    napake = seznam_napak(df)
    print(f"Matrika: {len(df):,} metod, {len(napake)} napak, {int(df['faulty'].sum())} krivih metod")

    p = Preverbe()
    dfn = normiraj_po_napaki(df, N.TEHNIKE)
    p("po normalizaciji so vse značilke v [0, 1]", v_intervalu_0_1(dfn, N.TEHNIKE))
    gube = gube_po_napaki(napake)
    p("gube pokrijejo vsako napako natanko enkrat",
      sum(len(g) for g in gube) == len(napake) and set().union(*gube) == set(napake))

    print(f"\n[Učenje: {N.K_GUB}-kratno prečno preverjanje po napakah, {EPOHE} epoh]")
    df["mlp"] = napovedi_cv(dfn, df, N.TEHNIKE, gube, df["fault_id"].to_numpy(), N.SEME_DELITVE, EPOHE)
    p("vsaka metoda ima napoved", not np.isnan(df["mlp"]).any())

    print(f"\n[Rezultati, {len(napake)} napak]")
    o_top, o_mfr, o_mar, o_rang = metrike_po_napakah(df, "ochiai")
    m_top, m_mfr, m_mar, m_rang = metrike_po_napakah(df, "mlp")
    print(f"  {'model':8s} " + "  ".join(f"Top-{n}" for n in N.NS) + "     MFR     MAR")
    print(f"  {'Ochiai':8s} " + "  ".join(f"{o_top[n]:5d}" for n in N.NS) + f"   {o_mfr:6.2f}  {o_mar:6.2f}")
    print(f"  {'MLP':8s} " + "  ".join(f"{m_top[n]:5d}" for n in N.NS) + f"   {m_mfr:6.2f}  {m_mar:6.2f}")
    w = wilcoxon_parni(m_rang, o_rang, napake, "less")
    print(f"  Wilcoxon (MLP boljši od Ochiai): W = {w['W']:.0f}, p = {w['p']:.3g} "
          f"(MLP boljši {w['a_better']}, slabši {w['a_worse']}, enako {w['tie']})")
    print(f"  Top-1 pri optimistični obravnavi enakih ocen: Ochiai {top1_po_konvencijah(df, 'ochiai')['min']}, "
          f"MLP {top1_po_konvencijah(df, 'mlp')['min']} (pri povprečnem rangu {o_top[1]} in {m_top[1]})")

    df[N.KLJUC + ["faulty", "ochiai", "mlp"]].to_csv(N.pot(N.MLP_H9_NAPOVEDI), index=False)
    with open(N.pot(N.MLP_H9_TXT), "w", encoding="utf-8") as f:
        f.write(f"MLP (9 klasičnih ocen) proti Ochiai, nivo metod, {len(napake)} napak, "
                f"{N.K_GUB}-kratno prečno preverjanje po napakah, {EPOHE} epoh\n")
        f.write("model     " + "  ".join(f"Top-{n}" for n in N.NS) + "   MFR    MAR\n")
        f.write("Ochiai    " + "  ".join(f"{o_top[n]:5d}" for n in N.NS) + f"   {o_mfr:.2f}  {o_mar:.2f}\n")
        f.write("MLP       " + "  ".join(f"{m_top[n]:5d}" for n in N.NS) + f"   {m_mfr:.2f}  {m_mar:.2f}\n")
        f.write(f"Wilcoxon p = {w['p']:.3g}\n")
    print(f"\n[Shranjeno] {N.MLP_H9_NAPOVEDI}, {N.MLP_H9_TXT}")
    p.zakljuci()


if __name__ == "__main__":
    main()
