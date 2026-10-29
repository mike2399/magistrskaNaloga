# -*- coding: utf-8 -*-
"""Metrike lokalizacije napak in statistični testi.

Povsod velja isto merilo. Znotraj ene napake metode uredimo padajoče po oceni, enake ocene dobijo
povprečni rang, rang napake pa je rang njene najbolje uvrščene krive metode. Iz teh rangov izhajajo
Top-N, MFR in MAR.
"""
import numpy as np
from scipy.stats import rankdata, wilcoxon

from nastavitve import NS


def metrike_po_napakah(df, stolpec, podmnozica=None):
    """Top-N, MFR, MAR in rang prve krive metode za vsako napako iz ocen v stolpcu `stolpec`.

    Vrne (top, mfr, mar, prvi_rang). top je slovar {n: število napak s prvo krivo metodo med prvimi n},
    prvi_rang pa slovar {fault_id: rang prve krive metode}. S podmnozica se omejimo na del napak.
    """
    top = {n: 0 for n in NS}
    prvi, povprecni, prvi_po_napaki = [], [], {}
    for fid, g in df.groupby("fault_id"):
        if podmnozica is not None and fid not in podmnozica:
            continue
        r = rankdata(-g[stolpec].to_numpy(dtype=float), method="average")
        krivi = r[g["faulty"].to_numpy() == 1]
        prvi_rang, povpr_rang = float(krivi.min()), float(krivi.mean())
        prvi.append(prvi_rang)
        povprecni.append(povpr_rang)
        prvi_po_napaki[fid] = prvi_rang
        for n in NS:
            if prvi_rang <= n:
                top[n] += 1
    return top, float(np.mean(prvi)), float(np.mean(povprecni)), prvi_po_napaki


def prvi_rangi(df, stolpec):
    """Slovar {fault_id: rang prve krive metode} po pravilu povprečnega ranga."""
    out = {}
    for fid, g in df.groupby("fault_id"):
        r = rankdata(-g[stolpec].to_numpy(dtype=float), method="average")
        out[fid] = float(r[g["faulty"].to_numpy() == 1].min())
    return out


def top1_po_konvencijah(df, stolpec):
    """Top-1 pri treh obravnavah enakih ocen: optimistični (min), povprečni (avg) in pesimistični (max)."""
    out = {}
    for metoda, oznaka in (("min", "min"), ("average", "avg"), ("max", "max")):
        n = 0
        for fid, g in df.groupby("fault_id"):
            r = rankdata(-g[stolpec].to_numpy(dtype=float), method=metoda)
            if float(r[g["faulty"].to_numpy() == 1].min()) <= 1.0:
                n += 1
        out[oznaka] = n
    return out


def meji_ranga(ocene, krivo):
    """Spodnja in zgornja meja ranga prvega krivega elementa glede na obravnavo enakih ocen.

    Spodnja meja šteje le elemente z višjo oceno od najboljšega krivega, zgornja doda še nekrive
    elemente z enako oceno.
    """
    s_max = ocene[krivo].max()
    visjih = int(np.sum(ocene > s_max))
    enakih = int(np.sum(ocene == s_max))
    krivih_enakih = int(np.sum((ocene == s_max) & krivo))
    return visjih + 1, visjih + (enakih - krivih_enakih) + 1


def wilcoxon_parni(rang_a, rang_b, napake, alternativa="less", metoda="auto"):
    """Parni Wilcoxonov test na rangih prve krive metode modelov A in B.

    Pri alternativa="less" preverjamo, ali ima A manjše (boljše) range. Vrne statistiko W, vrednost p
    in število napak, pri katerih je A boljši, slabši ali enak. Če test ni izračunljiv, sta W in p nan.
    """
    x = np.array([rang_a[f] for f in napake], dtype=float)
    y = np.array([rang_b[f] for f in napake], dtype=float)
    izid = {"W": float("nan"), "p": float("nan"), "alternative": alternativa,
            "a_better": int((x < y).sum()), "a_worse": int((x > y).sum()), "tie": int((x == y).sum())}
    if len(napake) >= 3 and not bool(np.all(x == y)):
        try:
            W, p = wilcoxon(x, y, zero_method="wilcox", alternative=alternativa, method=metoda)
            izid["W"], izid["p"] = float(W), float(p)
        except ValueError:
            pass
    return izid


def holm(pvals):
    """Holm-Bonferronijev popravek za več primerjav. Vrne popravljene vrednosti p v istem vrstnem redu."""
    m = len(pvals)
    vrstni_red = np.argsort([p if p == p else 2.0 for p in pvals])
    popravljeni = [float("nan")] * m
    tekoci_max = 0.0
    for k, idx in enumerate(vrstni_red):
        p = pvals[idx]
        if p != p:                       # nan ostane nan
            continue
        tekoci_max = max(tekoci_max, min(1.0, (m - k) * p))
        popravljeni[idx] = tekoci_max
    return popravljeni
