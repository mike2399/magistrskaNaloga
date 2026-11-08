# -*- coding: utf-8 -*-
"""Metrike lokalizacije napak in statistični testi.

Merilo je povsod enako. Metode ene napake uredimo padajoče po oceni, enake ocene dobijo povprečni
rang, rang napake je rang njene prve krive metode. Iz teh rangov izhajajo Top-N, MFR in MAR
"""
import math

import numpy as np
from scipy.stats import rankdata, wilcoxon

from nastavitve import NS


def rangi_krivih(g, stolpec, metoda="average"):
    """Rangi krivih metod ene napake, enake ocene po pravilu metoda"""
    r = rankdata(-g[stolpec].to_numpy(dtype=float), method=metoda)
    return r[g["faulty"].to_numpy() == 1]


def metrike_po_napakah(df, stolpec, podmnozica=None):
    """Vrne (top, mfr, mar, prvi_rang) iz ocen v stolpcu, po želji le za napake v podmnozica.

    top je {n: število napak v Top-n}, prvi_rang je {fault_id: rang prve krive metode}
    """
    top = {n: 0 for n in NS}
    prvi, povprecni, prvi_po_napaki = [], [], {}
    for fid, g in df.groupby("fault_id"):
        if podmnozica is not None and fid not in podmnozica:
            continue
        krivi = rangi_krivih(g, stolpec)
        prvi_rang = float(krivi.min())
        prvi.append(prvi_rang)
        povprecni.append(float(krivi.mean()))
        prvi_po_napaki[fid] = prvi_rang
        for n in NS:
            if prvi_rang <= n:
                top[n] += 1
    return top, float(np.mean(prvi)), float(np.mean(povprecni)), prvi_po_napaki


def metrike_iz_napovedi(df, napovedi, podmnozica=None):
    """Kot metrike_po_napakah, a za napovedi v polju, poravnanem z vrsticami df"""
    tmp = df[["fault_id", "faulty"]].copy()
    tmp["_p"] = napovedi
    return metrike_po_napakah(tmp, "_p", podmnozica)


def prvi_rangi(df, stolpec):
    """Slovar {fault_id: rang prve krive metode} pri povprečnem rangu enakih ocen"""
    return {fid: float(rangi_krivih(g, stolpec).min()) for fid, g in df.groupby("fault_id")}


def top1_po_konvencijah(df, stolpec):
    """Top-1 pri optimistični (min), povprečni (avg) in pesimistični (max) obravnavi enakih ocen"""
    out = {}
    for metoda, oznaka in (("min", "min"), ("average", "avg"), ("max", "max")):
        out[oznaka] = sum(1 for _, g in df.groupby("fault_id")
                          if rangi_krivih(g, stolpec, metoda).min() <= 1.0)
    return out


def meji_ranga(ocene, krivo):
    """Spodnja in zgornja meja ranga prvega krivega elementa pri enakih ocenah.

    Spodnja šteje elemente z višjo oceno od najboljšega krivega, zgornja še nekrive z enako oceno
    """
    s_max = ocene[krivo].max()
    visjih = int(np.sum(ocene > s_max))
    enakih = int(np.sum(ocene == s_max))
    krivih_enakih = int(np.sum((ocene == s_max) & krivo))
    return visjih + 1, visjih + (enakih - krivih_enakih) + 1


def wilcoxon_parni(rang_a, rang_b, napake, alternativa="less", metoda="auto"):
    """Parni Wilcoxonov test rangov modelov A in B, pri "less" preverjamo, ali ima A manjše range.

    Vrne W, p in število napak, kjer je A boljši, slabši ali enak. Če test ni mogoč, sta W in p nan
    """
    x = np.array([rang_a[f] for f in napake], dtype=float)
    y = np.array([rang_b[f] for f in napake], dtype=float)
    izid = {"W": float("nan"), "p": float("nan"), "alternative": alternativa,
            "a_better": int((x < y).sum()), "a_worse": int((x > y).sum()), "tie": int((x == y).sum())}
    if len(napake) >= 3 and not np.all(x == y):
        try:
            W, p = wilcoxon(x, y, zero_method="wilcox", alternative=alternativa, method=metoda)
            izid["W"], izid["p"] = float(W), float(p)
        except ValueError:
            pass
    return izid


def holm(pvals):
    """Holm-Bonferronijev popravek vrednosti p, vrne jih v istem vrstnem redu, nan ostane nan"""
    m = len(pvals)
    vrstni_red = np.argsort([2.0 if math.isnan(p) else p for p in pvals])      # nan na konec
    popravljeni = [float("nan")] * m
    tekoci_max = 0.0
    for k, idx in enumerate(vrstni_red):
        p = pvals[idx]
        if math.isnan(p):
            continue
        tekoci_max = max(tekoci_max, min(1.0, (m - k) * p))
        popravljeni[idx] = tekoci_max
    return popravljeni
