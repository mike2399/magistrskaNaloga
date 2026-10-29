#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Slike 5.2, 6.1 in 6.2 naloge, izrisane neposredno iz shranjenih rezultatov.

Slika 5.2: validacijske krivulje izbire števila epoh (mlp_llm.py).
Slika 6.1: stroga uporabnost devetih tehnik (primerjava_druzin.py).
Slika 6.2: porazdelitev ranga prve krive metode za MLP H9 in MLP H9+LLM (mlp_llm.py).
Vsaka vrednost na sliki je preverjena proti datotekam JSON, ob neskladju se skripta ustavi.
Širina slik je enaka stavčnemu pasu predloge FRI (137 mm), zato jih LaTeX vstavi brez skaliranja.

Zagon:  python porocila/slike.py
        (16. skripta v cevovodu, vrstni red je v README)
Vhod:   izhod/modeli/mlp_llm_rezultati.json, modeli/mlp_llm_izbrane_epohe.json, modeli/mlp_llm_napovedi.csv, analize/primerjava_druzin.json
Izhod:  izhod/porocila/slike/slika_*.pdf in .png, izhod/porocila/podatki_slik.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # koren repozitorija (nastavitve.py, skupno/)

import json
import math
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager, ticker  # noqa: E402

import nastavitve as N  # noqa: E402
from skupno.metrike import prvi_rangi  # noqa: E402

SIRINA_IN = 5.4          # 137 mm = širina stavčnega pasu predloge
DPI_PREDOGLED = 200
MAPA = N.pot(N.SLIKE_MAPA)

# Predloga uporablja Computer Modern. Če ni nameščena, vzamemo najbližjo pisavo s šumniki
PISAVE = ["CMU Serif", "Latin Modern Roman", "STIXGeneral", "Times New Roman", "DejaVu Serif"]
_na_voljo = {f.name for f in font_manager.fontManager.ttflist}
PISAVA = next(p for p in PISAVE if p in _na_voljo)
plt.rcParams.update({
    "font.family": "serif", "font.serif": [PISAVA], "font.size": 9,
    "axes.labelsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "mathtext.fontset": "stix" if PISAVA in ("STIXGeneral", "Times New Roman") else "cm",
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 3, "ytick.major.size": 3, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "hatch.linewidth": 0.5, "savefig.bbox": "standard",
    "figure.constrained_layout.h_pad": 0.02, "figure.constrained_layout.w_pad": 0.02, "legend.frameon": False,
})
MREZA = dict(color="#e3e3e3", linewidth=0.5, linestyle="-")
PREVERBE = []


def preveri(ime, pogoj, podrobnost=""):
    PREVERBE.append({"preverba": ime, "izid": "OK" if pogoj else "NAPAKA", "podrobnost": podrobnost})
    print(f"  [{'OK' if pogoj else 'NAPAKA'}] {ime}" + (f" ({podrobnost})" if podrobnost else ""))
    if not pogoj:
        raise SystemExit(f"PREVERBA NI USPELA: {ime} ({podrobnost})")


def dec(x, nd=2):
    return f"{x:.{nd}f}".replace(".", ",")


def os_oznaka(x, pos=None):
    if abs(x - round(x)) < 1e-9:
        return f"{int(round(x)):,}".replace(",", ".")
    return f"{x:.1f}".replace(".", ",")


def beri(ime):
    with open(N.pot(ime), encoding="utf-8") as f:
        return json.load(f)


def shrani(fig, ime):
    fig.savefig(os.path.join(MAPA, f"{ime}.pdf"))
    fig.savefig(os.path.join(MAPA, f"{ime}.png"), dpi=DPI_PREDOGLED)
    plt.close(fig)
    print(f"  zapisano: {ime}.pdf, {ime}.png")


def main():
    os.makedirs(MAPA, exist_ok=True)
    podatki = {"pisava": PISAVA, "slike": {}}
    print(f"Pisava: {PISAVA}")

    # ---------------------------------------------------------------- slika 5.2
    print("\n[Slika 5.2: validacijske krivulje]")
    c08 = beri(N.MLP_LLM_REZULTATI)
    epohe_json = beri(N.MLP_LLM_EPOHE)
    krivulje, izbrane = c08["val_curves"], c08["chosen_epochs"]
    EPOHE = [60, 150, 300, 600]
    NIZI = [("MLP H9", "H9", "-", "o", (7, 0), ("left", "center")),
            ("MLP H9+LLM", "H9+LLM", (0, (5, 2.5)), "s", (0, -8), ("center", "top")),
            ("MLP H9+premešan", "H9 s premešanimi ocenami LLM", (0, (1, 1.6)), "^", (7, 0), ("left", "center"))]
    preveri("rezultati in datoteka izbranih epoh imata enake krivulje in izbire",
            krivulje == epohe_json["val_curves"] and izbrane == epohe_json["chosen"])
    for kljuc, *_ in NIZI:
        preveri(f"{kljuc}: krivulja ima natanko epohe {EPOHE}", sorted(int(e) for e in krivulje[kljuc]) == EPOHE)
        najm = min(EPOHE, key=lambda e: krivulje[kljuc][str(e)])
        preveri(f"{kljuc}: izbrana epoha {izbrane[kljuc]} je najmanjši validacijski MFR", najm == izbrane[kljuc])
    fig, ax = plt.subplots(figsize=(SIRINA_IN, 2.45), layout="constrained")
    for kljuc, oznaka, crta, znak, odmik, (ha, va) in NIZI:
        y = [krivulje[kljuc][str(e)] for e in EPOHE]
        ax.plot(EPOHE, y, linestyle=crta, color="black", linewidth=1.0, marker=znak, markersize=5,
                markerfacecolor="white", markeredgecolor="black", markeredgewidth=0.9, label=oznaka, zorder=3)
        e = izbrane[kljuc]
        yi = krivulje[kljuc][str(e)]
        ax.plot([e], [yi], linestyle="none", marker=znak, markersize=5.6, markerfacecolor="black", markeredgecolor="black", zorder=4)
        ax.annotate(dec(yi), xy=(e, yi), xytext=odmik, textcoords="offset points", ha=ha, va=va)
        podatki["slike"].setdefault("slika_5_2_val_krivulje", {})[oznaka] = {
            "mfr_po_epohah": {str(ep): krivulje[kljuc][str(ep)] for ep in EPOHE}, "izbrana_epoha": e, "mfr_izbrane": yi}
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(ticker.FixedLocator(EPOHE))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter([str(e) for e in EPOHE]))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlim(48, 820)
    ax.set_ylim(10.0, 16.9)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(os_oznaka))
    ax.set_xlabel("število epoh")
    ax.set_ylabel("MFR na notranji validaciji")
    ax.grid(axis="y", **MREZA)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, handlelength=3.0, columnspacing=1.6, borderaxespad=0.3)
    shrani(fig, "slika_5_2_val_krivulje")

    # ---------------------------------------------------------------- slika 6.1
    print("\n[Slika 6.1: stroga uporabnost tehnik]")
    t14 = beri(N.DRUZINE_REZULTATI)
    tk = t14["techniques"]
    IME = {"dstar": "DStar", "ochiai": "Ochiai", "metallaxis": "Metallaxis", "muse": "MUSE", "stacktrace": "sled sklada",
           "slicing_count": "slicing_count (frekvenca)", "slicing_intersection": "slicing_intersection (presek)",
           "slicing": "slicing (unija)", "predicateswitching": "menjava predikatov"}
    n_napak = c08["n_faults"]
    preveri("v rezultatih primerjava_druzin.py je natanko devet tehnik z znanimi imeni", sorted(tk) == sorted(IME))
    for k in tk:
        a = tk[k]["applicable_strict"]
        preveri(f"{IME[k]}: stroga uporabnost je celo število med 0 in {n_napak}", isinstance(a, int) and 0 <= a <= n_napak, str(a))
    red = sorted(tk, key=lambda k: (-tk[k]["applicable_strict"], -tk[k]["all"]["top"]["1"]))
    vred = [tk[k]["applicable_strict"] for k in red]
    fig, ax = plt.subplots(figsize=(SIRINA_IN, 2.75), layout="constrained")
    yy = np.arange(len(red))
    ax.barh(yy, vred, height=0.62, color="#555555", edgecolor="none", zorder=2)
    ax.set_yticks(yy)
    ax.set_yticklabels([IME[k] for k in red])
    ax.invert_yaxis()
    for y, v in zip(yy, vred):
        ax.annotate(str(v), xy=(v, y), xytext=(-3, 0), textcoords="offset points", ha="right", va="center", color="white")
    ax.axvline(n_napak, color="black", linewidth=0.8, linestyle=(0, (4, 2.5)), zorder=3)
    ax.annotate(f"vse napake ({n_napak})", xy=(n_napak, (len(red) - 1) / 2), xytext=(3, 0), textcoords="offset points",
                rotation=90, ha="left", va="center")
    ax.set_xlim(0, 372)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(os_oznaka))
    ax.set_xlabel("stroga uporabnost (število napak)")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", **MREZA)
    ax.set_axisbelow(True)
    podatki["slike"]["slika_6_1_uporabnost"] = {IME[k]: tk[k]["applicable_strict"] for k in red}
    podatki["slike"]["slika_6_1_uporabnost"]["_vse_napake"] = n_napak
    shrani(fig, "slika_6_1_uporabnost")

    # ---------------------------------------------------------------- slika 6.2
    print("\n[Slika 6.2: porazdelitev ranga prve krive metode]")
    d = pd.read_csv(N.pot(N.MLP_LLM_NAPOVEDI), usecols=["fault_id", "faulty", "mlp_h9", "mlp_h9llm"])
    # razred 1 = rang natanko 1, ostali razredi so polodprti intervali (rang 1,5 pade v (1, 3])
    RAZREDI = [("1", 0, 1), ("(1, 3]", 1, 3), ("(3, 5]", 3, 5), ("(5, 10]", 5, 10), ("(10, 50]", 10, 50),
               ("(50, $\\infty$)", 50, math.inf)]
    MODELA = [("mlp_h9", "MLP H9 (kontrola)"), ("mlp_h9llm", "MLP H9+LLM")]
    stetja, rangi = {}, {}
    for stolpec, kljuc in MODELA:
        r = pd.Series(prvi_rangi(d, stolpec))
        rangi[stolpec] = r
        st = [int(((r > lo) & (r <= hi)).sum()) for _, lo, hi in RAZREDI]
        stetja[stolpec] = st
        ref = c08["all"][kljuc]
        preveri(f"{kljuc}: število napak je {n_napak}", len(r) == n_napak and sum(st) == n_napak)
        preveri(f"{kljuc}: razred 1 je enak Top-1 iz rezultatov", st[0] == ref["top"]["1"], str(st[0]))
        preveri(f"{kljuc}: kumulativni razredi so enaki Top-3, Top-5 in Top-10",
                [sum(st[:2]), sum(st[:3]), sum(st[:4])] == [ref["top"]["3"], ref["top"]["5"], ref["top"]["10"]])
        preveri(f"{kljuc}: MFR iz istih rangov je enak MFR iz rezultatov", abs(r.mean() - ref["mfr"]) < 1e-9, dec(r.mean()))
    med_1_in_2 = {s: int(((rangi[s] > 1) & (rangi[s] < 2)).sum()) for s, _ in MODELA}
    fig, ax = plt.subplots(figsize=(SIRINA_IN, 2.5), layout="constrained")
    x = np.arange(len(RAZREDI))
    w = 0.38
    st1, st2 = stetja["mlp_h9"], stetja["mlp_h9llm"]
    s1 = ax.bar(x - w / 2 - 0.01, st1, w, facecolor="white", edgecolor="#1a1a1a", hatch="////", linewidth=0.6, label="MLP H9 (kontrola)", zorder=2)
    s2 = ax.bar(x + w / 2 + 0.01, st2, w, facecolor="#555555", edgecolor="#555555", linewidth=0.6, label="MLP H9+LLM", zorder=2)
    for stolpci, vred in ((s1, st1), (s2, st2)):
        for stolpec, v in zip(stolpci, vred):
            ax.annotate(str(v), xy=(stolpec.get_x() + stolpec.get_width() / 2, v), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([o for o, _, _ in RAZREDI])
    ax.set_xlabel("rang prve krive metode")
    ax.set_ylabel("število napak")
    ax.set_ylim(0, 285)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(os_oznaka))
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", **MREZA)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    podatki["slike"]["slika_6_2_porazdelitev_rangov"] = {
        "razredi": ["1", "(1, 3]", "(3, 5]", "(5, 10]", "(10, 50]", "(50, neskončno)"],
        "meje_razredov": "razred 1 je rang natanko 1, ostali razredi so polodprti intervali",
        "MLP H9 (kontrola)": st1, "MLP H9+LLM": st2,
        "prvi_rang_strogo_med_1_in_2": {"MLP H9 (kontrola)": med_1_in_2["mlp_h9"], "MLP H9+LLM": med_1_in_2["mlp_h9llm"]}}
    shrani(fig, "slika_6_2_porazdelitev_rangov")

    podatki["preverbe"] = PREVERBE
    with open(N.pot(N.PODATKI_SLIK), "w", encoding="utf-8") as f:
        json.dump(podatki, f, ensure_ascii=False, indent=2)
    print(f"\nVse preverbe ({len(PREVERBE)}) so uspele, podatki slik so v {N.PODATKI_SLIK}.")


if __name__ == "__main__":
    main()
