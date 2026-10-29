#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pozivi za jezikovni model (brez klicev API, brez stroška).

Za vsako napako sestavi poziv z očiščeno sledjo sklada sprožilnih testov in izvorno kodo do 50 kandidatnih
metod. Kandidate izbere po vsoti devetih normiranih klasičnih ocen, v pozivu pa jih predstavi v naključnem
vrstnem redu (seme iz oznake napake). Ob pozivu shrani tabelo kandidatov, prek katere klici.py pripne ocene
k metodam.

Zagon:  python jezikovni_model/pozivi.py                    vse napake
        python jezikovni_model/pozivi.py lang1,closure1     naštete napake
        python jezikovni_model/pozivi.py --brisi-kopije     sproti briši delovne kopije (varčuje z diskom)
        (6. skripta v cevovodu, vrstni red je v README)
Vhod:   izhod/podatki/matrika_metod.parquet, Defects4J (delovne kopije in sledi sprožilnih testov)
Izhod:  izhod/jezikovni_model/pozivi/poziv_<napaka>.txt, jezikovni_model/pozivi/kandidati_<napaka>.csv,
        jezikovni_model/manifest_pozivov.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # koren repozitorija (nastavitve.py, skupno/)

import argparse
import hashlib
import os
import random
import shutil

import numpy as np
import pandas as pd

import nastavitve as N
from skupno.defects4j import preveri_defects4j, checkout, export, mapa_kopije, pot_razreda, sled_sklada, izsek_kode
from skupno.podatki import ime_d4j
from skupno.preverbe import glava

STOLPCI_MANIFESTA = ["fault_id", "project", "bug", "n_cand", "n_faulty_in_cand",
                     "faulty_indices", "n_chars", "tok_lo", "tok_hi", "status"]


def skupna_ocena(g):
    """Vsota devetih znotraj napake min-max normiranih ocen, konstantna tehnika prispeva 0."""
    F = g[N.TEHNIKE].to_numpy(dtype=float)
    mn, mx = F.min(axis=0), F.max(axis=0)
    razpon = mx - mn
    razpon[razpon == 0] = 1.0
    return ((F - mn) / razpon).sum(axis=1)


def sestavi_poziv(sled, metode):
    """Besedilo poziva (angleško, ker sta v tem jeziku koda in sled sklada)."""
    uvod = ("You are an expert Java debugging assistant performing fault localization.\n"
            "One or more failing tests revealed a bug. Below is their error output and stack trace(s), "
            "followed by numbered candidate methods (with source and ORIGINAL line numbers), listed in "
            "no particular order. One or a few of them contain the fault.\n\n"
            "For EACH candidate method, give an integer suspiciousness score from 0 to 100 "
            "(100 = almost certainly contains the fault, 0 = almost certainly not). "
            "Base it on the error, the stack trace, and the code. "
            'Output ONLY a JSON object mapping each index (as a string) to its score, '
            'e.g. {"1": 10, "2": 85}. Output no other text.\n')
    deli = [uvod, "=== FAILING TEST(S) / STACK TRACE ===", sled, "\n=== CANDIDATE METHODS ==="]
    for m in metode:
        deli.append(f"\n[{m['idx']}] {m['classname']}#{m['method']}  (lines {m['s']}-{m['e']})\n{m['src']}")
    return "\n".join(deli)


def manifest_vrstica(fid, projekt, bug, status, **kw):
    v = {"fault_id": fid, "project": projekt.lower(), "bug": bug, "n_cand": 0, "n_faulty_in_cand": 0,
         "faulty_indices": "", "n_chars": 0, "tok_lo": 0, "tok_hi": 0, "status": status}
    v.update(kw)
    return v


def poziv_za_napako(fid, g, mapa_pozivov, brisi_kopije):
    """Sestavi in shrani poziv ter tabelo kandidatov za eno napako. Vrne vrstico manifesta."""
    projekt, bug = ime_d4j(fid)
    g = g.copy()
    g["_skupna"] = skupna_ocena(g)
    # determinističen vrstni red: po skupni oceni, nato po imenu razreda, metode in vrstici
    g = g.sort_values(["_skupna", "classname", "method", "m_start"],
                      ascending=[False, True, True, True], kind="mergesort")
    kandidati = g.head(N.K_KANDIDATOV)

    wb = mapa_kopije(projekt, bug, "b")
    if not checkout(projekt, bug, "b", wb):
        return manifest_vrstica(fid, projekt, bug, "CHECKOUT-FAIL", n_cand=len(kandidati))
    rc, srcdir = export("dir.src.classes", wb)
    if rc != 0 or not srcdir:
        if brisi_kopije:
            shutil.rmtree(wb, ignore_errors=True)
        return manifest_vrstica(fid, projekt, bug, "EXPORT-FAIL", n_cand=len(kandidati))
    sled, _, _ = sled_sklada(projekt, bug)

    # naključen vrstni red predstavitve, seme iz oznake napake, zato ponovljiv
    vrstice = [r for _, r in kandidati.iterrows()]
    seme = int.from_bytes(hashlib.sha1(fid.encode()).digest()[:8], "big")
    vrstni_red = list(range(len(vrstice)))
    random.Random(seme).shuffle(vrstni_red)

    metode, krivi_idx, tabela, brez_vira = [], [], [], 0
    for idx, i in enumerate(vrstni_red, 1):
        r = vrstice[i]
        src = izsek_kode(pot_razreda(wb, srcdir, r["classname"]), int(r["m_start"]), int(r["m_end"]))
        if src is None:
            src = "(source unavailable)"
            brez_vira += 1
        metode.append(dict(idx=idx, classname=r["classname"], method=r["method"],
                           s=int(r["m_start"]), e=int(r["m_end"]), src=src))
        if int(r["faulty"]) == 1:
            krivi_idx.append(idx)
        tabela.append({"idx": idx, "classname": r["classname"], "method": r["method"],
                       "m_start": int(r["m_start"]), "m_end": int(r["m_end"]),
                       "ochiai": float(r["ochiai"]), "combined": float(r["_skupna"]), "faulty": int(r["faulty"])})

    poziv = sestavi_poziv(sled, metode)
    with open(os.path.join(mapa_pozivov, f"poziv_{fid}.txt"), "w", encoding="utf-8") as f:
        f.write(poziv)
    pd.DataFrame(tabela).to_csv(os.path.join(mapa_pozivov, f"kandidati_{fid}.csv"), index=False)
    if brisi_kopije:
        shutil.rmtree(wb, ignore_errors=True)
    n = len(poziv)
    status = "OK" if brez_vira == 0 else f"SRC-MISSING:{brez_vira}"
    return manifest_vrstica(fid, projekt, bug, status, n_cand=len(kandidati), n_faulty_in_cand=len(krivi_idx),
                            faulty_indices=" ".join(map(str, krivi_idx)), n_chars=n,
                            tok_lo=n // 4, tok_hi=int(n / 3.5))


def main():
    ap = argparse.ArgumentParser(description="Sestavi pozive za jezikovni model.")
    ap.add_argument("izbor", nargs="?", default="", help="seznam napak (a,b,c), privzeto vse")
    ap.add_argument("--brisi-kopije", action="store_true", help="po obdelavi napake izbriši njeno delovno kopijo")
    args = ap.parse_args()

    glava(f"pozivi za jezikovni model (K = {N.K_KANDIDATOV} kandidatov)")
    preveri_defects4j()
    mapa_pozivov = N.pot(N.POZIVI_MAPA)
    os.makedirs(mapa_pozivov, exist_ok=True)
    pot_manifesta = N.pot(N.POZIVI_MANIFEST)

    df = pd.read_parquet(N.pot(N.MATRIKA_METOD))
    vse = list(dict.fromkeys(df["fault_id"].tolist()))
    if args.izbor:
        zelene = set(args.izbor.split(","))
        napake = [f for f in vse if f in zelene]
    else:
        napake = vse

    # nadaljevanje: napake s statusom OK preskočimo, neuspele poskusimo znova
    narejene = set()
    if os.path.isfile(pot_manifesta):
        m = pd.read_csv(pot_manifesta)
        narejene = set(m[m["status"].astype(str).str.startswith("OK")]["fault_id"].astype(str))
    za_narediti = [f for f in napake if f not in narejene]
    print(f"Napak: {len(napake)}, že narejenih: {len(narejene)}, za narediti: {len(za_narediti)}")

    skupine = {fid: g for fid, g in df.groupby("fault_id")}
    for i, fid in enumerate(za_narediti, 1):
        try:
            vrstica = poziv_za_napako(fid, skupine[fid], mapa_pozivov, args.brisi_kopije)
        except Exception as e:
            projekt, bug = ime_d4j(fid)
            vrstica = manifest_vrstica(fid, projekt, bug, f"ERROR:{type(e).__name__}")
            print(f"  napaka pri {fid}: {type(e).__name__}: {e}")
        pd.DataFrame([vrstica], columns=STOLPCI_MANIFESTA).to_csv(
            pot_manifesta, mode="a", header=not os.path.isfile(pot_manifesta), index=False)
        if i % 10 == 0 or i == len(za_narediti):
            print(f"  ... {i}/{len(za_narediti)}, zadnja {fid} [{vrstica['status']}, {vrstica['n_cand']} kandidatov, "
                  f"{vrstica['n_chars']:,} znakov]")

    # povzetek in groba ocena stroška, natančno štetje žetonov opravi klici.py
    m = pd.read_csv(pot_manifesta)
    m = m[m["fault_id"].isin(napake)].drop_duplicates(subset=["fault_id"], keep="last")
    ok = m[m["status"].astype(str).str.startswith(("OK", "SRC-MISSING"))]
    znakov = int(ok["n_chars"].sum())
    tok_lo, tok_hi = int(ok["tok_lo"].sum()), int(ok["tok_hi"].sum())
    pokritost = float((ok["n_faulty_in_cand"] >= 1).mean()) if len(ok) else 0.0
    izhodnih = int(len(ok) * N.K_KANDIDATOV * 10 / 3.7)
    print("\n[Povzetek]")
    print(f"  pozivov: {len(ok)}/{len(m)} (statusi: "
          f"{m['status'].astype(str).str.replace(r':.*', '', regex=True).value_counts().to_dict()})")
    print(f"  delež napak s krivo metodo med kandidati: {pokritost:.3f}")
    if len(ok):
        print(f"  kandidatov na napako: mediana {int(ok['n_cand'].median())}, povprečje {ok['n_cand'].mean():.1f}, "
              f"največ {int(ok['n_cand'].max())}")
    print(f"  znakov skupaj: {znakov:,}, ocena vhodnih žetonov {tok_lo:,} do {tok_hi:,}, izhodnih okoli {izhodnih:,}")
    print(f"  groba ocena stroška: {tok_lo / 1e6 * N.CENA_VHOD_USD + izhodnih / 1e6 * N.CENA_IZHOD_USD:.2f} do "
          f"{tok_hi / 1e6 * N.CENA_VHOD_USD + izhodnih / 1e6 * N.CENA_IZHOD_USD:.2f} USD")
    print(f"\n[Shranjeno] {mapa_pozivov}/, {pot_manifesta}")


if __name__ == "__main__":
    main()
