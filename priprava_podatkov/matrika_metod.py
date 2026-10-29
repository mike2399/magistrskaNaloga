#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Matrika ocen na nivoju metod z oznakami krivih metod.

Za vsako napako pripravi delovni kopiji napačne in popravljene različice programa, iz razlike med njima
določi krive metode, vsak stavek preslika v objemajočo metodo (meje metod da tree-sitter) in za vsako
metodo vzame največjo oceno njenih stavkov. Napake brez lokalizabilne krive metode in napake, ki jih
Defects4J 3.0.1 več ne podpira, izloči in razlog zapiše v razvrstitev napak. Zagon je dolg, rezultati
se zapisujejo sproti in prekinjen zagon nadaljuje tam, kjer je ostal.

Zagon:  python priprava_podatkov/matrika_metod.py                   vse napake
        python priprava_podatkov/matrika_metod.py Lang              samo projekt
        python priprava_podatkov/matrika_metod.py lang1,lang23      naštete napake
        python priprava_podatkov/matrika_metod.py 10                prvih 10 napak
        (3. skripta v cevovodu, vrstni red je v README)
Vhod:   izhod/podatki/matrika_stavkov.parquet, Defects4J
Izhod:  izhod/podatki/matrika_metod.parquet (pri delnem zagonu s pripono izbora), podatki/matrika_metod_sprotno.csv
        (sprotni zapis), podatki/napake_razvrstitev.csv (razvrstitev vsake napake)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # koren repozitorija (nastavitve.py, skupno/)

import argparse
import os
import re
import shutil
import time

import pandas as pd

import nastavitve as N
from skupno.defects4j import preveri_defects4j, checkout, export, mapa_kopije, pot_razreda
from skupno.java_metode import nalozi_parser, metode_v_datoteki, objemajoca_metoda, krive_metode_iz_popravka
from skupno.podatki import ime_d4j
from skupno.preverbe import Preverbe, glava

STOLPCI_INFO = ["fault_id", "kind", "n_methods", "n_faulty", "gt_total", "gt_covered", "unmapped_stmts", "xcheck"]


def info_vrstica(fid, vrsta, **kw):
    v = {"fault_id": fid, "kind": vrsta, "n_methods": 0, "n_faulty": 0, "gt_total": 0,
         "gt_covered": 0, "unmapped_stmts": 0, "xcheck": True}
    v.update(kw)
    return v


def obdelaj_napako(fid, sub_df, parser, brisi_kopije):
    """Vrne (vrstice metod ali None, info o napaki)."""
    projekt, bug = ime_d4j(fid)
    wb, wf = mapa_kopije(projekt, bug, "b"), mapa_kopije(projekt, bug, "f")
    if not (checkout(projekt, bug, "b", wb) and checkout(projekt, bug, "f", wf)):
        return None, info_vrstica(fid, "CHECKOUT-FAIL")

    _, srcdir_b = export("dir.src.classes", wb)
    _, srcdir_f = export("dir.src.classes", wf)
    rc, spremenjeni = export("classes.modified", wb)
    spremenjeni = [m for m in spremenjeni.split() if m.strip()]
    if rc != 0 or not spremenjeni:
        return None, info_vrstica(fid, "EXPORT-FAIL", xcheck=False)
    krive, nove, vrsta, ujemanje = krive_metode_iz_popravka(wb, wf, srcdir_b, srcdir_f, spremenjeni, parser)

    if vrsta != "LOKALIZABILNO":
        if brisi_kopije:
            shutil.rmtree(wb, ignore_errors=True)
            shutil.rmtree(wf, ignore_errors=True)
        return None, info_vrstica(fid, vrsta, gt_total=len(krive), xcheck=ujemanje)

    # preslikava stavkov v metode in največja ocena na metodo
    metode_datoteke, zdruzeno, nepreslikanih = {}, {}, 0
    for _, r in sub_df.iterrows():
        pot = pot_razreda(wb, srcdir_b, r["classname"])
        if pot not in metode_datoteke:
            metode_datoteke[pot] = metode_v_datoteki(pot, parser)
        metode = metode_datoteke[pot]
        m = objemajoca_metoda(metode, int(r["line"])) if metode else None
        if m is None:
            nepreslikanih += 1          # vrstica zunaj metod (polja, deklaracije razreda)
            continue
        kljuc = (r["classname"].split("$")[0], m[0], m[1], m[2])
        ocene = zdruzeno.setdefault(kljuc, {t: 0.0 for t in N.TEHNIKE})
        for t in N.TEHNIKE:
            v = float(r[t])
            if v > ocene[t]:
                ocene[t] = v

    vrstice = []
    for (razred, ime, od, do), ocene in zdruzeno.items():
        v = {"fault_id": fid, "project": projekt.lower(), "classname": razred, "method": ime,
             "m_start": od, "m_end": do, "faulty": int((razred, ime, od, do) in krive)}
        v.update(ocene)
        vrstice.append(v)
    pokritih = sum(1 for k in krive if k in zdruzeno)
    info = info_vrstica(fid, vrsta, n_methods=len(vrstice), n_faulty=sum(v["faulty"] for v in vrstice),
                        gt_total=len(krive), gt_covered=pokritih, unmapped_stmts=nepreslikanih, xcheck=ujemanje)
    if brisi_kopije:
        shutil.rmtree(wb, ignore_errors=True)
        shutil.rmtree(wf, ignore_errors=True)
    return vrstice, info


def razcleni_izbor(vse, izbor):
    """Vrne (seznam napak, oznaka izbora za imena datotek)."""
    if not izbor:
        return vse, ""
    if izbor.isdigit():
        return vse[:int(izbor)], f"prvih{izbor}"
    if "," in izbor or re.fullmatch(r"[a-z]+\d+", izbor):
        zelene = set(izbor.split(","))
        return [f for f in vse if f in zelene], "izbor"
    return [f for f in vse if f.startswith(izbor.lower())], izbor.lower()


def pripni(pot, vrstice, stolpci=None):
    pd.DataFrame(vrstice, columns=stolpci).to_csv(pot, mode="a", header=not os.path.isfile(pot), index=False)


def main():
    ap = argparse.ArgumentParser(description="Združevanje stavkov v metode in oznake krivih metod.")
    ap.add_argument("izbor", nargs="?", default="", help="projekt, seznam napak (a,b,c) ali število prvih napak")
    ap.add_argument("--brisi-kopije", action="store_true", help="po obdelavi napake izbriši njeni delovni kopiji")
    args = ap.parse_args()

    glava("matrika ocen na nivoju metod")
    preveri_defects4j()
    df = pd.read_parquet(N.pot(N.MATRIKA_STAVKOV))
    vse = list(dict.fromkeys(df["fault_id"].tolist()))
    napake, oznaka = razcleni_izbor(vse, args.izbor)
    pripona = f"_{oznaka}" if oznaka else ""
    pot_info = N.pot(N.NAPAKE_INFO.replace(".csv", f"{pripona}.csv"))
    pot_csv = N.pot(N.MATRIKA_METOD_CSV.replace(".csv", f"{pripona}.csv"))
    pot_parquet = N.pot(N.MATRIKA_METOD.replace(".parquet", f"{pripona}.parquet"))

    obdelane = set(pd.read_csv(pot_info)["fault_id"].astype(str)) if os.path.isfile(pot_info) else set()
    za_obdelati = [f for f in napake if f not in obdelane]
    print(f"Napak v izboru: {len(napake)}, že obdelanih: {len(obdelane)}, za obdelati: {len(za_obdelati)}")

    parser = nalozi_parser()
    skupine = {fid: g for fid, g in df.groupby("fault_id")}
    zacetek = time.time()
    for i, fid in enumerate(za_obdelati, 1):
        try:
            vrstice, info = obdelaj_napako(fid, skupine[fid], parser, args.brisi_kopije)
        except Exception as e:                      # ena napaka ne sme ustaviti celotnega zagona
            vrstice, info = None, info_vrstica(fid, f"ERROR:{type(e).__name__}")
            print(f"  napaka pri {fid}: {type(e).__name__}: {e}")
        if vrstice:
            pripni(pot_csv, vrstice)
        pripni(pot_info, [info], STOLPCI_INFO)
        if i % 5 == 0 or i == len(za_obdelati):
            print(f"  ... {i}/{len(za_obdelati)} ({time.time() - zacetek:.0f} s), zadnja {fid} [{info['kind']}]")

    # združitev sprotnih zapisov v končno tabelo
    info = pd.read_csv(pot_info) if os.path.isfile(pot_info) else pd.DataFrame(columns=STOLPCI_INFO)
    info["kind"] = info["kind"].astype(str)
    info = info.drop_duplicates(subset=["fault_id"], keep="last")
    metode = pd.read_csv(pot_csv) if os.path.isfile(pot_csv) else pd.DataFrame()
    if len(metode):
        metode = metode.drop_duplicates(subset=["fault_id", "classname", "method", "m_start", "m_end"])

    print("\n[Statistika]")
    print(f"  razvrstitev napak: {info['kind'].value_counts().to_dict()}")
    lok = info[info["kind"] == "LOKALIZABILNO"]
    print(f"  matrika metod: {len(metode)} metod, {metode['fault_id'].nunique() if len(metode) else 0} napak, "
          f"krivih metod {int(metode['faulty'].sum()) if len(metode) else 0}")

    print("\n[Preverbe]")
    p = Preverbe()
    uspesne = info[~info["kind"].isin(["CHECKOUT-FAIL", "EXPORT-FAIL"]) & ~info["kind"].str.startswith("ERROR")]
    neuspeli_checkouti = info[info["kind"] == "CHECKOUT-FAIL"]["fault_id"].tolist()
    napake_zagona = info[info["kind"].str.startswith("ERROR") | (info["kind"] == "EXPORT-FAIL")]["fault_id"].tolist()
    p("vsaka napaka iz izbora ima vrstico v razvrstitvi napak", set(info["fault_id"].astype(str)) >= set(napake))
    p("razredi z razlikami se ujemajo s classes.modified pri vseh obdelanih napakah",
      bool(uspesne["xcheck"].all()) if len(uspesne) else True)
    p("ni napak zagona ali neuspelih izvozov", len(napake_zagona) == 0, f"{napake_zagona[:10]}")
    if neuspeli_checkouti:
        print(f"  opomba: checkout ni uspel pri {len(neuspeli_checkouti)} napakah (v Defects4J 3.0.1 opuščene): "
              f"{neuspeli_checkouti}")
    if len(metode):
        p("v matriki metod ni manjkajočih vrednosti", int(metode[N.TEHNIKE + ["faulty"]].isna().sum().sum()) == 0)
        p("število krivih metod je enako številu pokritih krivih metod iz popravka",
          bool((lok["n_faulty"] == lok["gt_covered"]).all()))
        p("osem tehnik (brez DStar) ostane v [0, 1] po združevanju",
          float(metode[[t for t in N.TEHNIKE if t != "dstar"]].max().max()) <= 1.0)
        nepokrite = int((lok["gt_total"] - lok["gt_covered"]).clip(lower=0).sum())
        print(f"  krivih metod brez kandidatnih stavkov v matriki (nepokrite): {nepokrite}")
        metode.to_parquet(pot_parquet, index=False)
    print(f"\n[Shranjeno] {pot_parquet if len(metode) else '(brez metod)'}\n            {pot_info}")
    p.zakljuci()


if __name__ == "__main__":
    main()
