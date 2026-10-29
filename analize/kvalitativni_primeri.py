#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Izbor konkretnih napak za kvalitativno analizo (brez klicev API).

Iz shranjenih napovedi izračuna rang prve krive metode za vsak model, preveri nekaj skupnih
števil in po vnaprej določenih pravilih izbere okoli 20 napak v šestih skupinah:
  K1_prednost_llm   jezikovni model uvrsti krivo metodo med prvi dve, klasični modeli slabo
  K2_slabost_llm    kriva metoda je bila med kandidati, a jo jezikovni model uvrsti na rang 10 ali slabše
  K3_pokritost      kriva metoda sploh ni prišla med 50 kandidatov
  K4_polni_pogled   napake z največ 50 metodami, kjer se H9+LLM in LLM sam razlikujeta
  K5_time22         edina napaka, za katero model ni vrnil nobene ocene
  K6_klasika        kontrola H9 je boljša od H9+LLM
Za vsako izbrano napako zapiše dokazni izpis: range, ocene krivih metod, prvih deset po LLM, H9
in Ochiai ter iz dejansko poslanega poziva sled sklada in izvorno kodo krivih metod.

Zagon:  python analize/kvalitativni_primeri.py
        (13. skripta v cevovodu, vrstni red je v README)
Vhod:   izhod/modeli/mlp_llm_napovedi.csv, modeli/ranksvm_h9_napovedi.csv, jezikovni_model/ocene.csv, jezikovni_model/manifest_pozivov.csv,
        jezikovni_model/tokeni.csv, jezikovni_model/pozivi/
Izhod:  izhod/analize/kvalitativni_primeri/<skupina>__<napaka>.txt, izhod/analize/kvalitativni_primeri_pregled.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # koren repozitorija (nastavitve.py, skupno/)

import os
import re

import pandas as pd
from scipy.stats import rankdata

import nastavitve as N
from skupno.metrike import prvi_rangi
from skupno.preverbe import glava

MAX_VRSTIC_BLOKA = 55          # koliko vrstic izvorne kode obdržimo na kandidata v dokaznem izpisu
MAX_ZNAKOV_SLEDI = 5000
POZIVI = N.pot(N.POZIVI_MAPA)


def razcleni_poziv(fid):
    """Vrne (sled sklada, {zaporedna številka kandidata: blok besedila})."""
    with open(os.path.join(POZIVI, f"poziv_{fid}.txt"), encoding="utf-8") as f:
        txt = f.read()
    m = re.search(r"=== FAILING TEST\(S\) / STACK TRACE ===\n(.*?)\n=== CANDIDATE METHODS ===", txt, re.S)
    sled = m.group(1).strip() if m else "(sledi ni bilo mogoče izluščiti)"
    telo = txt.split("=== CANDIDATE METHODS ===", 1)[-1]
    bloki = {int(b.group(1)): b.group(0).rstrip() for b in re.finditer(r"^\[(\d+)\] .*?(?=^\[\d+\] |\Z)", telo, re.S | re.M)}
    return sled, bloki


def skrajsaj(blok):
    vrstice = blok.splitlines()
    if len(vrstice) > MAX_VRSTIC_BLOKA:
        vrstice = vrstice[:MAX_VRSTIC_BLOKA] + [f"    ... [skrajšano, še {len(blok.splitlines()) - MAX_VRSTIC_BLOKA} vrstic]"]
    return "\n".join(vrstice)


def main():
    glava("izbor napak za kvalitativno analizo")
    mapa = N.pot(N.PRIMERI_MAPA)
    os.makedirs(mapa, exist_ok=True)
    d = pd.read_csv(N.pot(N.MLP_LLM_NAPOVEDI))
    d["llm_scored"] = d["llm_scored"].astype(str).str.lower().isin(["true", "1"])
    d10 = pd.read_csv(N.pot(N.RANKSVM_NAPOVEDI))
    ocene = pd.read_csv(N.pot(N.LLM_OCENE))
    manifest = pd.read_csv(N.pot(N.POZIVI_MANIFEST)).set_index("fault_id")
    tokeni = pd.read_csv(N.pot(N.LLM_TOKENI)).set_index("fault_id")

    rangi = {ime: prvi_rangi(d, ime) for ime in ("ochiai", "llm", "mlp_h9", "mlp_h9llm", "kaskada")}
    rangi["ranksvm"] = prvi_rangi(d10, "ranksvm_raw")

    info = d.groupby("fault_id").agg(n_methods=("faulty", "size"), n_faulty=("faulty", "sum"))
    krive = d[d["faulty"] == 1]
    info["covered"] = krive.groupby("fault_id")["llm_scored"].any().reindex(info.index).fillna(False)
    info["fullview"] = info["n_methods"] <= N.K_KANDIDATOV
    info["max_llm_faulty"] = krive.groupby("fault_id")["llm"].max().reindex(info.index)
    for ime in rangi:
        info["r_" + ime] = pd.Series(rangi[ime])
    # ali je bila kriva metoda v odgovoru res ocenjena (llm_missing = 0) ali je dobila privzeto 0
    info["faulty_llm_missing"] = ocene[ocene["faulty"] == 1].groupby("fault_id")["llm_missing"].min().reindex(info.index)

    print("\n[Skupna števila, preračunana iz shranjenih napovedi]")
    print(f"  napak {len(info)}, metod {len(d):,}, največ metod na napako {int(info['n_methods'].max())}")
    print(f"  polni pogled (do {N.K_KANDIDATOV} metod): {int(info['fullview'].sum())}, "
          f"mediana metod {int(info.loc[info['fullview'], 'n_methods'].median())}")
    print(f"  napak z več kot eno krivo metodo: {int((info['n_faulty'] > 1).sum())} (največ {int(info['n_faulty'].max())})")
    nepokrite = info[~info["covered"]]
    print(f"  kriva metoda ni med kandidati: {len(nepokrite)} ({len(nepokrite) / len(info):.1%})")
    polni = info[info["fullview"]]
    razlicni = polni[polni["r_llm"] != polni["r_mlp_h9llm"]]
    print(f"  polni pogled, LLM sam in H9+LLM se razlikujeta: {len(razlicni)} (kombinacija boljša "
          f"{int((razlicni['r_mlp_h9llm'] < razlicni['r_llm']).sum())}, LLM boljši {int((razlicni['r_llm'] < razlicni['r_mlp_h9llm']).sum())})")
    t22 = ocene[ocene["fault_id"] == "time22"]
    print(f"  time22: kandidatov {len(t22)}, brez ocene {int(t22['llm_missing'].sum())}, vhodnih žetonov "
          f"{int(tokeni.loc['time22', 'input_tokens'])} (mediana vseh {int(tokeni['input_tokens'].median())})")
    delni = ocene.groupby("fault_id")["llm_missing"].agg(["sum", "size"])
    delni = delni[(delni["sum"] > 0) & (delni["sum"] < delni["size"])]
    print(f"  napak z delnim odgovorom (del kandidatov brez ocene): {len(delni)}")

    # izbor po skupinah, vrstni red skupin določa prednost, če napaka ustreza več skupinam
    izbrane = {}

    def vzemi(kandidati, skupina, n):
        out = []
        for f in kandidati:
            if f not in izbrane:
                izbrane[f] = skupina
                out.append(f)
            if len(out) == n:
                break
        return out

    vzemi(["time22"], "K5_time22", 1)
    k3 = nepokrite.sort_values("r_mlp_h9llm")
    vzemi(list(dict.fromkeys([k3.index[0], k3.index[-1], nepokrite.sort_values("n_methods").index[-1]] + list(k3.index[1:]))),
          "K3_pokritost", 3)
    komb = razlicni[razlicni["r_mlp_h9llm"] < razlicni["r_llm"]].copy()
    komb["diff"] = komb["r_llm"] - komb["r_mlp_h9llm"]
    sam = razlicni[razlicni["r_llm"] < razlicni["r_mlp_h9llm"]].copy()
    sam["diff"] = sam["r_mlp_h9llm"] - sam["r_llm"]
    vzemi(list(komb.sort_values("diff", ascending=False).index), "K4_polni_pogled", 2)
    vzemi(list(sam.sort_values("diff", ascending=False).index), "K4_polni_pogled", 2)
    k1 = info[info["covered"] & (info["r_llm"] <= 2)].copy()
    k1["klasika_min"] = k1[["r_mlp_h9", "r_ochiai"]].min(axis=1)
    k1 = k1[k1["klasika_min"] >= 5].sort_values("klasika_min", ascending=False)
    vzemi(list(k1.index), "K1_prednost_llm", 5)
    k2 = info[info["covered"] & (info["r_llm"] >= 10)].copy()
    k2a = k2[k2["faulty_llm_missing"] == 0].sort_values("r_llm", ascending=False)   # izrecno nizko ocenjene
    k2b = k2[k2["faulty_llm_missing"] == 1].sort_values("r_llm", ascending=False)   # brez ocene v delnem odgovoru
    izbor2 = vzemi(list(k2a.index), "K2_slabost_llm", 3) + vzemi(list(k2b.index), "K2_slabost_llm", 2)
    if len(izbor2) < 5:
        vzemi(list(k2a.index), "K2_slabost_llm", 5 - len(izbor2))
    print(f"  K2: izrecno nizko ocenjenih {len(k2a)}, brez ocene v delnem odgovoru {len(k2b)}")
    k6 = info[info["r_mlp_h9"] < info["r_mlp_h9llm"]].copy()
    k6["diff"] = k6["r_mlp_h9llm"] - k6["r_mlp_h9"]
    print(f"  K6: napak, kjer je H9 boljši od H9+LLM: {len(k6)}, kjer je Ochiai boljši od H9+LLM: "
          f"{int((info['r_ochiai'] < info['r_mlp_h9llm']).sum())}")
    vzemi(list(k6.sort_values("diff", ascending=False).index), "K6_klasika", 2)

    pregled = info.copy()
    pregled["kategorija"] = pd.Series(izbrane)
    pregled.to_csv(N.pot(N.PRIMERI_PREGLED))

    print(f"\n[Izbrane napake: {len(izbrane)}]")
    print(f"  {'napaka':12s} {'skupina':16s} {'metod':>5s} {'kriv':>4s} {'Och':>7s} {'rSVM':>7s} {'H9':>7s} {'LLM':>7s} {'H9+L':>7s} {'kask':>7s}")
    for f, k in sorted(izbrane.items(), key=lambda x: (x[1], x[0])):
        r = info.loc[f]
        print(f"  {f:12s} {k:16s} {int(r['n_methods']):5d} {int(r['n_faulty']):4d} {r['r_ochiai']:7.1f} {r['r_ranksvm']:7.1f} "
              f"{r['r_mlp_h9']:7.1f} {r['r_llm']:7.1f} {r['r_mlp_h9llm']:7.1f} {r['r_kaskada']:7.1f}")

    # dokazni izpisi
    def vrstica_kandidata(tabela, row):
        return tabela[(tabela["classname"] == row["classname"]) & (tabela["method"] == row["method"]) &
                      (tabela["m_start"] == row["m_start"]) & (tabela["m_end"] == row["m_end"])]

    for fid, skupina in sorted(izbrane.items(), key=lambda x: (x[1], x[0])):
        r = info.loc[fid]
        g = d[d["fault_id"] == fid].copy()
        for stolpec, ime in (("llm", "rang_llm"), ("mlp_h9", "rang_h9"), ("mlp_h9llm", "rang_h9llm"), ("ochiai", "rang_och")):
            g[ime] = rankdata(-g[stolpec].to_numpy(float), method="average")
        kandidati = pd.read_csv(os.path.join(POZIVI, f"kandidati_{fid}.csv"))
        sled, bloki = razcleni_poziv(fid)
        ocene_f = ocene[ocene["fault_id"] == fid]

        L = [f"{'=' * 88}\nDOKAZNI IZPIS: {fid}   [{skupina}]\n{'=' * 88}",
             f"metod: {int(r['n_methods'])} | krivih: {int(r['n_faulty'])} | kandidatov v pozivu: {int(manifest.loc[fid, 'n_cand'])} "
             f"| vhodnih žetonov: {int(tokeni.loc[fid, 'input_tokens'])} | polni pogled: {bool(r['fullview'])} "
             f"| kriva med kandidati: {bool(r['covered'])}",
             f"RANG PRVE KRIVE METODE: Ochiai {r['r_ochiai']:.1f}, rankSVM {r['r_ranksvm']:.1f}, H9 {r['r_mlp_h9']:.1f}, "
             f"LLM {r['r_llm']:.1f}, H9+LLM {r['r_mlp_h9llm']:.1f}, kaskada {r['r_kaskada']:.1f}",
             "\n--- KRIVE METODE ---"]
        for _, row in g[g["faulty"] == 1].iterrows():
            k = vrstica_kandidata(kandidati, row)
            o = vrstica_kandidata(ocene_f, row)
            idx = int(k["idx"].iloc[0]) if len(k) else None
            manjka = int(o["llm_missing"].iloc[0]) if len(o) else None
            L.append(f"  {row['classname']}#{row['method']} (vrstice {row['m_start']}-{row['m_end']}) | "
                     f"št. v pozivu: {idx if idx is not None else 'ni med kandidati'} | llm {row['llm']:.0f} (brez ocene: {manjka}) | "
                     f"ochiai {row['ochiai']:.4f} | h9 {row['mlp_h9']:.3f} | h9llm {row['mlp_h9llm']:.3f} | "
                     f"rangi: och {row['rang_och']:.1f}, h9 {row['rang_h9']:.1f}, llm {row['rang_llm']:.1f}, h9llm {row['rang_h9llm']:.1f}")

        def prvih(stolpec, rang, n=10):
            t = g.sort_values(rang).head(n)
            return "\n".join(f"    {'*' if x['faulty'] == 1 else ' '} rang {x[rang]:6.1f}  {stolpec} {x[stolpec]:8.3f}  "
                             f"{x['classname'].split('.')[-1]}#{x['method']} ({x['m_start']}-{x['m_end']})" for _, x in t.iterrows())

        L.append("\n--- PRVIH 10 PO LLM (zvezdica = kriva) ---\n" + prvih("llm", "rang_llm"))
        L.append("\n--- PRVIH 10 PO H9 ---\n" + prvih("mlp_h9", "rang_h9"))
        L.append("\n--- PRVIH 10 PO OCHIAI ---\n" + prvih("ochiai", "rang_och"))
        L.append("\n--- SPROŽILNI TESTI IN SLED SKLADA IZ POZIVA (skrajšano) ---")
        L.append(sled[:MAX_ZNAKOV_SLEDI] + ("\n... [skrajšano]" if len(sled) > MAX_ZNAKOV_SLEDI else ""))
        L.append("\n--- IZVORNA KODA IZ POZIVA ---")
        prikazani = set()
        for _, row in g[g["faulty"] == 1].iterrows():
            k = vrstica_kandidata(kandidati, row)
            if len(k):
                idx = int(k["idx"].iloc[0])
                if idx not in prikazani:
                    prikazani.add(idx)
                    L.append(f"\n[KRIVA, št. {idx}]\n" + skrajsaj(bloki.get(idx, "(bloka ni)")))
        nekrive = g[g["llm_scored"] & (g["faulty"] == 0)].sort_values("rang_llm")
        if len(nekrive):
            prva = nekrive.iloc[0]
            k = vrstica_kandidata(kandidati, prva)
            if len(k):
                idx = int(k["idx"].iloc[0])
                if idx not in prikazani:
                    L.append(f"\n[NAJVIŠJE OCENJENA NEKRIVA PO LLM, št. {idx}, llm {prva['llm']:.0f}]\n" + skrajsaj(bloki.get(idx, "(bloka ni)")))
        with open(os.path.join(mapa, f"{skupina}__{fid}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(L) + "\n")

    print(f"\n[Shranjeno] {len(izbrane)} dokaznih izpisov v {mapa}/, pregled vseh napak v {N.PRIMERI_PREGLED}")


if __name__ == "__main__":
    main()
