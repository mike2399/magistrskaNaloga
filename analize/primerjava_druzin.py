#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Primerjava posameznih tehnik in družin tehnik na nivoju metod.

Vsako od devetih tehnik ovrednoti samostojno in določi njeno uporabnost, to je napake, pri katerih
tehnika sploh da signal. Ker tehnike z ozko uporabnostjo delujejo na manjših programih, vsako primerja
z Ochiai na isti podmnožici napak, s parnim testom, mediano in Top-1. Družine primerja prek predstavnikov
na vseh napakah in samo tam, kjer imata obe tehniki signal, vrednosti p popravi po Holmu. Doda še analizo
komplementarnosti.

Zagon:  python analize/primerjava_druzin.py
        (14. skripta v cevovodu, vrstni red je v README)
Vhod:   izhod/podatki/matrika_metod.parquet, modeli/mlp_llm_rezultati.json (dosežen Top-1 naučene kombinacije)
Izhod:  izhod/analize/primerjava_druzin.json, analize/primerjava_druzin.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # koren repozitorija (nastavitve.py, skupno/)

import itertools
import json
import os

import numpy as np

import nastavitve as N
from skupno.metrike import metrike_po_napakah, top1_po_konvencijah, wilcoxon_parni, holm
from skupno.podatki import nalozi_matriko_metod, seznam_napak, normiraj_po_napaki, v_intervalu_0_1
from skupno.preverbe import Preverbe, glava


def test(rang_a, rang_b, napake):
    """Dvostranski parni Wilcoxon (asimptotični)."""
    w = wilcoxon_parni(rang_a, rang_b, napake, "two-sided", "asymptotic")
    return w["W"], w["p"], w["a_better"], w["a_worse"], w["tie"]


def main():
    glava("primerjava tehnik in družin")
    p = Preverbe()
    df = nalozi_matriko_metod(N.pot(N.MATRIKA_METOD))
    napake = seznam_napak(df)
    n_f = len(napake)
    velikosti = df.groupby("fault_id").size().to_dict()
    p("vseh devet značilk je prisotnih brez manjkajočih vrednosti",
      all(c in df.columns for c in N.TEHNIKE) and bool(df[N.TEHNIKE].notna().all().all()))
    p("družine pokrijejo natanko devet tehnik brez prekrivanja",
      sorted(itertools.chain.from_iterable(N.DRUZINE.values())) == sorted(N.TEHNIKE))
    druzina = {t: fam for fam, ts in N.DRUZINE.items() for t in ts}

    trivialne = {f for f in napake if velikosti[f] == 1}
    netrivialne = [f for f in napake if f not in trivialne]
    print(f"\nMatrika: {len(df):,} metod, {n_f} napak, {int(df['faulty'].sum())} krivih metod")
    print(f"Napak z eno samo metodo (vsaka tehnika jih uvrsti v Top-1): {len(trivialne)}")

    # uporabnost: ohlapna (vsaj ena neničelna ocena) in stroga (ocene se razlikujejo)
    print("\n[Uporabnost tehnik: ohlapno = vsaj ena neničelna ocena, strogo = ocene se razlikujejo]")
    ohlapno, strogo = {}, {}
    print(f"  {'tehnika':22s} {'ohlapno':>8s} {'strogo':>7s} {'neničelnih metod':>17s}")
    for t in N.TEHNIKE:
        g = df.groupby("fault_id")[t]
        nz = g.apply(lambda s: bool((s != 0).any()))
        var = g.apply(lambda s: bool(s.max() > s.min()))
        ohlapno[t] = set(nz.index[nz.to_numpy()])
        strogo[t] = set(var.index[var.to_numpy()])
        print(f"  {t:22s} {len(ohlapno[t]):8d} {len(strogo[t]):7d} {int((df[t] != 0).sum()):17,}")
    p("stroga uporabnost je podmnožica ohlapne", all(strogo[t] <= ohlapno[t] for t in N.TEHNIKE))

    # (A) vsaka tehnika na vseh napakah
    print(f"\n[A: posamezne tehnike na vseh {n_f} napakah]")
    print(f"  {'tehnika':22s} {'družina':11s} " + " ".join(f"{'T' + str(n):>5s}" for n in N.NS) + f" {'MFR':>8s} {'MAR':>8s} {'T1 netriv.':>10s}")
    rez, rangi = {}, {}
    for t in N.TEHNIKE:
        top, mfr, mar, rang = metrike_po_napakah(df, t)
        t1_nt = sum(1 for f in netrivialne if rang[f] <= 1)
        rez[t] = {"family": druzina[t], "all": {"top": top, "mfr": mfr, "mar": mar}, "top1_nontrivial": t1_nt,
                  "top1_tie": top1_po_konvencijah(df, t), "applicable_loose": len(ohlapno[t]), "applicable_strict": len(strogo[t])}
        rangi[t] = rang
        print(f"  {t:22s} {druzina[t]:11s} " + " ".join(f"{top[n]:5d}" for n in N.NS) + f" {mfr:8.2f} {mar:8.2f} {t1_nt:10d}")

    # (B) vsaka tehnika tam, kjer ima signal, z Ochiai in DStar na isti podmnožici
    print("\n[B: tehnika na napakah, kjer ima signal, proti Ochiai na istih napakah, parni test]")
    print(f"  {'tehnika':22s} {'n':>4s} {'MFR t':>7s} {'MFR O':>7s} {'med t':>6s} {'med O':>6s} {'T1 t':>5s} {'T1 O':>5s} "
          f"{'boljša/slabša':>13s} {'p(Holm)':>9s}  sodba")
    surovi_p = []
    for t in N.TEHNIKE:
        sub = strogo[t]
        top_t, mfr_t, mar_t, rang_t = metrike_po_napakah(df, t, podmnozica=sub)
        top_o, mfr_o, _, rang_o = metrike_po_napakah(df, "ochiai", podmnozica=sub)
        top_d, mfr_d, _, _ = metrike_po_napakah(df, "dstar", podmnozica=sub)
        presek = sorted(strogo[t] & strogo["ochiai"])
        if t == "ochiai":
            W, pv, boljsa, slabsa, enako = float("nan"), float("nan"), 0, 0, len(sub)
        else:
            W, pv, boljsa, slabsa, enako = test(rang_t, rang_o, presek)
        rez[t]["signal_only"] = {
            "n_faults": len(sub), "median_methods": float(np.median([velikosti[f] for f in sub])),
            "top": top_t, "mfr": mfr_t, "mar": mar_t, "median_first_rank": float(np.median([rang_t[f] for f in sub])),
            "matched_ochiai": {"top": top_o, "mfr": mfr_o, "median_first_rank": float(np.median([rang_o[f] for f in sub]))},
            "matched_dstar": {"top": top_d, "mfr": mfr_d},
            "paired_vs_ochiai": {"n": len(presek), "W": W, "p_raw": pv, "tech_better": boljsa, "tech_worse": slabsa, "tie": enako},
            "mfr_lower_than_ochiai": bool(mfr_t < mfr_o),
        }
        surovi_p.append((t, pv))
    # Holmov popravek čez osem primerjav (brez Ochiai proti sebi)
    brez_och = [i for i, (t, _) in enumerate(surovi_p) if t != "ochiai"]
    popravljeni = dict(zip([surovi_p[i][0] for i in brez_och], holm([surovi_p[i][1] for i in brez_och])))
    for t in N.TEHNIKE:
        s = rez[t]["signal_only"]
        pv = s["paired_vs_ochiai"]
        ph = popravljeni.get(t, float("nan"))
        pv["p_holm"] = ph
        if t == "ochiai":
            sodba = "= sam s sabo"
        elif ph == ph and ph < 0.05:
            sodba = "OCHIAI ZNAČILNO BOLJŠI" if pv["tech_worse"] > pv["tech_better"] else "TEHNIKA ZNAČILNO BOLJŠA"
        else:
            sodba = "NI ZNAČILNE RAZLIKE"
        s["verdict_paired"] = sodba
        print(f"  {t:22s} {pv['n']:4d} {s['mfr']:7.2f} {s['matched_ochiai']['mfr']:7.2f} {s['median_first_rank']:6.1f} "
              f"{s['matched_ochiai']['median_first_rank']:6.1f} {s['top'][1]:5d} {s['matched_ochiai']['top'][1]:5d} "
              f"{pv['tech_better']:5d}/{pv['tech_worse']:<7d} {ph:9.3g}  {sodba}")
    print("  (n = napake, kjer imata signal tehnika in Ochiai, Holm čez osem primerjav)")

    print("\n[Top-1 po obravnavi enakih ocen: optimistična / povprečna / pesimistična]")
    for t in N.TEHNIKE:
        tt = rez[t]["top1_tie"]
        print(f"  {t:22s} {tt['min']:4d} {tt['avg']:4d} {tt['max']:4d}")

    # (C) družine, predstavnik je tehnika z najnižjim MFR, za primerjavo še največja normirana ocena v družini
    print("\n[C: družine (predstavnik je tehnika z najnižjim MFR, izbran na istih podatkih, zato optimistično)]")
    dfn = normiraj_po_napaki(df, N.TEHNIKE)
    p("po normalizaciji so vse značilke v [0, 1]", v_intervalu_0_1(dfn, N.TEHNIKE))
    druzine_rez = {}
    for fam, ts in N.DRUZINE.items():
        predstavnik = min(ts, key=lambda t: rez[t]["all"]["mfr"])
        stolpec = f"_fam_{fam}"
        df[stolpec] = dfn[ts].max(axis=1) if len(ts) > 1 else dfn[ts[0]]
        top, mfr, mar, _ = metrike_po_napakah(df, stolpec)
        druzine_rez[fam] = {
            "techniques": ts, "representative": predstavnik, "representative_metrics": rez[predstavnik]["all"],
            "family_max_construction": ({"top": top, "mfr": mfr, "mar": mar} if len(ts) > 1 else "= sama tehnika (družina ima eno tehniko)"),
            "union_applicable_strict": len(set().union(*[strogo[t] for t in ts])),
        }
        rm = rez[predstavnik]["all"]
        print(f"  {fam:11s} tehnik {len(ts)}, uporabna (strogo, unija) {druzine_rez[fam]['union_applicable_strict']:3d}, "
              f"predstavnik {predstavnik:20s} T1 {rm['top'][1]:3d} MFR {rm['mfr']:7.2f}"
              + (f", največja ocena v družini T1 {top[1]:3d} MFR {mfr:7.2f}" if len(ts) > 1 else ""))
        df.drop(columns=[stolpec], inplace=True)

    # (D) komplementarnost
    print("\n[D: komplementarnost]")
    resene = {t: {f for f in napake if rangi[t][f] <= 1} for t in N.TEHNIKE}
    samo = {}
    print(f"  {'tehnika':22s} {'T1':>5s} {'T1 netriv.':>10s} {'reši samo ta':>13s}")
    for t in N.TEHNIKE:
        ostale = set().union(*[resene[o] for o in N.TEHNIKE if o != t])
        samo[t] = resene[t] - ostale
        print(f"  {t:22s} {len(resene[t]):5d} {len(resene[t] - trivialne):10d} {len(samo[t]):13d}")
    zgornja_meja = set().union(*resene.values())
    najboljsa = max(N.TEHNIKE, key=lambda t: len(resene[t]))
    unija_sbfl = len(resene["ochiai"] | resene["dstar"])
    p("zgornja meja vsebuje vsako posamezno tehniko", all(resene[t] <= zgornja_meja for t in N.TEHNIKE))
    p("nobena napaka, ki jo reši samo ena tehnika, ni trivialna", all(not (samo[t] & trivialne) for t in N.TEHNIKE))
    dosezeno = None
    pot08 = N.pot(N.MLP_LLM_REZULTATI)
    if os.path.isfile(pot08):
        with open(pot08, encoding="utf-8") as f:
            dosezeno = json.load(f)["all"]["MLP H9 (kontrola)"]["top"]["1"]
    print(f"  zgornja meja (popoln izbiralnik tehnike za vsako napako): {len(zgornja_meja)}/{n_f}, "
          f"najboljša posamezna {najboljsa} {len(resene[najboljsa])}, unija Ochiai in DStar {unija_sbfl}, "
          f"dosežena naučena kombinacija (mlp_llm.py, MLP H9) {dosezeno}")

    # (E) statistika med predstavniki družin, na vseh napakah in na preseku, s Holmovim popravkom
    print("\n[E: parni Wilcoxon med predstavniki družin: na vseh napakah in tam, kjer imata obe signal]")
    predstavniki = {fam: druzine_rez[fam]["representative"] for fam in N.DRUZINE}
    pari = list(itertools.combinations(N.DRUZINE.keys(), 2))
    vrstice = []
    for fa, fb in pari:
        a, b = predstavniki[fa], predstavniki[fb]
        W1, p1, b1, w1, t1 = test(rangi[a], rangi[b], napake)
        presek = sorted(strogo[a] & strogo[b])
        W2, p2, b2, w2, t2 = test(rangi[a], rangi[b], presek)
        vrstice.append([fa, fb, a, b, W1, p1, b1, w1, t1, len(presek), W2, p2, b2, w2, t2])
    h1 = holm([v[5] for v in vrstice])
    h2 = holm([v[11] for v in vrstice])
    stat, obrnitve, izgube, podrobnosti = {}, 0, 0, []
    print(f"  {'primerjava':24s} | {'vse: p (Holm)':>22s} {'smer':>5s} | {'presek n':>8s} {'presek: p (Holm)':>22s} {'smer':>5s} | opomba")
    for v, pa1, pa2 in zip(vrstice, h1, h2):
        fa, fb, a, b, W1, p1, b1, w1, t1, ni, W2, p2, b2, w2, t2 = v
        z1, z2 = (p1 == p1) and pa1 < 0.05, (p2 == p2) and pa2 < 0.05
        s1 = "A" if b1 > w1 else ("B" if w1 > b1 else "=")
        s2 = "A" if b2 > w2 else ("B" if w2 > b2 else "=")
        opomba = []
        if z1 and z2 and s1 != s2:
            opomba.append("OBRNITEV SMERI")
            obrnitve += 1
            podrobnosti.append({"comparison": f"{fa} vs {fb}", "n_intersection": ni,
                                "all343_winner": fa if s1 == "A" else fb, "all343_p_holm": pa1,
                                "intersection_winner": fa if s2 == "A" else fb, "intersection_p_holm": pa2})
        if z1 and not z2:
            opomba.append("izgubi značilnost")
            izgube += 1
        if (not z1) and z2:
            opomba.append("pridobi značilnost")
        stat[f"{fa} vs {fb}"] = {
            "rep_a": a, "rep_b": b,
            "all343": {"W": W1, "p_raw": p1, "p_holm": pa1, "a_better": b1, "a_worse": w1, "tie": t1, "significant_holm": bool(z1), "direction": s1},
            "both_have_signal": {"n": ni, "W": W2, "p_raw": p2, "p_holm": pa2, "a_better": b2, "a_worse": w2, "tie": t2, "significant_holm": bool(z2), "direction": s2},
            "flag": " + ".join(opomba),
        }
        print(f"  {fa + ' vs ' + fb:24s} | {p1:9.2e} ({pa1:8.2e}) {s1:>5s} | {ni:8d} {p2:9.2e} ({pa2:8.2e}) {s2:>5s} | {' + '.join(opomba)}")
    print(f"  obrnitev smeri {obrnitve}/{len(pari)}, izgubi značilnost {izgube}/{len(pari)} "
          "(primerjava na preseku je primarna, na vseh napakah vključuje umetne range tehnik brez signala)")

    print("\n[E': občutljivost na izbiro predstavnika družine (presek, surovi p)]")
    obcutljivost = {}
    for fam, ts in N.DRUZINE.items():
        for alt in ts:
            if len(ts) < 2 or alt == predstavniki[fam]:
                continue
            spremembe = []
            for fa, fb in pari:
                if fam not in (fa, fb):
                    continue
                a = alt if fa == fam else predstavniki[fa]
                b = alt if fb == fam else predstavniki[fb]
                presek = sorted(strogo[a] & strogo[b])
                _, pv, _, _, _ = test(rangi[a], rangi[b], presek)
                osnova = stat[f"{fa} vs {fb}"]["both_have_signal"]["p_raw"]
                if pv == pv and osnova == osnova and (pv < 0.05) != (osnova < 0.05):
                    spremembe.append(f"{fa} vs {fb} (p {osnova:.3g} -> {pv:.3g})")
            obcutljivost[f"{fam}: {predstavniki[fam]} -> {alt}"] = spremembe
            print(f"  {fam:11s} {predstavniki[fam]:20s} -> {alt:20s}: " + (", ".join(spremembe) if spremembe else "brez spremembe sklepov"))

    povzetek = {
        "config": {"n_faults": n_f, "n_methods": len(df), "n_faulty": int(df["faulty"].sum()), "families": N.DRUZINE,
                   "metric": "rang prve krive metode po pravilu povprečnega ranga, NS = [1, 3, 5, 10]",
                   "wilcoxon": "dvostranski, zero_method = wilcox, asimptotični. Holm v treh ločenih družinah: "
                               "tehnike proti Ochiai (m = 8), predstavniki na vseh napakah (m = 10), predstavniki na preseku (m = 10)"},
        "trivial_faults": {"n": len(trivialne), "ids": sorted(trivialne)},
        "techniques": rez,
        "families": druzine_rez,
        "applicability": {t: {"loose_nonzero": len(ohlapno[t]), "strict_variation": len(strogo[t])} for t in N.TEHNIKE},
        "complementarity": {
            "solved_top1": {t: len(resene[t]) for t in N.TEHNIKE},
            "solved_top1_nontrivial": {t: len(resene[t] - trivialne) for t in N.TEHNIKE},
            "solved_only_by": {t: sorted(samo[t]) for t in N.TEHNIKE},
            "oracle_top1_UPPER_BOUND": len(zgornja_meja),
            "best_single": {"technique": najboljsa, "top1": len(resene[najboljsa])},
            "realistic_union_ochiai_dstar": unija_sbfl,
            "achieved_learned_combination_05c_MLP_H9_top1": dosezeno,
        },
        "statistics_family_representatives": stat,
        "statistics_reversals": {"direction_reversals": obrnitve, "lost_significance": izgube, "reversal_details": podrobnosti},
        "representative_sensitivity": obcutljivost,
        "opombe": [
            "Metrik tehnike na njeni podmnožici napak ne primerjajte z metrikami druge tehnike na drugi podmnožici. "
            "Berite stolpec matched_ochiai (Ochiai na istih napakah).",
            "Razlika povprečij (MFR) ni dokaz. Sodba temelji na parnem testu, mediani in Top-1.",
            "Zgornja meja (oracle) predpostavlja popoln izbiralnik tehnike in ni dosežen rezultat.",
            "Predstavnik družine je izbran po MFR na istih podatkih, zato so vrednosti p optimistične. "
            "Napake tudi niso neodvisne (pet projektov).",
            "Največja normirana ocena znotraj družine je naša konstrukcija, ne objavljena tehnika.",
        ],
    }
    with open(N.pot(N.DRUZINE_REZULTATI), "w", encoding="utf-8") as f:
        json.dump(povzetek, f, ensure_ascii=False, indent=2)

    with open(N.pot(N.DRUZINE_TXT), "w", encoding="utf-8") as f:
        f.write(f"Primerjava tehnik in družin, {n_f} napak, nivo metod, metrika kot v mlp_llm.py\n\n")
        f.write(f"{'tehnika':22s} {'družina':11s} {'upor. strogo':>12s} " + " ".join(f"{'T' + str(n):>5s}" for n in N.NS) + f" {'MFR':>8s} {'MAR':>8s}\n")
        for t in N.TEHNIKE:
            a = rez[t]["all"]
            f.write(f"{t:22s} {druzina[t]:11s} {rez[t]['applicable_strict']:12d} " + " ".join(f"{a['top'][n]:5d}" for n in N.NS) + f" {a['mfr']:8.2f} {a['mar']:8.2f}\n")
        f.write("\nTehnika tam, kjer ima signal, proti Ochiai na istih napakah (parni test):\n")
        f.write(f"{'tehnika':22s} {'n':>4s} {'MFR t':>7s} {'MFR O':>7s} {'med t':>7s} {'med O':>7s} {'T1 t':>5s} {'T1 O':>5s} {'p(Holm)':>9s}  sodba\n")
        for t in N.TEHNIKE:
            s = rez[t]["signal_only"]
            pv = s["paired_vs_ochiai"]
            f.write(f"{t:22s} {pv['n']:4d} {s['mfr']:7.2f} {s['matched_ochiai']['mfr']:7.2f} {s['median_first_rank']:7.1f} "
                    f"{s['matched_ochiai']['median_first_rank']:7.1f} {s['top'][1]:5d} {s['matched_ochiai']['top'][1]:5d} "
                    f"{pv['p_holm']:9.3g}  {s['verdict_paired']}\n")
        f.write(f"\nZgornja meja (popoln izbiralnik): {len(zgornja_meja)}/{n_f}, najboljša posamezna {najboljsa} {len(resene[najboljsa])}, "
                f"Ochiai ali DStar {unija_sbfl}, dosežena naučena kombinacija (mlp_llm.py, MLP H9) {dosezeno}\n")
        f.write(f"Predstavniki družin: obrnitev smeri {obrnitve}/{len(pari)}, izgubi značilnost {izgube}/{len(pari)}.\n")
    print(f"\n[Shranjeno] {N.DRUZINE_REZULTATI}, {N.DRUZINE_TXT}")
    p.zakljuci()


if __name__ == "__main__":
    main()
