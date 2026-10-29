#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tabele šestega poglavja naloge (Markdown, slovenski zapis števil) iz shranjenih rezultatov.

Vsaka celica pride iz datotek JSON skript v mapah modeli in analize. Števila so zapisana z decimalno vejico,
vrednosti p pod 0,001 pa v obliki a · 10^b. Potrebuje samo standardno knjižnico.

Zagon:  python porocila/tabele_poglavje6.py
        (15. skripta v cevovodu, vrstni red je v README)
Vhod:   izhod/modeli/mlp_llm_rezultati.json, analize/polni_pogled.json, modeli/ranksvm_h9_rezultati.json,
        modeli/mlp_lopo_rezultati.json, modeli/ranksvm_llm_rezultati.json, analize/primerjava_druzin.json
Izhod:  izhod/porocila/tabele_poglavje6.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # koren repozitorija (nastavitve.py, skupno/)

import json
import math
import os

import nastavitve as N


def beri(ime):
    with open(N.pot(ime), encoding="utf-8") as f:
        return json.load(f)


SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
VRSTICE = []


def dec(x, nd=2):
    return f"{x:.{nd}f}".replace(".", ",")


def tis(n):
    """Celo število s piko za tisočice."""
    return f"{int(n):,}".replace(",", ".")


def wst(w):
    """Wilcoxonova statistika W: celo število s tisočicami ali z eno decimalko."""
    return tis(w) if float(w).is_integer() else f"{int(w):,}".replace(",", ".") + "," + f"{w:.1f}".split(".")[1]


def pv(p, sig=2, fiksno=None):
    """p na dve veljavni mesti, pod 0,001 v zapisu a · 10^b."""
    if fiksno is not None:
        return f"{p:.{fiksno}f}".replace(".", ",")
    if p >= 0.001:
        mest = sig - 1 - math.floor(math.log10(p))
        s = f"{p:.{mest}f}"
        return "1,0" if float(s) >= 1.0 else s.replace(".", ",")
    m, e = f"{p:.{sig - 1}e}".split("e")
    return f"{m.replace('.', ',')} · 10{str(int(e)).translate(SUP)}"


def med(x):
    return dec(x, 2) if abs(x * 2 - round(x * 2)) > 1e-9 else dec(x, 1)


def tri(d):
    return f"{d['a_better']} / {d['a_worse']} / {d['tie']}"


def tabela(naslov, glava, vrstice, poravnava=None):
    VRSTICE.append(f"\n<!-- {naslov} -->")
    VRSTICE.append("| " + " | ".join(glava) + " |")
    VRSTICE.append("|" + "|".join(poravnava or (["---"] + ["---:"] * (len(glava) - 1))) + "|")
    for v in vrstice:
        VRSTICE.append("| " + " | ".join(str(z) for z in v) + " |")


IME = {"dstar": "DStar", "ochiai": "Ochiai", "metallaxis": "Metallaxis", "muse": "MUSE", "stacktrace": "sled sklada",
       "slicing_count": "slicing_count (frekvenca)", "slicing_intersection": "slicing_intersection (presek)",
       "slicing": "slicing (unija)", "predicateswitching": "menjava predikatov"}
DRUZ = {"SBFL": "SBFL", "MBFL": "MBFL", "SledSklada": "sled sklada", "Rezanje": "rezanje", "PS": "PS"}
PROJEKT = {"chart": "Chart", "closure": "Closure", "lang": "Lang", "math": "Math", "time": "Time"}


def main():
    c08, d09, s10, l11, w12, t14 = (beri(N.MLP_LLM_REZULTATI), beri(N.POLNI_POGLED_REZULTATI), beri(N.RANKSVM_REZULTATI),
                                    beri(N.LOPO_REZULTATI), beri(N.RANKSVM_LLM_REZULTATI), beri(N.DRUZINE_REZULTATI))
    tk = t14["techniques"]

    # Tabela 6.1: posamezne tehnike
    red = sorted(tk, key=lambda k: -tk[k]["all"]["top"]["1"])
    tabela("TABELA 6.1", ["Tehnika", "Družina", "Uporabnost (strogo)", "Top-1 (netrivialne)", "Top-3", "Top-5", "Top-10", "MFR", "MAR"],
           [[IME[k], DRUZ[tk[k]["family"]], tk[k]["applicable_strict"], f"{tk[k]['all']['top']['1']} ({tk[k]['top1_nontrivial']})",
             tk[k]["all"]["top"]["3"], tk[k]["all"]["top"]["5"], tk[k]["all"]["top"]["10"], dec(tk[k]["all"]["mfr"]), dec(tk[k]["all"]["mar"])]
            for k in red], ["---", "---"] + ["---:"] * 7)

    # Tabela 6.2: tehnika proti Ochiai na isti podmnožici
    SODBA = {"NI ZNAČILNE RAZLIKE": "ni značilne razlike", "OCHIAI ZNAČILNO BOLJŠI": "Ochiai značilno boljši",
             "TEHNIKA ZNAČILNO BOLJŠA": "tehnika značilno boljša"}
    vr = []
    for k in ["dstar", "metallaxis", "muse", "stacktrace", "slicing", "slicing_count", "slicing_intersection", "predicateswitching"]:
        s = tk[k]["signal_only"]
        mo, pp = s["matched_ochiai"], s["paired_vs_ochiai"]
        vr.append([IME[k], s["n_faults"], f"{dec(s['mfr'])} / {dec(mo['mfr'])}", f"{med(s['median_first_rank'])} / {med(mo['median_first_rank'])}",
                   f"{s['top']['1']} / {mo['top']['1']}", f"{pp['tech_better']} / {pp['tech_worse']} / {pp['tie']} (n = {pp['n']})",
                   pv(pp["p_holm"]), SODBA[s["verdict_paired"]]])
    tabela("TABELA 6.2", ["Tehnika", "n (podmnožica)", "MFR: tehnika / Ochiai", "Mediana prvega ranga: tehnika / Ochiai",
                          "Top-1: tehnika / Ochiai", "Zmage / porazi / vezi (parni n)", "Holm p (družina 8)", "Sodba"], vr,
           ["---"] + ["---:"] * 6 + ["---"])

    # Tabela 6.3: predstavniki družin
    ZAST = {"izgubi značilnost": "izgubi značilnost", "OBRNITEV SMERI": "OBRNITEV SMERI", "": "—",
            "OBRNITEV SMERI + izgubi značilnost": "OBRNITEV SMERI", "pridobi značilnost": "pridobi značilnost"}
    vr = []
    for par, v in t14["statistics_family_representatives"].items():
        a, b = par.split(" vs ")
        A, B = DRUZ[a], DRUZ[b]

        def stran(x):
            zn = "" if x["significant_holm"] else " (n. z.)"
            return [wst(x["W"]), pv(x["p_holm"]) + zn, f"{x['a_better']} / {x['a_worse']} / {x['tie']}", A if x["direction"] == "A" else B]
        vse, pre = v["all343"], v["both_have_signal"]
        vr.append([f"{A} – {B}", f"{IME[v['rep_a']].split(' (')[0]} – {IME[v['rep_b']].split(' (')[0]}"] + stran(vse) + [pre["n"]] + stran(pre)
                  + [ZAST.get(v["flag"], v["flag"])])
    tabela("TABELA 6.3", ["Par (A – B)", "Predstavnika", "Vseh 343: W", "Holm p", "zmage / porazi / vezi (za A)", "smer v prid",
                          "Presek: n", "W", "Holm p", "zmage / porazi / vezi (za A)", "smer v prid", "Opozorilo"], vr,
           ["---", "---"] + ["---:", "---:", "---:", "---"] + ["---:", "---:", "---:", "---:", "---"] + ["---"])
    rv = t14["statistics_reversals"]
    VRSTICE.append(f"<!-- 6.3 kontrola: izgubi značilnost = {rv['lost_significance']}, obrnitev smeri = {rv['direction_reversals']} -->")

    # Tabela 6.4: komplementarnost
    cp = t14["complementarity"]
    un = {k: len(v) for k, v in cp["solved_only_by"].items() if v}
    dosezeno = cp["achieved_learned_combination_05c_MLP_H9_top1"]
    vr = [[f"unikatno rešene napake: {IME[k].split(' (')[0]}", n] for k, n in sorted(un.items(), key=lambda kv: -kv[1])]
    vr += [["unikatno rešene napake skupaj", sum(un.values())],
           ["Ochiai ∪ DStar (kombiniranje znotraj SBFL)", cp["realistic_union_ochiai_dstar"]],
           [f"najboljša posamezna tehnika ({IME[cp['best_single']['technique']]})", cp["best_single"]["top1"]],
           ["zgornja meja popolnega izbiralnika (ne rezultat)", cp["oracle_top1_UPPER_BOUND"]],
           ["dosežena naučena kombinacija (razdelek 6.3)", dosezeno],
           ["delež zgornje meje", f"{dosezeno} / {cp['oracle_top1_UPPER_BOUND']} ≈ {round(100 * dosezeno / cp['oracle_top1_UPPER_BOUND'])} %"]]
    tabela("TABELA 6.4", ["Količina (Top-1)", "Vrednost"], vr)

    # Tabela 6.5: polni in okrnjeni pogled
    M5 = [("LLM sam", "LLM sam"), ("H9", "klasična kombinacija H9"), ("H9+LLM", "H9+LLM"), ("kaskada", "kaskada")]
    vr = []
    for kljuc, oznaka in [("polni_pogled", "polni pogled"), ("okrnjeni_pogled", "okrnjeni pogled")]:
        t = d09["tabele"][kljuc]
        for mk, mi in M5:
            m = t["models"][mk]
            vr.append([f"{oznaka} (n = {t['n_faults']}, pokritost izbora {dec(100 * t['coverage'], 1) if t['coverage'] < 1 else '100'} %)",
                       mi, m["top"]["1"], m["top"]["3"], m["top"]["5"], m["top"]["10"], dec(m["mfr"])])
    tabela("TABELA 6.5", ["Pogled", "Model", "Top-1", "Top-3", "Top-5", "Top-10", "MFR"], vr, ["---", "---"] + ["---:"] * 5)

    # Tabela 6.6: glavna tabela modelov
    def vrst(ime, m):
        return [ime, m["top"]["1"], m["top"]["3"], m["top"]["5"], m["top"]["10"], dec(m["mfr"]), dec(m["mar"])]
    A = c08["all"]
    tabela("TABELA 6.6", ["Model", "Top-1", "Top-3", "Top-5", "Top-10", "MFR", "MAR"],
           [vrst("Ochiai", A["Ochiai"]), vrst("rankSVM (surove značilke)", s10["results"]["rankSVM (surove, Zou)"]),
            vrst("MLP H9 (kontrola)", A["MLP H9 (kontrola)"]), vrst("**MLP H9+LLM**", A["MLP H9+LLM"]),
            vrst("MLP H9 s premešanimi ocenami LLM (kontrolni preizkus, 6.4)", A["MLP H9+premešan (ablacija)"]),
            vrst("kaskada LLM→H9 (brez dodatnega učenja, 6.5)", A["KASKADA LLM→H9 (brez učenja)"]),
            vrst("LLM sam (ožji izbor + preureditev)", A["LLM (izbor+preureditev)"])])

    # Tabela 6.7: samo napake s krivo metodo z oceno
    C = c08["covered_only"]
    tabela(f"TABELA 6.7 (n = {c08['covered_faults']})", ["Model", "MFR", "Top-1"],
           [[i, dec(C[k]["mfr"]), C[k]["top"]["1"]] for i, k in
            [("LLM sam", "LLM (izbor+preureditev)"), ("kaskada", "KASKADA LLM→H9 (brez učenja)"), ("**H9+LLM**", "MLP H9+LLM"),
             ("H9", "MLP H9 (kontrola)"), ("H9 s premešanimi ocenami", "MLP H9+premešan (ablacija)"), ("Ochiai", "Ochiai")]])
    VRSTICE.append(f"<!-- 6.7 MAR samo pokrite: H9+LLM {dec(C['MLP H9+LLM']['mar'])}, kaskada {dec(C['KASKADA LLM→H9 (brez učenja)']['mar'])}, "
                   f"LLM sam {dec(C['LLM (izbor+preureditev)']['mar'])} -->")

    # Tabela 6.8: rankSVM z LLM
    R12 = w12["results"]
    vr = []
    for i, k in [("Ochiai", "Ochiai"), ("rankSVM, surove značilke", "rankSVM (surove, Zou)"), ("rankSVM, min-max", "rankSVM (min-max)"),
                 ("**rankSVM+LLM, surove značilke**", "rankSVM H9+LLM (surove)"), ("rankSVM+LLM, min-max", "rankSVM H9+LLM (min-max)"),
                 ("MLP H9", "MLP H9 (kontrola)"), ("MLP H9+LLM", "MLP H9+LLM")]:
        m, t = R12[k], R12[k]["top1_tie"]
        vr.append(vrst(i, m) + [f"{t['min']} / {t['avg']} / {t['max']}"])
    tabela("TABELA 6.8", ["Model", "Top-1", "Top-3", "Top-5", "Top-10", "MFR", "MAR", "Top-1: optimistična / povprečna / pesimistična"], vr)
    W = w12["weights_mean_sd_per_fold"]
    if W:
        VRSTICE.append(f"<!-- utež LLM: surove {dec(W['rawllm']['llm']['mean'], 4)} ± {dec(W['rawllm']['llm']['sd'], 4)}, "
                       f"min-max {dec(W['normllm']['llm']['mean'], 3)} ± {dec(W['normllm']['llm']['sd'], 4)} -->")

    # Tabeli 6.9: LOPO po projektih in združeno
    pp = l11["per_project_primary_seed"]
    tabela("TABELA 6.9a (LOPO po projektih, primarno seme)", ["Projekt (napak)", "MFR: H9 → H9+LLM", "Top-1: H9 → H9+LLM"],
           [[f"{PROJEKT[k]} ({pp[k]['MLP H9']['n']})", f"{dec(pp[k]['MLP H9']['mfr'])} → {dec(pp[k]['MLP H9+LLM']['mfr'])}",
             f"{pp[k]['MLP H9']['top']['1']} → {pp[k]['MLP H9+LLM']['top']['1']}"] for k in PROJEKT])
    po = l11["pooled_primary_seed"]
    tabela("TABELA 6.9b (LOPO združeno, primarno seme)", ["Model", "Top-1", "MFR"],
           [[i, po[k]["top"]["1"], dec(po[k]["mfr"])] for i, k in
            [("Ochiai (nenaučena referenca)", "Ochiai (netrenirana referenca)"), ("H9", "MLP H9 (LOPO)"), ("H9+LLM", "MLP H9+LLM (LOPO)")]])

    # Tabela 6.10: LOPO proti prečnemu preverjanju po napakah
    ss, t10f = l11["seed_stats"]["mfr"], l11["context_10fold_05c"]
    vr = []
    for i, k, kp in [("H9", "MLP H9", "MLP H9 (LOPO)"), ("H9+LLM", "MLP H9+LLM", "MLP H9+LLM (LOPO)")]:
        a, b = t10f[k]["mfr"], po[kp]["mfr"]
        vr.append([i, dec(a), dec(b), f"+{round(100 * (b / a - 1))} %", f"{dec(min(ss[k]['values']), 1)} … {dec(max(ss[k]['values']), 1)}",
                   f"{dec(ss[k]['mean'])} ± {dec(ss[k]['sd'])}"])
    tabela("TABELA 6.10", ["Model", "MFR, desetkratna CV", "MFR, LOPO", "Sprememba", "MFR po semenih (LOPO): razpon", "povprečje ± st. odklon"], vr)
    dm, dt = l11["seed_stats"]["delta_mfr"], l11["seed_stats"]["delta_top1"]
    VRSTICE.append(f"<!-- razlika MFR: primarno seme {dec(dm['values'][0])}, povprečje {dec(dm['mean'])} ± {dec(dm['sd'])}, "
                   f"vrednosti {[dec(v) for v in dm['values']]}, izboljšanih semen {dm['improved_seeds']} | razlika Top-1: {dt['values']} "
                   f"povprečje {dec(dt['mean'], 1)} ± {dec(dt['sd'], 1)}, izboljšanih {dt['improved_seeds']} -->")
    fl, pl = l11["wilcoxon_fault_level"], l11["wilcoxon_project_level"]
    VRSTICE.append("<!-- LOPO p (napake): " + ", ".join(f"{s}: {pv(v['p'])} ({tri(v)})" for s, v in fl.items()) + " -->")
    VRSTICE.append("<!-- LOPO p (projekti): " + ", ".join(f"{s}: {v['p']} ({v['improved_projects']}/5)" for s, v in pl.items()) + " -->")

    # Tabela 6.12: MFR po projektih, prečno preverjanje po napakah
    pr = c08["per_project"]
    tabela("TABELA 6.12", ["Projekt (napak)", "MFR: H9 → H9+LLM", "Top-1: H9 → H9+LLM"],
           [[f"{PROJEKT[k]} ({pr[k]['n_faults']})", f"{dec(pr[k]['mfr_h9'])} → {dec(pr[k]['mfr_h9llm'])}",
             f"{pr[k]['top1_h9']} → {pr[k]['top1_h9llm']}"] for k in PROJEKT])

    # Tabela 6.13: vsi statistični testi
    wf, wt, w10, w12w, w09 = c08["wilcoxon_fault"], c08["wilcoxon_tests"], s10["wilcoxon"], w12["wilcoxon"], d09["testi_fullview"]
    f18 = fl[str(N.SEMENA[0])]
    vr = [
        ["H9+LLM proti H9 (glavna primerjava, desetkratna CV)", "enostranski", pv(wf["p"]), tri(wf)],
        ["isto na ravni projektov (desetkratna CV; n = 5 parov MFR)", "enostranski", str(c08["wilcoxon_project_p"]).replace(".", ","), "—"],
        ["H9+LLM proti H9 (LOPO, raven napak, primarno seme)", "enostranski", pv(f18["p"]), tri(f18)],
        ["isto na ravni projektov (LOPO, primarno seme)", "enostranski", f"{pl[str(N.SEMENA[0])]['p']:.3f}".replace(".", ","), "—"],
        ["H9 proti H9 s premešanimi ocenami (kontrolni preizkus)", "enostranski", pv(wt["H9_vs_H9shuf"]["p"]), tri(wt["H9_vs_H9shuf"])],
        ["H9+LLM proti LLM samemu", "enostranski", pv(wt["H9LLM_vs_LLM_alone"]["p"]), tri(wt["H9LLM_vs_LLM_alone"])],
        ["H9+LLM proti kaskadi", "enostranski", pv(wt["H9LLM_vs_cascade"]["p"]) + " (ni značilno)", tri(wt["H9LLM_vs_cascade"])],
        ["rankSVM proti Ochiai", "enostranski", pv(w10["rankSVM (surove, Zou) vs Ochiai"]["p"]), tri(w10["rankSVM (surove, Zou) vs Ochiai"])],
        ["H9+LLM proti rankSVM", "enostranski", pv(w10["MLP H9+LLM vs rankSVM (surove, Zou)"]["p"]), tri(w10["MLP H9+LLM vs rankSVM (surove, Zou)"])],
        ["MLP H9 proti rankSVM (brez LLM)", "dvostranski", pv(w10["MLP H9 (kontrola) vs rankSVM (surove, Zou)"]["p"]), tri(w10["MLP H9 (kontrola) vs rankSVM (surove, Zou)"])],
        ["rankSVM+LLM proti rankSVM (surove značilke)", "enostranski", pv(w12w["rankSVM H9+LLM (surove) vs rankSVM (surove, Zou)"]["p"]), tri(w12w["rankSVM H9+LLM (surove) vs rankSVM (surove, Zou)"])],
        ["rankSVM+LLM proti rankSVM (min-max)", "enostranski", pv(w12w["rankSVM H9+LLM (min-max) vs rankSVM (min-max)"]["p"]), tri(w12w["rankSVM H9+LLM (min-max) vs rankSVM (min-max)"])],
        ["rankSVM+LLM proti MLP H9+LLM (surove značilke)", "dvostranski", pv(w12w["rankSVM H9+LLM (surove) vs MLP H9+LLM"]["p"]), tri(w12w["rankSVM H9+LLM (surove) vs MLP H9+LLM"])],
        ["rankSVM+LLM proti MLP H9+LLM (min-max)", "dvostranski", pv(w12w["rankSVM H9+LLM (min-max) vs MLP H9+LLM"]["p"]), tri(w12w["rankSVM H9+LLM (min-max) vs MLP H9+LLM"])],
        ["rankSVM+LLM: surove značilke proti min-max (isti model)", "dvostranski", pv(w12w["rankSVM H9+LLM (surove) vs rankSVM H9+LLM (min-max)"]["p"], fiksno=4), tri(w12w["rankSVM H9+LLM (surove) vs rankSVM H9+LLM (min-max)"])],
        ["LLM sam proti H9 (polni pogled, n = 151)", "enostranski", pv(w09["LLM sam vs H9"]["p"]), tri(w09["LLM sam vs H9"])],
        ["H9+LLM proti LLM samemu (polni pogled, n = 151)", "enostranski", pv(w09["H9+LLM vs LLM sam"]["p"]) + " (test skoraj brez moči)", tri(w09["H9+LLM vs LLM sam"])],
    ]
    tabela("TABELA 6.13", ["Primerjava", "Test", "p", "Zmage / porazi / vezi"], vr, ["---", "---", "---:", "---:"])
    VRSTICE.append(f"<!-- LOPO raven napak, razpon p čez semena: {pv(min(v['p'] for v in fl.values()))} … {pv(max(v['p'] for v in fl.values()))} -->")

    # Tabela 6.14: robustnost na semena
    tabela("TABELA 6.14", ["Protokol", "Količina", "Povprečje ± st. odklon", "Izboljšanje po semenih"],
           [["desetkratna CV", "Δ Top-1 (H9+LLM − H9)", f"+{dec(c08['seed_delta_top1_mean'], 1)} ± {dec(c08['seed_delta_top1_sd'], 1)}", "5 od 5"],
            ["desetkratna CV", "Δ MFR", f"−{dec(abs(c08['seed_delta_mfr_mean']))} ± {dec(c08['seed_delta_mfr_sd'])}", "5 od 5"],
            ["LOPO", "Δ Top-1 (H9+LLM − H9)", f"+{dec(dt['mean'], 1)} ± {dec(dt['sd'], 1)}", f"{dt['improved_seeds']} od 5 (vrednosti {min(dt['values'])}–{max(dt['values'])})"],
            ["LOPO", "Δ MFR", f"−{dec(abs(dm['mean']))} ± {dec(dm['sd'])}", f"{dm['improved_seeds']} od 5 (pri enem semenu se predznak obrne)"]],
           ["---", "---", "---:", "---"])

    # Tabela 6.15: Top-1 po obravnavi enakih ocen
    vr = []
    for i, k in [("Ochiai", "Ochiai"), ("rankSVM, surove značilke", "rankSVM (surove, Zou)"), ("rankSVM, min-max", "rankSVM (min-max)"),
                 ("MLP H9", "MLP H9 (kontrola)"), ("**MLP H9+LLM**", "MLP H9+LLM"), ("rankSVM+LLM, surove značilke", "rankSVM H9+LLM (surove)"),
                 ("rankSVM+LLM, min-max", "rankSVM H9+LLM (min-max)")]:
        t = R12[k]["top1_tie"]
        vr.append([i, t["min"], t["avg"], t["max"]])
    tabela("TABELA 6.15", ["Model", "optimistična (najmanjši rang)", "povprečna", "pesimistična (največji rang)"], vr)

    # skrajšani različici tabel po projektih z združeno vrstico
    vr = [[f"{PROJEKT[k]} ({pr[k]['n_faults']})", f"{dec(pr[k]['mfr_h9'])} → {dec(pr[k]['mfr_h9llm'])}", f"{pr[k]['top1_h9']} → {pr[k]['top1_h9llm']}"] for k in PROJEKT]
    vr.append([f"**Združeno ({c08['n_faults']})**", f"{dec(A['MLP H9 (kontrola)']['mfr'])} → {dec(A['MLP H9+LLM']['mfr'])}",
               f"{A['MLP H9 (kontrola)']['top']['1']} → {A['MLP H9+LLM']['top']['1']}"])
    tabela("TABELA 6.12z (desetkratna CV po projektih + združeno)", ["Projekt (napak)", "MFR: H9 → H9+LLM", "Top-1: H9 → H9+LLM"], vr)
    vr = [[f"{PROJEKT[k]} ({pp[k]['MLP H9']['n']})", f"{dec(pp[k]['MLP H9']['mfr'])} → {dec(pp[k]['MLP H9+LLM']['mfr'])}",
           f"{pp[k]['MLP H9']['top']['1']} → {pp[k]['MLP H9+LLM']['top']['1']}"] for k in PROJEKT]
    vr.append([f"**Združeno ({c08['n_faults']})**", f"{dec(po['MLP H9 (LOPO)']['mfr'])} → {dec(po['MLP H9+LLM (LOPO)']['mfr'])}",
               f"{po['MLP H9 (LOPO)']['top']['1']} → {po['MLP H9+LLM (LOPO)']['top']['1']}"])
    tabela("TABELA 6.9z (LOPO po projektih + združeno, primarno seme)", ["Projekt (napak)", "MFR: H9 → H9+LLM", "Top-1: H9 → H9+LLM"], vr)

    pot = N.pot(N.TABELE_MD)
    with open(pot, "w", encoding="utf-8") as f:
        f.write("# Tabele šestega poglavja (samodejno iz rezultatov, porocila/tabele_poglavje6.py)\n")
        f.write("\n".join(VRSTICE) + "\n")
    print(f"[Shranjeno] {pot} ({len([v for v in VRSTICE if v.strip().startswith('<!-- TABELA')])} tabel)")


if __name__ == "__main__":
    main()
