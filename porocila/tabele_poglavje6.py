#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tabele rezultatov (šesto poglavje in dodatek A) v Markdownu iz shranjenih rezultatov

Števila v celicah pridejo iz datotek JSON skript v mapah modeli in analize. Števila imajo decimalno
vejico, vrednosti p pod 0,001 so v obliki a · 10^b.

Vhod:   izhod/modeli/mlp_llm_rezultati.json, modeli/ranksvm_h9_rezultati.json,
        modeli/mlp_lopo_rezultati.json, modeli/ranksvm_llm_rezultati.json,
        analize/polni_pogled.json, analize/primerjava_druzin.json
Izhod:  izhod/porocila/tabele_poglavje6.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # koren repozitorija

import math

import nastavitve as N
from skupno.podatki import beri_json


SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")   # nadpisane števke za eksponent
VRSTICE = []


def dec(x, nd=2):
    return f"{x:.{nd}f}".replace(".", ",")


def znak(x):
    """Predznak za izpis, minus je tipografski"""
    return "+" if x >= 0 else "−"


def tis(n):
    """Celo število s piko za tisočice"""
    return f"{int(n):,}".replace(",", ".")


def wst(w):
    """Wilcoxonova statistika W s piko za tisočice, necela še z eno decimalko"""
    if float(w).is_integer():
        return tis(w)
    return tis(w) + "," + f"{w:.1f}".split(".")[1]


def pv(p):
    """p na dve veljavni mesti z decimalno vejico, pod 0,001 v zapisu a · 10^b"""
    if p >= 0.001:
        mest = 1 - math.floor(math.log10(p))   # decimalk za dve veljavni mesti
        s = f"{p:.{mest}f}"
        return "1,0" if float(s) >= 1.0 else s.replace(".", ",")
    m, e = f"{p:.1e}".split("e")
    return f"{m.replace('.', ',')} · 10{str(int(e)).translate(SUP)}"


def med(x):
    """Mediana ranga: ena decimalka, če je večkratnik 0,5, sicer dve"""
    return dec(x, 2) if abs(x * 2 - round(x * 2)) > 1e-9 else dec(x, 1)


def tri(d):
    return f"{d['a_better']} / {d['a_worse']} / {d['tie']}"


def tabela(naslov, glava, vrstice, poravnava=None):
    """Tabelo doda v VRSTICE, brez poravnave je prvi stolpec levo, ostali desno"""
    VRSTICE.append(f"\n<!-- {naslov} -->")
    VRSTICE.append("| " + " | ".join(glava) + " |")
    VRSTICE.append("|" + "|".join(poravnava or (["---"] + ["---:"] * (len(glava) - 1))) + "|")
    for v in vrstice:
        VRSTICE.append("| " + " | ".join(str(z) for z in v) + " |")


DRUZ = {"SBFL": "SBFL", "MBFL": "MBFL", "SledSklada": "sled sklada", "Rezanje": "rezanje", "PS": "PS"}
PROJEKT = {"chart": "Chart", "closure": "Closure", "lang": "Lang", "math": "Math", "time": "Time"}


def main():
    # rezultati modelov, polnega pogleda in primerjave družin
    mlp, pogled, ranksvm, lopo, ranksvm_llm, druzine = (
        beri_json(N.MLP_LLM_REZULTATI), beri_json(N.POLNI_POGLED_REZULTATI), beri_json(N.RANKSVM_REZULTATI),
        beri_json(N.LOPO_REZULTATI), beri_json(N.RANKSVM_LLM_REZULTATI), beri_json(N.DRUZINE_REZULTATI))
    tk = druzine["techniques"]

    # Tabela 6.1: posamezne tehnike
    red = sorted(tk, key=lambda k: -tk[k]["all"]["top"]["1"])
    tabela("Tabela 6.1: devet tehnik samostojno", ["Tehnika", "Družina", "Uporabnost (strogo)", "Top-1 (netrivialne)", "Top-3", "Top-5", "Top-10", "MFR", "MAR"],
           [[N.IMENA_TEHNIK[k], DRUZ[tk[k]["family"]], tk[k]["applicable_strict"], f"{tk[k]['all']['top']['1']} ({tk[k]['top1_nontrivial']})",
             tk[k]["all"]["top"]["3"], tk[k]["all"]["top"]["5"], tk[k]["all"]["top"]["10"], dec(tk[k]["all"]["mfr"]), dec(tk[k]["all"]["mar"])]
            for k in red], ["---", "---"] + ["---:"] * 7)

    # Tabela 6.2: tehnika proti Ochiai na isti podmnožici
    SODBA = {"NI ZNAČILNE RAZLIKE": "ni značilne razlike", "OCHIAI ZNAČILNO BOLJŠI": "Ochiai značilno boljši",
             "TEHNIKA ZNAČILNO BOLJŠA": "tehnika značilno boljša"}
    vr = []
    for k in ["dstar", "metallaxis", "muse", "stacktrace", "slicing", "slicing_count", "slicing_intersection", "predicateswitching"]:
        s = tk[k]["signal_only"]
        mo, pp = s["matched_ochiai"], s["paired_vs_ochiai"]
        vr.append([N.IMENA_TEHNIK[k], s["n_faults"], f"{dec(s['mfr'])} / {dec(mo['mfr'])}", f"{med(s['median_first_rank'])} / {med(mo['median_first_rank'])}",
                   f"{s['top']['1']} / {mo['top']['1']}", f"{pp['tech_better']} / {pp['tech_worse']} / {pp['tie']} (n = {pp['n']})",
                   pv(pp["p_holm"]), SODBA[s["verdict_paired"]]])
    tabela("Tabela 6.2: tehnika proti Ochiai na podmnožici s signalom", ["Tehnika", "n (podmnožica)", "MFR: tehnika / Ochiai", "Mediana prvega ranga: tehnika / Ochiai",
                          "Top-1: tehnika / Ochiai", "Zmage / porazi / vezi (parni n)", "Holm p (družina 8)", "Sodba"], vr,
           ["---"] + ["---:"] * 6 + ["---"])

    # Tabela A.1: predstavniki družin
    ZAST = {"izgubi značilnost": "izgubi značilnost", "OBRNITEV SMERI": "OBRNITEV SMERI", "": "—",
            "OBRNITEV SMERI + izgubi značilnost": "OBRNITEV SMERI", "pridobi značilnost": "pridobi značilnost"}

    def stran(x, A, B):
        zn = "" if x["significant_holm"] else " (n. z.)"
        return [wst(x["W"]), pv(x["p_holm"]) + zn, tri(x), A if x["direction"] == "A" else B]
    vr = []
    for par, v in druzine["statistics_family_representatives"].items():
        a, b = par.split(" vs ")
        A, B = DRUZ[a], DRUZ[b]
        vse, pre = v["all343"], v["both_have_signal"]
        vr.append([f"{A} – {B}", f"{N.IMENA_TEHNIK[v['rep_a']].split(' (')[0]} – {N.IMENA_TEHNIK[v['rep_b']].split(' (')[0]}"] + stran(vse, A, B) + [pre["n"]]
                  + stran(pre, A, B) + [ZAST.get(v["flag"], v["flag"])])
    tabela("Tabela A.1: predstavniki družin", ["Par (A – B)", "Predstavnika", "Vseh 343: W", "Holm p", "zmage / porazi / vezi (za A)", "smer v prid",
                          "Presek: n", "W", "Holm p", "zmage / porazi / vezi (za A)", "smer v prid", "Opozorilo"], vr,
           ["---", "---"] + ["---:", "---:", "---:", "---"] + ["---:", "---:", "---:", "---:", "---"] + ["---"])
    rv = druzine["statistics_reversals"]
    VRSTICE.append(f"<!-- A.1 kontrola: izgubi značilnost = {rv['lost_significance']}, obrnitev smeri = {rv['direction_reversals']} -->")

    # komplementarnost, številke so v besedilu šestega poglavja
    cp = druzine["complementarity"]
    un = {k: len(v) for k, v in cp["solved_only_by"].items() if v}
    dosezeno = cp["achieved_learned_combination_MLP_H9_top1"]
    vr = [[f"unikatno rešene napake: {N.IMENA_TEHNIK[k].split(' (')[0]}", n] for k, n in sorted(un.items(), key=lambda kv: -kv[1])]
    vr += [["unikatno rešene napake skupaj", sum(un.values())],
           ["Ochiai ∪ DStar (kombiniranje znotraj SBFL)", cp["realistic_union_ochiai_dstar"]],
           [f"najboljša posamezna tehnika ({N.IMENA_TEHNIK[cp['best_single']['technique']]})", cp["best_single"]["top1"]],
           ["zgornja meja popolnega izbiralnika (ne rezultat)", cp["oracle_top1_UPPER_BOUND"]],
           ["dosežena naučena kombinacija (MLP H9)", dosezeno],
           ["delež zgornje meje", f"{dosezeno} / {cp['oracle_top1_UPPER_BOUND']} ≈ {round(100 * dosezeno / cp['oracle_top1_UPPER_BOUND'])} %"]]
    tabela("Komplementarnost tehnik (številke v besedilu)", ["Količina (Top-1)", "Vrednost"], vr)

    # Tabela 6.3: polni in okrnjeni pogled
    M5 = [("LLM sam", "LLM sam"), ("H9", "klasična kombinacija H9"), ("H9+LLM", "H9+LLM"), ("kaskada", "kaskada")]
    vr = []
    for kljuc, oznaka in [("polni_pogled", "polni pogled"), ("okrnjeni_pogled", "okrnjeni pogled")]:
        t = pogled["tabele"][kljuc]
        for mk, mi in M5:
            m = t["models"][mk]
            vr.append([f"{oznaka} (n = {t['n_faults']}, pokritost izbora {dec(100 * t['coverage'], 1) if t['coverage'] < 1 else '100'} %)",
                       mi, m["top"]["1"], m["top"]["3"], m["top"]["5"], m["top"]["10"], dec(m["mfr"])])
    tabela("Tabela 6.3: polni in okrnjeni pogled", ["Pogled", "Model", "Top-1", "Top-3", "Top-5", "Top-10", "MFR"], vr, ["---", "---"] + ["---:"] * 5)

    # Tabela 6.4: osrednja primerjava modelov
    def vrst(ime, m):
        return [ime, m["top"]["1"], m["top"]["3"], m["top"]["5"], m["top"]["10"], dec(m["mfr"]), dec(m["mar"])]
    A = mlp["all"]
    tabela("Tabela 6.4: osrednja primerjava modelov", ["Model", "Top-1", "Top-3", "Top-5", "Top-10", "MFR", "MAR"],
           [vrst("Ochiai", A["Ochiai"]), vrst("rankSVM (surove značilke)", ranksvm["results"]["rankSVM (surove, Zou)"]),
            vrst("MLP H9 (kontrola)", A["MLP H9 (kontrola)"]), vrst("**MLP H9+LLM**", A["MLP H9+LLM"]),
            vrst("MLP H9 s premešanimi ocenami LLM (kontrolni preizkus)", A["MLP H9+premešan (ablacija)"]),
            vrst("kaskada LLM→H9 (brez dodatnega učenja)", A["KASKADA LLM→H9 (brez učenja)"]),
            vrst("LLM sam (ožji izbor + preureditev)", A["LLM (izbor+preureditev)"])])

    # Tabela 6.6: samo napake s krivo metodo z oceno
    C = mlp["covered_only"]
    tabela(f"Tabela 6.6: napake s krivo metodo z oceno (n = {mlp['covered_faults']})", ["Model", "MFR", "Top-1"],
           [[i, dec(C[k]["mfr"]), C[k]["top"]["1"]] for i, k in
            [("LLM sam", "LLM (izbor+preureditev)"), ("kaskada", "KASKADA LLM→H9 (brez učenja)"), ("**H9+LLM**", "MLP H9+LLM"),
             ("H9", "MLP H9 (kontrola)"), ("H9 s premešanimi ocenami", "MLP H9+premešan (ablacija)"), ("Ochiai", "Ochiai")]])
    VRSTICE.append(f"<!-- 6.6 MAR samo pokrite: H9+LLM {dec(C['MLP H9+LLM']['mar'])}, kaskada {dec(C['KASKADA LLM→H9 (brez učenja)']['mar'])}, "
                   f"LLM sam {dec(C['LLM (izbor+preureditev)']['mar'])} -->")

    # Tabela 6.7: rankSVM z LLM
    rez_ranksvm_llm = ranksvm_llm["results"]
    vr = []
    for i, k in [("Ochiai", "Ochiai"), ("rankSVM, surove značilke", "rankSVM (surove, Zou)"), ("rankSVM, min-max", "rankSVM (min-max)"),
                 ("**rankSVM+LLM, surove značilke**", "rankSVM H9+LLM (surove)"), ("rankSVM+LLM, min-max", "rankSVM H9+LLM (min-max)"),
                 ("MLP H9", "MLP H9 (kontrola)"), ("MLP H9+LLM", "MLP H9+LLM")]:
        m, t = rez_ranksvm_llm[k], rez_ranksvm_llm[k]["top1_tie"]
        vr.append(vrst(i, m) + [f"{t['min']} / {t['avg']} / {t['max']}"])
    tabela("Tabela 6.7: rankSVM in MLP z oceno LLM", ["Model", "Top-1", "Top-3", "Top-5", "Top-10", "MFR", "MAR", "Top-1: optimistična / povprečna / pesimistična"], vr)
    W = ranksvm_llm["weights_mean_sd_per_fold"]
    if W:
        VRSTICE.append(f"<!-- utež LLM: surove {dec(W['rawllm']['llm']['mean'], 4)} ± {dec(W['rawllm']['llm']['sd'], 4)}, "
                       f"min-max {dec(W['normllm']['llm']['mean'], 3)} ± {dec(W['normllm']['llm']['sd'], 4)} -->")

    # Tabela 6.8: LOPO po projektih, združeni rezultat je v besedilu
    pp = lopo["per_project_primary_seed"]
    vr_lopo = [[f"{PROJEKT[k]} ({pp[k]['MLP H9']['n']})", f"{dec(pp[k]['MLP H9']['mfr'])} → {dec(pp[k]['MLP H9+LLM']['mfr'])}",
                f"{pp[k]['MLP H9']['top']['1']} → {pp[k]['MLP H9+LLM']['top']['1']}"] for k in PROJEKT]
    tabela("Tabela 6.8: LOPO po projektih (primarno seme)", ["Projekt (napak)", "MFR: H9 → H9+LLM", "Top-1: H9 → H9+LLM"], vr_lopo)
    po = lopo["pooled_primary_seed"]
    tabela("LOPO združeno (primarno seme, številke v besedilu)", ["Model", "Top-1", "MFR"],
           [[i, po[k]["top"]["1"], dec(po[k]["mfr"])] for i, k in
            [("Ochiai (nenaučena referenca)", "Ochiai (netrenirana referenca)"), ("H9", "MLP H9 (LOPO)"), ("H9+LLM", "MLP H9+LLM (LOPO)")]])

    # LOPO proti prečnemu preverjanju po napakah, številke v besedilu
    ss, t10f = lopo["seed_stats"]["mfr"], lopo["context_10fold"]
    vr = []
    for i, k, kp in [("H9", "MLP H9", "MLP H9 (LOPO)"), ("H9+LLM", "MLP H9+LLM", "MLP H9+LLM (LOPO)")]:
        a, b = t10f[k]["mfr"], po[kp]["mfr"]
        sprem = round(100 * (b / a - 1))
        vr.append([i, dec(a), dec(b), f"{znak(sprem)}{abs(sprem)} %", f"{dec(min(ss[k]['values']), 1)} … {dec(max(ss[k]['values']), 1)}",
                   f"{dec(ss[k]['mean'])} ± {dec(ss[k]['sd'])}"])
    tabela("LOPO proti desetkratni CV (številke v besedilu)", ["Model", "MFR, desetkratna CV", "MFR, LOPO", "Sprememba", "MFR po semenih (LOPO): razpon", "povprečje ± st. odklon"], vr)
    dm, dt = lopo["seed_stats"]["delta_mfr"], lopo["seed_stats"]["delta_top1"]
    VRSTICE.append(f"<!-- razlika MFR: primarno seme {dec(dm['values'][0])}, povprečje {dec(dm['mean'])} ± {dec(dm['sd'])}, "
                   f"vrednosti {[dec(v) for v in dm['values']]}, izboljšanih semen {dm['improved_seeds']} | razlika Top-1: {dt['values']} "
                   f"povprečje {dec(dt['mean'], 1)} ± {dec(dt['sd'], 1)}, izboljšanih {dt['improved_seeds']} -->")
    fl, pl = lopo["wilcoxon_fault_level"], lopo["wilcoxon_project_level"]
    VRSTICE.append("<!-- LOPO p (napake): " + ", ".join(f"{s}: {pv(v['p'])} ({tri(v)})" for s, v in fl.items()) + " -->")
    VRSTICE.append("<!-- LOPO p (projekti): " + ", ".join(f"{s}: {v['p']} ({v['improved_projects']}/5)" for s, v in pl.items()) + " -->")

    # Tabela 6.5: MFR po projektih, prečno preverjanje po napakah
    pr = mlp["per_project"]
    vr_cv = [[f"{PROJEKT[k]} ({pr[k]['n_faults']})", f"{dec(pr[k]['mfr_h9'])} → {dec(pr[k]['mfr_h9llm'])}",
              f"{pr[k]['top1_h9']} → {pr[k]['top1_h9llm']}"] for k in PROJEKT]
    tabela("Tabela 6.5: po projektih (desetkratna CV)", ["Projekt (napak)", "MFR: H9 → H9+LLM", "Top-1: H9 → H9+LLM"], vr_cv)

    # Tabela A.2: vsi statistični testi
    wf, wt, w_ranksvm, w_ranksvm_llm, w_pogled = mlp["wilcoxon_fault"], mlp["wilcoxon_tests"], ranksvm["wilcoxon"], ranksvm_llm["wilcoxon"], pogled["testi_fullview"]
    seme = str(N.SEMENA[0])
    vr = [
        ["H9+LLM proti H9 (glavna primerjava, desetkratna CV)", "enostranski", pv(wf["p"]), tri(wf)],
        ["isto na ravni projektov (desetkratna CV; n = 5 parov MFR)", "enostranski", str(mlp["wilcoxon_project_p"]).replace(".", ","), "—"],
        ["H9+LLM proti H9 (LOPO, raven napak, primarno seme)", "enostranski", pv(fl[seme]["p"]), tri(fl[seme])],
        ["isto na ravni projektov (LOPO, primarno seme)", "enostranski", dec(pl[seme]["p"], 3), "—"],
        ["H9 proti H9 s premešanimi ocenami (kontrolni preizkus)", "enostranski", pv(wt["H9_vs_H9shuf"]["p"]), tri(wt["H9_vs_H9shuf"])],
        ["H9+LLM proti LLM samemu", "enostranski", pv(wt["H9LLM_vs_LLM_alone"]["p"]), tri(wt["H9LLM_vs_LLM_alone"])],
        ["H9+LLM proti kaskadi", "enostranski", pv(wt["H9LLM_vs_cascade"]["p"]) + (" (ni značilno)" if wt["H9LLM_vs_cascade"]["p"] >= 0.05 else ""), tri(wt["H9LLM_vs_cascade"])],
        ["rankSVM proti Ochiai", "enostranski", pv(w_ranksvm["rankSVM (surove, Zou) vs Ochiai"]["p"]), tri(w_ranksvm["rankSVM (surove, Zou) vs Ochiai"])],
        ["H9+LLM proti rankSVM", "enostranski", pv(w_ranksvm["MLP H9+LLM vs rankSVM (surove, Zou)"]["p"]), tri(w_ranksvm["MLP H9+LLM vs rankSVM (surove, Zou)"])],
        ["MLP H9 proti rankSVM (brez LLM)", "dvostranski", pv(w_ranksvm["MLP H9 (kontrola) vs rankSVM (surove, Zou)"]["p"]), tri(w_ranksvm["MLP H9 (kontrola) vs rankSVM (surove, Zou)"])],
        ["rankSVM+LLM proti rankSVM (surove značilke)", "enostranski", pv(w_ranksvm_llm["rankSVM H9+LLM (surove) vs rankSVM (surove, Zou)"]["p"]), tri(w_ranksvm_llm["rankSVM H9+LLM (surove) vs rankSVM (surove, Zou)"])],
        ["rankSVM+LLM proti rankSVM (min-max)", "enostranski", pv(w_ranksvm_llm["rankSVM H9+LLM (min-max) vs rankSVM (min-max)"]["p"]), tri(w_ranksvm_llm["rankSVM H9+LLM (min-max) vs rankSVM (min-max)"])],
        ["rankSVM+LLM proti MLP H9+LLM (surove značilke)", "dvostranski", pv(w_ranksvm_llm["rankSVM H9+LLM (surove) vs MLP H9+LLM"]["p"]), tri(w_ranksvm_llm["rankSVM H9+LLM (surove) vs MLP H9+LLM"])],
        ["rankSVM+LLM proti MLP H9+LLM (min-max)", "dvostranski", pv(w_ranksvm_llm["rankSVM H9+LLM (min-max) vs MLP H9+LLM"]["p"]), tri(w_ranksvm_llm["rankSVM H9+LLM (min-max) vs MLP H9+LLM"])],
        ["rankSVM+LLM: surove značilke proti min-max (isti model)", "dvostranski", dec(w_ranksvm_llm["rankSVM H9+LLM (surove) vs rankSVM H9+LLM (min-max)"]["p"], 4), tri(w_ranksvm_llm["rankSVM H9+LLM (surove) vs rankSVM H9+LLM (min-max)"])],
        [f"LLM sam proti H9 (polni pogled, n = {pogled['fullview']['n']})", "enostranski", pv(w_pogled["LLM sam vs H9"]["p"]), tri(w_pogled["LLM sam vs H9"])],
        [f"H9+LLM proti LLM samemu (polni pogled, n = {pogled['fullview']['n']})", "enostranski", pv(w_pogled["H9+LLM vs LLM sam"]["p"]) + " (test skoraj brez moči)", tri(w_pogled["H9+LLM vs LLM sam"])],
    ]
    tabela("Tabela A.2: statistični testi", ["Primerjava", "Test", "p", "Zmage / porazi / vezi"], vr, ["---", "---", "---:", "---:"])
    VRSTICE.append(f"<!-- LOPO raven napak, razpon p čez semena: {pv(min(v['p'] for v in fl.values()))} … {pv(max(v['p'] for v in fl.values()))} -->")

    # robustnost na semena, številke v besedilu
    tabela("Robustnost na semena (številke v besedilu)", ["Protokol", "Količina", "Povprečje ± st. odklon", "Izboljšanje po semenih"],
           [["desetkratna CV", "Δ Top-1 (H9+LLM − H9)", f"{znak(mlp['seed_delta_top1_mean'])}{dec(abs(mlp['seed_delta_top1_mean']), 1)} ± {dec(mlp['seed_delta_top1_sd'], 1)}", "5 od 5"],
            ["desetkratna CV", "Δ MFR", f"{znak(mlp['seed_delta_mfr_mean'])}{dec(abs(mlp['seed_delta_mfr_mean']))} ± {dec(mlp['seed_delta_mfr_sd'])}", "5 od 5"],
            ["LOPO", "Δ Top-1 (H9+LLM − H9)", f"{znak(dt['mean'])}{dec(abs(dt['mean']), 1)} ± {dec(dt['sd'], 1)}", f"{dt['improved_seeds']} od 5 (vrednosti {min(dt['values'])}–{max(dt['values'])})"],
            ["LOPO", "Δ MFR", f"{znak(dm['mean'])}{dec(abs(dm['mean']))} ± {dec(dm['sd'])}", f"{dm['improved_seeds']} od 5 (pri enem semenu se predznak obrne)"]],
           ["---", "---", "---:", "---"])

    # Top-1 po obravnavi enakih ocen, zadnji stolpec tabele 6.7
    vr = []
    for i, k in [("Ochiai", "Ochiai"), ("rankSVM, surove značilke", "rankSVM (surove, Zou)"), ("rankSVM, min-max", "rankSVM (min-max)"),
                 ("MLP H9", "MLP H9 (kontrola)"), ("**MLP H9+LLM**", "MLP H9+LLM"), ("rankSVM+LLM, surove značilke", "rankSVM H9+LLM (surove)"),
                 ("rankSVM+LLM, min-max", "rankSVM H9+LLM (min-max)")]:
        t = rez_ranksvm_llm[k]["top1_tie"]
        vr.append([i, t["min"], t["avg"], t["max"]])
    tabela("Top-1 po obravnavi enakih ocen (zadnji stolpec tabele 6.7)", ["Model", "optimistična (najmanjši rang)", "povprečna", "pesimistična (največji rang)"], vr)

    # tabeli 6.5 in 6.8 še z združeno vrstico
    zdruzeno_cv = [f"**Združeno ({mlp['n_faults']})**", f"{dec(A['MLP H9 (kontrola)']['mfr'])} → {dec(A['MLP H9+LLM']['mfr'])}",
                   f"{A['MLP H9 (kontrola)']['top']['1']} → {A['MLP H9+LLM']['top']['1']}"]
    tabela("Tabela 6.5 z združeno vrstico", ["Projekt (napak)", "MFR: H9 → H9+LLM", "Top-1: H9 → H9+LLM"],
           vr_cv + [zdruzeno_cv])
    zdruzeno_lopo = [f"**Združeno ({mlp['n_faults']})**", f"{dec(po['MLP H9 (LOPO)']['mfr'])} → {dec(po['MLP H9+LLM (LOPO)']['mfr'])}",
                     f"{po['MLP H9 (LOPO)']['top']['1']} → {po['MLP H9+LLM (LOPO)']['top']['1']}"]
    tabela("Tabela 6.8 z združeno vrstico", ["Projekt (napak)", "MFR: H9 → H9+LLM", "Top-1: H9 → H9+LLM"],
           vr_lopo + [zdruzeno_lopo])

    pot = N.pot(N.TABELE_MD)
    with open(pot, "w", encoding="utf-8") as f:
        f.write("# Tabele rezultatov, šesto poglavje in dodatek A (samodejno iz rezultatov, porocila/tabele_poglavje6.py)\n")
        f.write("\n".join(VRSTICE) + "\n")
    st_tabel = sum(1 for v in VRSTICE if v.startswith("|---"))   # ločilna vrstica glave vsake tabele
    print(f"[Shranjeno] {pot} ({st_tabel} tabel)")


if __name__ == "__main__":
    main()
