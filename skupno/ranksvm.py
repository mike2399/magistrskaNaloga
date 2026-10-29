# -*- coding: utf-8 -*-
"""Linearni model rankSVM (program svm_rank), kot ga uporablja CombineFL.

Podatke zapišemo v obliki LETOR, model naučimo s programom svm_rank_learn in napovemo s svm_rank_classify.
Napovedi shranimo v predpomnilnik, katerega ključ je izračunan iz nastavitev in vhodnih podatkov.
"""
import json
import os
import subprocess

import numpy as np

from nastavitve import SVM_RANK_LEARN, SVM_RANK_CLASSIFY, SEME_DELITVE, K_GUB
from skupno.podatki import kljuc_predpomnilnika

C_REGULARIZACIJA = 1          # privzeta vrednost iz skripte 3-crossvalidation.sh paketa CombineFL


def preveri_programa():
    """Ustavi se, če programa svm_rank nista na voljo ali nista izvršljiva."""
    for p in (SVM_RANK_LEARN, SVM_RANK_CLASSIFY):
        if not os.access(p, os.X_OK):
            raise RuntimeError(f"program {p} ne obstaja ali ni izvršljiv (glej README, razdelek o CombineFL)")


def zapisi_letor(sub, znacilke, qid_napake, pot):
    """Zapiše tabelo v datoteko LETOR, urejeno po qid. Vrne oznake vrstic v zapisanem vrstnem redu."""
    q = sub["fault_id"].map(qid_napake).to_numpy()
    vrstni_red = np.argsort(q, kind="mergesort")        # stabilno urejanje ohrani vrstni red znotraj qid
    q2 = q[vrstni_red]
    cilj = sub["faulty"].to_numpy(dtype=int)[vrstni_red]
    F = sub[znacilke].to_numpy(dtype=float)[vrstni_red]
    with open(pot, "w") as f:
        for t, qq, vrstica in zip(cilj, q2, F):
            feat = " ".join(f"{j + 1}:{v:.9g}" for j, v in enumerate(vrstica))
            f.write(f"{t} qid:{qq} {feat}\n")
    return sub.index.to_numpy()[vrstni_red]


def preberi_napovedi(pot):
    with open(pot) as f:
        return np.array([float(x) for x in f if x.strip() != ""], dtype=np.float64)


def pozeni_svm(args, kaj):
    r = subprocess.run([str(a) for a in args], capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(f"{kaj}: izhodna koda {r.returncode}\nSTDERR:\n{r.stderr}\nSTDOUT:\n{r.stdout}")
    return r.stdout


def preveri_letor(pot, n_znacilk, vrni_podatke=False):
    """Preveri obliko datoteke LETOR (cilj 0 ali 1, strnjeni qid, značilke 1..n po vrsti). Vrne množico qid."""
    videni, trenutni = set(), None
    cilji, vrstice = [], []
    with open(pot) as f:
        for st, line in enumerate(f, 1):
            deli = line.split()
            if not deli:
                continue
            t = deli[0]
            if t not in ("0", "1"):
                raise RuntimeError(f"{pot}:{st}: cilj '{t}' ni 0/1")
            if not deli[1].startswith("qid:"):
                raise RuntimeError(f"{pot}:{st}: manjka 'qid:'")
            q = int(deli[1][4:])
            if q != trenutni:
                if q in videni:
                    raise RuntimeError(f"{pot}:{st}: qid {q} ni strnjen")
                videni.add(q)
                trenutni = q
            zetoni = deli[2:]
            if len(zetoni) != n_znacilk:
                raise RuntimeError(f"{pot}:{st}: {len(zetoni)} značilk namesto {n_znacilk}")
            vrednosti = []
            for j, zeton in enumerate(zetoni, 1):
                fid, _, sval = zeton.partition(":")
                if int(fid) != j:
                    raise RuntimeError(f"{pot}:{st}: oznaka značilke '{fid}' ni {j}")
                vrednosti.append(float(sval))
            if vrni_podatke:
                cilji.append(int(t))
                vrstice.append(vrednosti)
    if vrni_podatke:
        return videni, np.array(cilji, dtype=int), np.array(vrstice, dtype=np.float64)
    return videni


def preberi_model(pot, n_znacilk):
    """Iz datoteke modela svm_rank (oblika SVM-light) izlušči prag b in vektor uteži w (vsota alpha krat podporni vektor)."""
    with open(pot) as f:
        vrstice = f.read().splitlines()
    if "SVM-light" not in vrstice[0]:
        raise RuntimeError(f"{pot}: ni model SVM-light (prva vrstica: {vrstice[0]!r})")
    najvisji = int(vrstice[7].split()[0])
    if najvisji > n_znacilk + 1:
        raise RuntimeError(f"{pot}: najvišji indeks značilke {najvisji} > {n_znacilk}+1")
    n_sv1 = int(vrstice[9].split()[0])
    b = float(vrstice[10].split()[0])
    sv = [v for v in vrstice[11:] if v.strip()]
    if len(sv) != n_sv1 - 1:
        raise RuntimeError(f"{pot}: {len(sv)} podpornih vektorjev, pričakovano {n_sv1 - 1}")
    w = np.zeros(n_znacilk, dtype=np.float64)
    for line in sv:
        deli = line.split()
        alpha = float(deli[0])
        for zeton in deli[1:]:
            if zeton == "#":
                break
            fid, _, sval = zeton.partition(":")
            i = int(fid)
            if not (1 <= i <= n_znacilk):
                raise RuntimeError(f"{pot}: podporni vektor ima indeks {i} izven 1..{n_znacilk}")
            w[i - 1] += alpha * float(sval)
    return b, w


def ranksvm_cv(df, znacilke, gube, qid_napake, oznaka, mapa_tmp, mapa_cache):
    """Prečno preverjanje po napakah z modelom rankSVM.

    Vrne (napovedi, uteži po gubah). Uteži so None, če so napovedi iz predpomnilnika brez shranjenih uteži.
    Pri svežem izračunu preveri zapisane datoteke LETOR in da uteži reproducirajo napovedi programa.
    """
    os.makedirs(mapa_cache, exist_ok=True)
    kljuc = kljuc_predpomnilnika(df, znacilke, [C_REGULARIZACIJA, SEME_DELITVE, K_GUB, len(df)])
    pot_npy = os.path.join(mapa_cache, f"ranksvm_{oznaka}_{kljuc}.npy")
    pot_utezi = os.path.join(mapa_cache, f"utezi_{oznaka}_{kljuc}.json")
    if os.path.isfile(pot_npy):
        print(f"  [{oznaka}] napovedi iz predpomnilnika: {os.path.basename(pot_npy)}")
        utezi = None
        if os.path.isfile(pot_utezi):
            with open(pot_utezi, encoding="utf-8") as f:
                utezi = [np.array(w, dtype=float) for w in json.load(f)]
        return np.load(pot_npy), utezi

    os.makedirs(mapa_tmp, exist_ok=True)
    napoved = np.full(len(df), np.nan, dtype=np.float64)
    napaka_vrstice = df["fault_id"].to_numpy()
    vsi_qid = set(qid_napake.values())
    utezi = []
    for k, testne in enumerate(gube):
        ucna = ~np.isin(napaka_vrstice, list(testne))
        testna = ~ucna
        mapa = os.path.join(mapa_tmp, f"{oznaka}_fold{k}")
        os.makedirs(mapa, exist_ok=True)
        p_ucna, p_testna = os.path.join(mapa, "train.dat"), os.path.join(mapa, "test.dat")
        p_model, p_napoved = os.path.join(mapa, "model.dat"), os.path.join(mapa, "pred.dat")
        zapisi_letor(df.loc[ucna], znacilke, qid_napake, p_ucna)
        indeks_testne = zapisi_letor(df.loc[testna], znacilke, qid_napake, p_testna)

        # preverbe zapisanih datotek
        qid_ucna = preveri_letor(p_ucna, len(znacilke))
        qid_testna, cilji_testna, F_testna = preveri_letor(p_testna, len(znacilke), vrni_podatke=True)
        if qid_ucna & qid_testna:
            raise RuntimeError(f"{oznaka} guba {k}: isti qid v učni in testni množici")
        if qid_testna != {qid_napake[f] for f in testne}:
            raise RuntimeError(f"{oznaka} guba {k}: testni qid na disku niso qid testnih napak")
        if (qid_ucna | qid_testna) != vsi_qid:
            raise RuntimeError(f"{oznaka} guba {k}: učna in testna množica ne pokrijeta vseh qid")
        F_pric = df.loc[indeks_testne, znacilke].to_numpy(dtype=float)
        if F_testna.shape != F_pric.shape or not np.allclose(F_testna, F_pric, rtol=1e-6, atol=0.0):
            raise RuntimeError(f"{oznaka} guba {k}: zapisane testne značilke se ne ujemajo s tabelo")
        if not np.array_equal(cilji_testna, df.loc[indeks_testne, "faulty"].to_numpy(dtype=int)):
            raise RuntimeError(f"{oznaka} guba {k}: zapisani cilji se ne ujemajo s tabelo")

        pozeni_svm([SVM_RANK_LEARN, "-c", str(C_REGULARIZACIJA), p_ucna, p_model], f"svm_rank_learn [{oznaka} guba {k}]")
        pozeni_svm([SVM_RANK_CLASSIFY, p_testna, p_model, p_napoved], f"svm_rank_classify [{oznaka} guba {k}]")
        ocene = preberi_napovedi(p_napoved)
        if len(ocene) != len(indeks_testne):
            raise RuntimeError(f"{oznaka} guba {k}: {len(ocene)} napovedi za {len(indeks_testne)} testnih vrstic")

        # uteži in njihova preverba, w . x - b mora dati isto kot program
        b, w = preberi_model(p_model, len(znacilke))
        rekonstrukcija = F_testna @ w - b
        toleranca = 1e-3 * max(1.0, float(np.max(np.abs(ocene))))
        odklon = float(np.max(np.abs(rekonstrukcija - ocene)))
        if odklon > toleranca:
            raise RuntimeError(f"{oznaka} guba {k}: uteži ne reproducirajo napovedi (odklon {odklon:.3g})")
        utezi.append(w)

        napoved[indeks_testne] = ocene
        print(f"  [{oznaka}] guba {k + 1:2d}/{len(gube)}: učnih {int(ucna.sum()):6d}, testnih {int(testna.sum()):5d} "
              f"metod, {len(testne):3d} napak", flush=True)
    if np.isnan(napoved).any():
        raise RuntimeError(f"{oznaka}: {int(np.isnan(napoved).sum())} metod brez napovedi")
    np.save(pot_npy, napoved)
    with open(pot_utezi, "w", encoding="utf-8") as f:
        json.dump([w.tolist() for w in utezi], f)
    return napoved, utezi


def povzetek_utezi(utezi, znacilke):
    """Povprečje in standardni odklon uteži čez gube za vsako značilko."""
    W = np.vstack(utezi)
    return {ime: {"mean": float(W[:, j].mean()), "sd": float(W[:, j].std(ddof=1))}
            for j, ime in enumerate(znacilke)}
