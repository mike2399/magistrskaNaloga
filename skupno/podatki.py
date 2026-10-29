# -*- coding: utf-8 -*-
"""Nalaganje in priprava podatkov: matrika metod, normalizacija, delitev v gube, ocene LLM, predpomnilnik."""
import hashlib

import numpy as np
import pandas as pd

from nastavitve import KLJUC, SEME_DELITVE, K_GUB


def razcleni_fault_id(fid):
    """'closure127' -> ('closure', '127'). Števke na koncu so številka napake."""
    i = len(fid)
    while i > 0 and fid[i - 1].isdigit():
        i -= 1
    return fid[:i], fid[i:]


def projekt_napake(fid):
    """Ime projekta iz oznake napake: 'closure127' -> 'closure'."""
    return razcleni_fault_id(fid)[0]


def ime_d4j(fid):
    """Oznaka napake v obliki, ki jo pričakuje Defects4J: 'lang1' -> ('Lang', '1')."""
    projekt, bug = razcleni_fault_id(fid)
    return projekt.capitalize(), bug


def nalozi_matriko_metod(pot):
    """Prebere matriko metod in poskrbi za cele tipe stolpcev."""
    df = pd.read_parquet(pot).reset_index(drop=True)
    df["faulty"] = df["faulty"].astype(int)
    for c in ("m_start", "m_end"):
        df[c] = df[c].astype(int)
    return df


def seznam_napak(df):
    """Napake v vrstnem redu prve pojavitve v tabeli. Ta red določa delitev v gube."""
    return list(dict.fromkeys(df["fault_id"].tolist()))


def normiraj_po_napaki(df, znacilke):
    """Min-max normalizacija vsake značilke znotraj posamezne napake v [0, 1], konstantna značilka dobi 0."""
    out = df.copy()
    g = df.groupby("fault_id")
    for t in znacilke:
        mn, mx = g[t].transform("min"), g[t].transform("max")
        razpon = (mx - mn)
        out[t] = ((df[t] - mn) / razpon.where(razpon > 0, 1.0)).where(razpon > 0, 0.0)
    return out


def gube_po_napaki(napake, seme=SEME_DELITVE, k=K_GUB):
    """Razdeli napake v k gub (seznam množic fault_id). Vse metode iste napake so v isti gubi."""
    napake = np.array(napake)
    perm = np.random.default_rng(seme).permutation(len(napake))
    return [set(napake[perm[i::k]]) for i in range(k)]


def zdruzi_llm_ocene(df, pot_ocene):
    """Pripne ocene jezikovnega modela k matriki metod, metode brez ocene dobijo 0.

    Vrne (df, n_ocenjenih). Stolpec `llm` je ocena od 0 do 100, `llm_scored` pove, ali je bila metoda
    med kandidati, torej jo je model sploh videl.
    """
    ocene = pd.read_csv(pot_ocene)[KLJUC + ["llm_score"]]
    for c in ("m_start", "m_end"):
        ocene[c] = ocene[c].astype(int)
    pred = len(df)
    df = df.merge(ocene, on=KLJUC, how="left")
    if len(df) != pred:
        raise RuntimeError(f"združevanje z ocenami LLM je spremenilo število vrstic: {pred} -> {len(df)}")
    n_ocenjenih = int(df["llm_score"].notna().sum())
    df["llm_scored"] = df["llm_score"].notna()
    df["llm"] = df["llm_score"].fillna(0.0)
    df.drop(columns=["llm_score"], inplace=True)
    return df, n_ocenjenih


def v_intervalu_0_1(df, znacilke):
    """True, če so vse navedene značilke v [0, 1]."""
    return float(df[znacilke].min().min()) >= 0.0 and float(df[znacilke].max().max()) <= 1.0


def kljuc_predpomnilnika(df, znacilke, parametri, predpona=b""):
    """Zgoščena vrednost nastavitev in vhodnih podatkov za ime datoteke v predpomnilniku napovedi.

    Sprememba značilk, parametrov, vrednosti ali vrstnega reda napak razveljavi shranjene napovedi.
    """
    h = hashlib.sha1()
    h.update(predpona)
    h.update("|".join(znacilke).encode())
    h.update(np.asarray(parametri, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(df[znacilke].to_numpy(dtype=np.float64)).tobytes())
    h.update(np.ascontiguousarray(df["faulty"].to_numpy(dtype=np.int64)).tobytes())
    h.update(",".join(map(str, df["fault_id"].tolist())).encode())
    return h.hexdigest()[:12]
