#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Klici modela Claude prek paketnega vmesnika in zbiranje ocen.

Podukaz count prešteje žetone pozivov (brezplačno) in oceni strošek, send ustvari paket zahtev
(plačljivo, zahteva --potrdi), poll izpiše stanje paketa, fetch pa prenese odgovore, jih shrani
in ocene pripne k metodam. Ponovni paket zamenja le ocene napak iz tega paketa. Ključ API je v
spremenljivki ANTHROPIC_API_KEY ali v datoteki ~/.anthropic_api_key.

Zagon:  python jezikovni_model/klici.py count | poll | fetch
        python jezikovni_model/klici.py send --potrdi [--napake a,b | --seznam datoteka]
Vhod:   izhod/jezikovni_model/pozivi/, jezikovni_model/manifest_pozivov.csv
Izhod:  izhod/jezikovni_model/tokeni.csv, batch_id.txt, surovi_odgovori.jsonl, ocene.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # koren repozitorija

import argparse
import json
import os
import re
import time

import pandas as pd

import nastavitve as N

POZIVI = N.pot(N.POZIVI_MAPA)


def strosek_usd(vhodnih, izhodnih):
    """Okvirni strošek v USD za vhodne in izhodne žetone"""
    return vhodnih / 1e6 * N.CENA_VHOD_USD + izhodnih / 1e6 * N.CENA_IZHOD_USD


def kljuc_api():
    """Ključ API iz okolja ali iz datoteke v domači mapi"""
    k = os.environ.get("ANTHROPIC_API_KEY")
    if k:
        return k.strip()
    pot = os.path.expanduser("~/.anthropic_api_key")
    if os.path.isfile(pot):
        with open(pot, encoding="utf-8") as f:
            return f.read().strip()
    sys.exit("NAPAKA: ni ključa API. Nastavite ANTHROPIC_API_KEY ali datoteko ~/.anthropic_api_key.")


def odjemalec():
    import anthropic
    return anthropic.Anthropic(api_key=kljuc_api())


def napake_s_pozivom():
    """Napake, za katere je pozivi.py uspešno sestavil poziv"""
    m = pd.read_csv(N.pot(N.POZIVI_MANIFEST))
    return list(m[m["status"].astype(str).str.startswith("OK")]["fault_id"].astype(str))


def preberi_poziv(fid):
    with open(os.path.join(POZIVI, f"poziv_{fid}.txt"), encoding="utf-8") as f:
        return f.read()


def shranjeni_batch_id():
    with open(N.pot(N.LLM_BATCH_ID)) as f:
        return f.read().strip()


def parametri_zahteve(poziv, max_tokens):
    return dict(model=N.LLM_MODEL, max_tokens=max_tokens, thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": poziv}])


def izbrane_napake(args):
    """Napake iz --napake ali --seznam (ena napaka na vrstico), privzeto vse s pozivom"""
    vse = napake_s_pozivom()
    if args.napake:
        zelene = set(args.napake.split(","))
    elif args.seznam:
        with open(args.seznam, encoding="utf-8") as f:
            zelene = {v.strip() for v in f if v.strip()}
    else:
        return vse
    izbrane = [f for f in vse if f in zelene]
    manjka = zelene - set(izbrane)
    if manjka:
        print(f"  opozorilo: brez poziva (preskočeno): {sorted(manjka)[:5]}")
    return izbrane


def ukaz_count(args):
    klient = odjemalec()
    napake = napake_s_pozivom()
    print(f"[count] štejem žetone za {len(napake)} pozivov (model {N.LLM_MODEL}, brezplačno) ...")
    vrstice, zacetek = [], time.time()
    for i, fid in enumerate(napake, 1):
        r = klient.messages.count_tokens(model=N.LLM_MODEL, messages=[{"role": "user", "content": preberi_poziv(fid)}])
        vrstice.append({"fault_id": fid, "input_tokens": int(r.input_tokens)})
        if i % 25 == 0 or i == len(napake):
            print(f"  ... {i}/{len(napake)} ({time.time() - zacetek:.0f} s)")
    d = pd.DataFrame(vrstice)
    d.to_csv(N.pot(N.LLM_TOKENI), index=False)
    vhodnih = int(d["input_tokens"].sum())
    izhodnih = len(napake) * 400                  # konservativna ocena, odgovor je kratek JSON
    print(f"[count] vhodnih žetonov: {vhodnih:,}, ocena izhodnih: {izhodnih:,}")
    print(f"[count] ocena stroška (paketni vmesnik): {strosek_usd(vhodnih, izhodnih):.2f} USD")
    print(f"[count] shranjeno: {N.LLM_TOKENI}")


def ukaz_send(args):
    if not args.potrdi:
        sys.exit("Pošiljanje je plačljivo. Če se strinjate s stroškom, dodajte možnost --potrdi.")
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    klient = odjemalec()
    napake = izbrane_napake(args)
    zahteve = [Request(custom_id=fid, params=MessageCreateParamsNonStreaming(
        **parametri_zahteve(preberi_poziv(fid), args.max_tokens))) for fid in napake]
    print(f"[send] ustvarjam paket z {len(zahteve)} zahtevami (model {N.LLM_MODEL}, max_tokens {args.max_tokens}) ...")
    paket = klient.messages.batches.create(requests=zahteve)
    with open(N.pot(N.LLM_BATCH_ID), "w") as f:
        f.write(paket.id)
    print(f"[send] batch_id = {paket.id} (shranjen v {N.LLM_BATCH_ID}), stanje: {paket.processing_status}")


def ukaz_poll(args):
    klient = odjemalec()
    bid = shranjeni_batch_id()
    b = klient.messages.batches.retrieve(bid)
    print(f"[poll] {bid}: stanje {b.processing_status}, števci {b.request_counts}")


_JSON = re.compile(r"\{.*\}", re.DOTALL)


def izlusci_ocene(besedilo, n_kandidatov):
    """Ocene {indeks: ocena v [0, 100]} iz JSON v odgovoru, ob neveljavnem odgovoru prazen slovar"""
    m = _JSON.search(besedilo)
    if not m:
        return {}
    try:
        surovo = json.loads(m.group(0))
    except Exception:
        return {}
    out = {}
    for k, v in surovo.items():
        try:
            idx, ocena = int(str(k).strip()), float(v)
        except Exception:
            continue
        if 1 <= idx <= n_kandidatov:
            out[idx] = max(0.0, min(100.0, ocena))
    return out


def ukaz_fetch(args):
    klient = odjemalec()
    bid = shranjeni_batch_id()
    b = klient.messages.batches.retrieve(bid)
    if b.processing_status != "ended":
        sys.exit(f"[fetch] paket še ni končan (stanje {b.processing_status}), počakajte in poženite 'poll'.")

    pot_ocene = N.pot(N.LLM_OCENE)
    vrstice, n_ok, n_napak, vhodnih, izhodnih = [], 0, 0, 0, 0
    with open(N.pot(N.LLM_SUROVI_ODGOVORI), "a", encoding="utf-8") as surovo:
        for rezultat in klient.messages.batches.results(bid):
            fid = rezultat.custom_id
            vrsta = rezultat.result.type
            if vrsta != "succeeded":
                n_napak += 1
                surovo.write(json.dumps({"fault_id": fid, "type": vrsta}) + "\n")
                continue
            msg = rezultat.result.message
            vhodnih += int(msg.usage.input_tokens)
            izhodnih += int(msg.usage.output_tokens)
            besedilo = "".join(getattr(blok, "text", "") for blok in msg.content)
            kandidati = pd.read_csv(os.path.join(POZIVI, f"kandidati_{fid}.csv"))
            ocene = izlusci_ocene(besedilo, len(kandidati))
            surovo.write(json.dumps({"fault_id": fid, "type": vrsta, "text": besedilo}) + "\n")
            n_ok += 1
            for _, k in kandidati.iterrows():
                idx = int(k["idx"])
                vrstice.append({"fault_id": fid, "classname": k["classname"], "method": k["method"],
                                "m_start": int(k["m_start"]), "m_end": int(k["m_end"]), "idx": idx,
                                "ochiai": float(k["ochiai"]), "combined": float(k["combined"]),
                                "faulty": int(k["faulty"]), "llm_score": ocene.get(idx, 0.0),
                                "llm_missing": int(idx not in ocene)})
    d = pd.DataFrame(vrstice)
    if os.path.isfile(pot_ocene) and len(d):
        stare = pd.read_csv(pot_ocene)
        ohrani = stare[~stare["fault_id"].isin(set(d["fault_id"]))]
        print(f"[fetch] združujem: ohranjenih {ohrani['fault_id'].nunique()} napak, zamenjanih {d['fault_id'].nunique()}")
        d = pd.concat([ohrani, d], ignore_index=True)
    d.to_csv(pot_ocene, index=False)
    print(f"[fetch] uspešnih napak {n_ok}, neuspešnih {n_napak}")
    print(f"[fetch] žetoni: vhod {vhodnih:,}, izhod {izhodnih:,}, strošek paketa približno "
          f"{strosek_usd(vhodnih, izhodnih):.2f} USD")
    print(f"[fetch] metod skupaj {len(d)}, brez ocene (privzeto 0): {int(d['llm_missing'].sum()) if len(d) else 0}")
    print(f"[fetch] shranjeno: {N.LLM_OCENE}, {N.LLM_SUROVI_ODGOVORI}")


def main():
    ap = argparse.ArgumentParser(description="Klici jezikovnega modela prek paketnega vmesnika.")
    sub = ap.add_subparsers(dest="ukaz", required=True)
    sub.add_parser("count", help="preštej žetone in oceni strošek (brezplačno)")
    s = sub.add_parser("send", help="pošlji paket zahtev (plačljivo)")
    s.add_argument("--potrdi", action="store_true", help="potrditev, da se strinjate s stroškom")
    s.add_argument("--napake", default="", help="seznam napak a,b,c (privzeto vse s pozivom)")
    s.add_argument("--seznam", default="", help="datoteka z napakami, ena na vrstico")
    # pri premajhni meji gre ves proračun za razmišljanje in odgovor je prazen
    s.add_argument("--max-tokens", type=int, default=8000, help="skupna meja za razmišljanje in odgovor")
    sub.add_parser("poll", help="stanje paketa")
    sub.add_parser("fetch", help="prenesi odgovore in pripni ocene k metodam")
    args = ap.parse_args()
    {"count": ukaz_count, "send": ukaz_send, "poll": ukaz_poll, "fetch": ukaz_fetch}[args.ukaz](args)


if __name__ == "__main__":
    main()
