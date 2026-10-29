#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Klici jezikovnega modela (Claude) prek paketnega vmesnika in zbiranje ocen.

Podukazi:
  count   prešteje žetone vseh pozivov (brezplačno) in oceni strošek
  send    ustvari en paket zahtev (plačljivo, zahteva možnost --potrdi)
  poll    izpiše stanje paketa
  fetch   prenese odgovore, iz njih izlušči ocene in jih pripne k metodam, surove odgovore shrani.
          Ob ponovnem paketu zamenja samo ocene napak iz tega paketa, ostale ohrani.

Ključ API je v okoljski spremenljivki ANTHROPIC_API_KEY ali v datoteki ~/.anthropic_api_key.
Parameter max_tokens je skupna meja za razmišljanje in odgovor. Pri premajhni meji model ves proračun
porabi za razmišljanje in vrne prazen odgovor, zato je privzeta meja 8000.

Zagon:  python jezikovni_model/klici.py count
        python jezikovni_model/klici.py send --potrdi [--napake a,b | --seznam datoteka] [--max-tokens 8000]
        python jezikovni_model/klici.py poll
        python jezikovni_model/klici.py fetch
        (7. skripta v cevovodu, vrstni red je v README)
Vhod:   izhod/jezikovni_model/pozivi/, jezikovni_model/manifest_pozivov.csv
Izhod:  izhod/jezikovni_model/tokeni.csv, jezikovni_model/batch_id.txt, jezikovni_model/surovi_odgovori.jsonl,
        jezikovni_model/ocene.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # koren repozitorija (nastavitve.py, skupno/)

import argparse
import json
import os
import re
import time

import pandas as pd

import nastavitve as N

MODEL = N.LLM_MODEL
CENA_VHOD, CENA_IZHOD = N.CENA_VHOD_USD, N.CENA_IZHOD_USD
POZIVI = N.pot(N.POZIVI_MAPA)


def kljuc_api():
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
    """Napake, za katere je pozivi.py uspešno sestavil poziv."""
    m = pd.read_csv(N.pot(N.POZIVI_MANIFEST))
    return list(m[m["status"].astype(str).str.startswith("OK")]["fault_id"].astype(str))


def preberi_poziv(fid):
    with open(os.path.join(POZIVI, f"poziv_{fid}.txt"), encoding="utf-8") as f:
        return f.read()


def parametri_zahteve(poziv, max_tokens):
    return dict(model=MODEL, max_tokens=max_tokens, thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": poziv}])


def izbrane_napake(args):
    """Napake iz --napake ali --seznam (ena napaka na vrstico), privzeto vse s pozivom."""
    vse = napake_s_pozivom()
    zelene = None
    if args.napake:
        zelene = set(args.napake.split(","))
    elif args.seznam:
        with open(args.seznam, encoding="utf-8") as f:
            zelene = {v.strip() for v in f if v.strip()}
    if zelene is None:
        return vse
    izbrane = [f for f in vse if f in zelene]
    manjka = zelene - set(izbrane)
    if manjka:
        print(f"  opozorilo: brez poziva (preskočeno): {sorted(manjka)[:5]}")
    return izbrane


def ukaz_count(args):
    klient = odjemalec()
    napake = napake_s_pozivom()
    print(f"[count] štejem žetone za {len(napake)} pozivov (model {MODEL}, brezplačno) ...")
    vrstice, zacetek = [], time.time()
    for i, fid in enumerate(napake, 1):
        r = klient.messages.count_tokens(model=MODEL, messages=[{"role": "user", "content": preberi_poziv(fid)}])
        vrstice.append({"fault_id": fid, "input_tokens": int(r.input_tokens)})
        if i % 25 == 0 or i == len(napake):
            print(f"  ... {i}/{len(napake)} ({time.time() - zacetek:.0f} s)")
    d = pd.DataFrame(vrstice)
    d.to_csv(N.pot(N.LLM_TOKENI), index=False)
    vhodnih = int(d["input_tokens"].sum())
    izhodnih = len(napake) * 400                  # konservativna ocena, odgovor je kratek JSON
    print(f"[count] vhodnih žetonov: {vhodnih:,}, ocena izhodnih: {izhodnih:,}")
    print(f"[count] ocena stroška (paketni vmesnik): {vhodnih / 1e6 * CENA_VHOD + izhodnih / 1e6 * CENA_IZHOD:.2f} USD")
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
    print(f"[send] ustvarjam paket z {len(zahteve)} zahtevami (model {MODEL}, max_tokens {args.max_tokens}) ...")
    paket = klient.messages.batches.create(requests=zahteve)
    with open(N.pot(N.LLM_BATCH_ID), "w") as f:
        f.write(paket.id)
    print(f"[send] batch_id = {paket.id} (shranjen v {N.LLM_BATCH_ID}), stanje: {paket.processing_status}")


def ukaz_poll(args):
    klient = odjemalec()
    with open(N.pot(N.LLM_BATCH_ID)) as f:
        bid = f.read().strip()
    b = klient.messages.batches.retrieve(bid)
    print(f"[poll] {bid}: stanje {b.processing_status}, števci {b.request_counts}")


_JSON = re.compile(r"\{.*\}", re.DOTALL)


def izlusci_ocene(besedilo, n_kandidatov):
    """Iz odgovora izlušči JSON {indeks: ocena} in vrne {int: float v [0, 100]}, ob neveljavnem odgovoru {}."""
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
    with open(N.pot(N.LLM_BATCH_ID)) as f:
        bid = f.read().strip()
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
                vrstice.append({"fault_id": fid, "classname": k["classname"], "method": k["method"],
                                "m_start": int(k["m_start"]), "m_end": int(k["m_end"]), "idx": int(k["idx"]),
                                "ochiai": float(k["ochiai"]), "combined": float(k["combined"]),
                                "faulty": int(k["faulty"]), "llm_score": ocene.get(int(k["idx"]), 0.0),
                                "llm_missing": int(int(k["idx"]) not in ocene)})
    d = pd.DataFrame(vrstice)
    if os.path.isfile(pot_ocene) and len(d):
        stare = pd.read_csv(pot_ocene)
        ohrani = stare[~stare["fault_id"].isin(set(d["fault_id"]))]
        print(f"[fetch] združujem: ohranjenih {ohrani['fault_id'].nunique()} napak, zamenjanih {d['fault_id'].nunique()}")
        d = pd.concat([ohrani, d], ignore_index=True)
    d.to_csv(pot_ocene, index=False)
    print(f"[fetch] uspešnih napak {n_ok}, neuspešnih {n_napak}")
    print(f"[fetch] žetoni: vhod {vhodnih:,}, izhod {izhodnih:,}, strošek paketa približno "
          f"{vhodnih / 1e6 * CENA_VHOD + izhodnih / 1e6 * CENA_IZHOD:.2f} USD")
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
    s.add_argument("--max-tokens", type=int, default=8000, help="skupna meja za razmišljanje in odgovor")
    sub.add_parser("poll", help="stanje paketa")
    sub.add_parser("fetch", help="prenesi odgovore in pripni ocene k metodam")
    args = ap.parse_args()
    {"count": ukaz_count, "send": ukaz_send, "poll": ukaz_poll, "fetch": ukaz_fetch}[args.ukaz](args)


if __name__ == "__main__":
    main()
