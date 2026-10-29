# -*- coding: utf-8 -*-
"""Sprotne preverbe pravilnosti z izhodno kodo.

Preverbe se nanašajo na pravilnost cevovoda (oblika podatkov, nič izgubljenih vrstic, nič uhajanja
med učno in testno množico). Izida poskusa ne preverjajo, tega poročamo v tabelah.
"""


class Preverbe:
    """Zbirka preverb ene skripte. Klic `p(ime, pogoj, podrobnost)` izpiše in shrani izid."""

    def __init__(self):
        self.izidi = []

    def __call__(self, ime, pogoj, podrobnost=""):
        pogoj = bool(pogoj)
        self.izidi.append((ime, pogoj))
        print(f"  [{'OK' if pogoj else 'NAPAKA'}] {ime}" + (f" ({podrobnost})" if podrobnost else ""))
        return pogoj

    @property
    def st_napak(self):
        return sum(1 for _, ok in self.izidi if not ok)

    def zakljuci(self):
        """Izpiše skupni izid in konča program (izhodna koda 0 ob uspehu, 2 sicer)."""
        n = self.st_napak
        print("=" * 78)
        print("PREVERBE: vse uspešne." if n == 0 else f"PREVERBE: {n} ni uspelo (glej [NAPAKA] zgoraj).")
        print("=" * 78)
        raise SystemExit(0 if n == 0 else 2)


def glava(naslov):
    """Izpiše naslov skripte."""
    print("=" * 78)
    print(naslov)
    print("=" * 78)
