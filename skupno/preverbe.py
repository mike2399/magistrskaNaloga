# -*- coding: utf-8 -*-
"""Sprotne preverbe pravilnosti cevovoda, ki določijo izhodno kodo skripte.

Preverjajo obliko podatkov, število vrstic in uhajanje med učno in testno množico, ne izida poskusa
"""


class Preverbe:
    """Preverbe ene skripte, klic p(ime, pogoj, podrobnost) izpiše izid in šteje neuspehe"""

    def __init__(self):
        self.n_napak = 0

    def __call__(self, ime, pogoj, podrobnost=""):
        pogoj = bool(pogoj)
        if not pogoj:
            self.n_napak += 1
        print(f"  [{'OK' if pogoj else 'NAPAKA'}] {ime}" + (f" ({podrobnost})" if podrobnost else ""))
        return pogoj

    def zakljuci(self):
        """Izpiše skupni izid in konča program, izhodna koda 0 ob uspehu in 2 sicer"""
        n = self.n_napak
        print("=" * 78)
        print("PREVERBE: vse uspešne." if n == 0 else f"PREVERBE: {n} ni uspelo (glej [NAPAKA] zgoraj).")
        print("=" * 78)
        raise SystemExit(0 if n == 0 else 2)


def glava(naslov):
    """Izpiše naslov skripte"""
    print("=" * 78)
    print(naslov)
    print("=" * 78)
