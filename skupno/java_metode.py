# -*- coding: utf-8 -*-
"""Meje metod v izvorni kodi Java (tree-sitter) in krive metode iz razvijalčevega popravka"""
import difflib
import os

from skupno.defects4j import pot_razreda


def nalozi_parser():
    """Razčlenjevalnik tree-sitter za Javo"""
    import tree_sitter_java as tsjava
    from tree_sitter import Language, Parser
    return Parser(Language(tsjava.language()))


def _zberi_metode(vozlisce, bajti, out):
    """Rekurzivno zbere (ime, prva vrstica, zadnja vrstica) metod in konstruktorjev, vrstice od 1"""
    if vozlisce.type in ("method_declaration", "constructor_declaration"):
        ime_v = vozlisce.child_by_field_name("name")
        ime = bajti[ime_v.start_byte:ime_v.end_byte].decode("utf-8", "replace") if ime_v else "?"
        out.append((ime, vozlisce.start_point[0] + 1, vozlisce.end_point[0] + 1))
    for otrok in vozlisce.children:
        _zberi_metode(otrok, bajti, out)


def metode_v_datoteki(pot, parser):
    """Seznam (ime, prva, zadnja vrstica) metod in konstruktorjev, None, če datoteke ni"""
    if not os.path.isfile(pot):
        return None
    with open(pot, "rb") as f:
        bajti = f.read()
    out = []
    _zberi_metode(parser.parse(bajti).root_node, bajti, out)
    return out


def objemajoca_metoda(metode, vrstica):
    """Najbolj notranja metoda, ki vsebuje vrstico (ta ima največji začetek), ali None"""
    kandidati = [m for m in metode if m[1] <= vrstica <= m[2]]
    return max(kandidati, key=lambda m: m[1]) if kandidati else None


def _vrstice(pot):
    """Vrstice datoteke, deljene le na znak za novo vrstico, enako kot šteje tree-sitter"""
    with open(pot, "rb") as f:
        return f.read().decode("utf-8", "replace").split("\n")


def krive_metode_iz_popravka(mapa_b, mapa_f, srcdir_b, srcdir_f, spremenjeni_razredi, parser):
    """Krive metode iz razvijalčevega popravka, iz razlike med napačno in popravljeno različico.

    Vrne krive in nove metode, vrsto napake in ali se spremenjeni razredi ujemajo s classes.modified
    """
    krive, nove, razredni_nivo, z_razliko = set(), set(), False, set()
    for razred in spremenjeni_razredi:
        zunanji = razred.split("$")[0]
        fb = pot_razreda(mapa_b, srcdir_b, zunanji)
        ff = pot_razreda(mapa_f, srcdir_f, zunanji)
        if not (os.path.isfile(fb) and os.path.isfile(ff)):
            continue
        bl, fl = _vrstice(fb), _vrstice(ff)
        mb = metode_v_datoteki(fb, parser)
        mf = metode_v_datoteki(ff, parser)
        imena_b = {m[0] for m in mb}
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, bl, fl, autojunk=False).get_opcodes():
            if tag == "equal":
                continue
            z_razliko.add(zunanji)
            if tag in ("replace", "delete"):
                # zamenjane in izbrisane vrstice pripišemo objemajoči metodi napačne različice
                for ln in range(i1 + 1, i2 + 1):
                    m = objemajoca_metoda(mb, ln)
                    if m:
                        krive.add((zunanji, m[0], m[1], m[2]))
                    else:
                        razredni_nivo = True
            elif tag == "insert":
                # vstavljene vrstice pripišemo metodi okoli mesta vstavljanja,
                # to je vrstica i1 ali i1+1 napačne različice (štetje od 1)
                objema = objemajoca_metoda(mb, i1) or objemajoca_metoda(mb, i1 + 1)
                if objema:
                    krive.add((zunanji, objema[0], objema[1], objema[2]))
                else:
                    # zunaj metod napačne različice: nova metoda ali sprememba na nivoju razreda
                    for ln in range(j1 + 1, j2 + 1):
                        mfm = objemajoca_metoda(mf, ln)
                        if mfm is None:
                            razredni_nivo = True
                        elif mfm[0] not in imena_b:
                            nove.add((zunanji, mfm[0]))
                        else:
                            razredni_nivo = True        # istoimenska preobremenitev, ne ugibamo
    vrsta = ("LOKALIZABILNO" if krive else
             "NEW-METHOD-OMISSION" if nove else
             "RAZREDNI-NIVO" if razredni_nivo else "NEDOLOCENO")
    ujemanje = (z_razliko == {r.split("$")[0] for r in spremenjeni_razredi})
    return krive, nove, vrsta, ujemanje
