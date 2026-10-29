# -*- coding: utf-8 -*-
"""Skupne nastavitve cevovoda: poti do podatkov in orodij, konstante in imena izhodnih datotek.

Vse skripte berejo poti od tod, zato je dovolj, da jih prilagodite na enem mestu. Poti je mogoče
nastaviti tudi z okoljskimi spremenljivkami, kar pride prav, kadar podatki niso v mapi repozitorija:

  FL_IZHOD        mapa z rezultati                     (privzeto <repozitorij>/izhod)
  FL_COMBINEFL    mapa s paketom CombineFL             (privzeto <repozitorij>/combinefl)
  DEFECTS4J_HOME  koren namestitve Defects4J           (privzeto ~/defects4j)
  D4J_CHECKOUTS   mapa za delovne kopije programov     (privzeto ~/d4j_checkouts)
"""
import os
from pathlib import Path

# ----------------------------------------------------------------------------- poti
KOREN = Path(__file__).resolve().parent
IZHOD = Path(os.environ.get("FL_IZHOD", KOREN / "izhod"))
COMBINEFL = Path(os.environ.get("FL_COMBINEFL", KOREN / "combinefl"))

RELEASE_JSON = COMBINEFL / "data" / "release.json"            # ocene devetih tehnik za 357 napak
SVM_RANK_LEARN = COMBINEFL / "SVMRank" / "svm_rank_learn"       # program rankSVM (učenje)
SVM_RANK_CLASSIFY = COMBINEFL / "SVMRank" / "svm_rank_classify" # program rankSVM (napovedovanje)

DEFECTS4J_HOME = Path(os.environ.get("DEFECTS4J_HOME", Path.home() / "defects4j"))
DEFECTS4J = DEFECTS4J_HOME / "framework" / "bin" / "defects4j"
D4J_CHECKOUTS = Path(os.environ.get("D4J_CHECKOUTS", Path.home() / "d4j_checkouts"))


def pot(ime):
    """Polna pot do datoteke ali mape z relativnim imenom `ime` v mapi IZHOD. Manjkajoče nadmape ustvari."""
    p = IZHOD / ime
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


# ----------------------------------------------------------------------------- konstante
# Devet klasičnih tehnik iz zbirke CombineFL. Vrstni red je pomemben, ker so v tem redu
# zložene značilke v vseh modelih (in v datotekah LETOR za rankSVM)
TEHNIKE = ["ochiai", "dstar", "metallaxis", "muse", "slicing",
           "slicing_count", "slicing_intersection", "stacktrace", "predicateswitching"]

# Družine tehnik (za analize/primerjava_druzin.py)
DRUZINE = {
    "SBFL": ["ochiai", "dstar"],
    "MBFL": ["metallaxis", "muse"],
    "PS": ["predicateswitching"],
    "Rezanje": ["slicing", "slicing_count", "slicing_intersection"],
    "SledSklada": ["stacktrace"],
}

# Enolična identiteta metode v vseh tabelah (razred, ime, prva in zadnja vrstica)
KLJUC = ["fault_id", "classname", "method", "m_start", "m_end"]

NS = [1, 3, 5, 10]                      # Top-N, ki jih poročamo
SEME_DELITVE = 2018                     # seme za delitev napak v gube (kot v CombineFL)
K_GUB = 10                              # število gub pri prečnem preverjanju po napakah
SEMENA = [2018, 7, 101, 2024, 31337]    # semena inicializacije uteži MLP (prvo je primarno)
K_KANDIDATOV = 50                       # koliko metod na napako dobi jezikovni model

# Jezikovni model (jezikovni_model/klici.py) in okvirna cena v USD na milijon žetonov (paketni vmesnik, julij 2026)
LLM_MODEL = "claude-sonnet-5"
CENA_VHOD_USD = 1.0
CENA_IZHOD_USD = 5.0

# Pričakovana sestava zbirke CombineFL (357 napak v petih projektih Defects4J)
PROJEKTI_COMBINEFL = {"math": 106, "closure": 133, "time": 27, "chart": 26, "lang": 65}

# ----------------------------------------------------------------------------- imena izhodnih datotek
# Vse so relativne na mapo IZHOD in razvrščene v podmape po vsebini (enako kot skripte):
#   podatki/          matriki ocen in razvrstitev napak
#   izhodisca/        Ochiai in DStar na stavkih in metodah
#   jezikovni_model/  pozivi, odgovori in ocene jezikovnega modela
#   modeli/           naučeni modeli (MLP, rankSVM, LOPO) z napovedmi in predpomnilniki
#   analize/          polni pogled, primerjava družin, kvalitativni primeri
#   porocila/         tabele in slike za besedilo naloge

MATRIKA_STAVKOV = "podatki/matrika_stavkov.parquet"
MATRIKA_STAVKOV_VZOREC = "podatki/matrika_stavkov_vzorec.csv"
NAPAKE_BREZ_OZNAK = "podatki/napake_brez_oznak_stavkov.txt"
MATRIKA_METOD = "podatki/matrika_metod.parquet"
MATRIKA_METOD_CSV = "podatki/matrika_metod_sprotno.csv"    # sprotni zapis (omogoča nadaljevanje prekinjenega zagona)
NAPAKE_INFO = "podatki/napake_razvrstitev.csv"             # razvrstitev vsake napake (lokalizabilna ali razlog izločitve)

OCHIAI_STAVKI_RANGI = "izhodisca/ochiai_stavki_rangi.csv"
OCHIAI_STAVKI_TXT = "izhodisca/ochiai_stavki.txt"
OCHIAI_METODE_RANGI = "izhodisca/ochiai_metode_rangi.csv"
OCHIAI_METODE_TXT = "izhodisca/ochiai_metode.txt"

POZIVI_MAPA = "jezikovni_model/pozivi"
POZIVI_MANIFEST = "jezikovni_model/manifest_pozivov.csv"
LLM_TOKENI = "jezikovni_model/tokeni.csv"
LLM_BATCH_ID = "jezikovni_model/batch_id.txt"
LLM_SUROVI_ODGOVORI = "jezikovni_model/surovi_odgovori.jsonl"
LLM_OCENE = "jezikovni_model/ocene.csv"

MLP_H9_NAPOVEDI = "modeli/mlp_h9_napovedi.csv"
MLP_H9_TXT = "modeli/mlp_h9.txt"
MLP_LLM_EPOHE = "modeli/mlp_llm_izbrane_epohe.json"
MLP_LLM_CACHE = "modeli/mlp_llm_cache"
MLP_LLM_NAPOVEDI = "modeli/mlp_llm_napovedi.csv"
MLP_LLM_REZULTATI = "modeli/mlp_llm_rezultati.json"
RANKSVM_TMP = "modeli/ranksvm_h9_tmp"
RANKSVM_CACHE = "modeli/ranksvm_h9_cache"
RANKSVM_NAPOVEDI = "modeli/ranksvm_h9_napovedi.csv"
RANKSVM_REZULTATI = "modeli/ranksvm_h9_rezultati.json"
RANKSVM_TXT = "modeli/ranksvm_h9_rezultati.txt"
LOPO_CACHE = "modeli/mlp_lopo_cache"
LOPO_NAPOVEDI = "modeli/mlp_lopo_napovedi.csv"
LOPO_REZULTATI = "modeli/mlp_lopo_rezultati.json"
RANKSVM_LLM_TMP = "modeli/ranksvm_llm_tmp"
RANKSVM_LLM_CACHE = "modeli/ranksvm_llm_cache"
RANKSVM_LLM_NAPOVEDI = "modeli/ranksvm_llm_napovedi.csv"
RANKSVM_LLM_REZULTATI = "modeli/ranksvm_llm_rezultati.json"

POLNI_POGLED_REZULTATI = "analize/polni_pogled.json"
PRIMERI_MAPA = "analize/kvalitativni_primeri"
PRIMERI_PREGLED = "analize/kvalitativni_primeri_pregled.csv"
DRUZINE_REZULTATI = "analize/primerjava_druzin.json"
DRUZINE_TXT = "analize/primerjava_druzin.txt"

TABELE_MD = "porocila/tabele_poglavje6.md"
SLIKE_MAPA = "porocila/slike"
PODATKI_SLIK = "porocila/podatki_slik.json"
