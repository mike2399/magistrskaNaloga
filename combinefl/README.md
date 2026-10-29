# Paket CombineFL

V to mapo razpakirajte paket CombineFL (Zou in sod., *An Empirical Study of Fault Localization
Families and Their Combinations*, IEEE TSE 2021), ki je na voljo na
https://damingz.github.io/combinefl/index.html (datoteka `CombineFL-v0.1.tar.gz`).

Cevovod potrebuje iz paketa:

- `data/release.json` (približno 211 MB), ocene devetih tehnik za 357 napak zbirke Defects4J.
  To je vhod skripte `priprava_podatkov/matrika_stavkov.py`.
- `SVMRank/svm_rank_learn` in `SVMRank/svm_rank_classify`, programa rankSVM (T. Joachims), ki ju
  uporabljata skripti `modeli/ranksvm_h9.py` in `modeli/ranksvm_llm.py`. Programa sta v paketu CombineFL
  v obliki za Linux, licenca SVM-rank dovoljuje uporabo v znanstvene namene.

Mapa je izvzeta iz gita (`.gitignore`), ker so datoteke velike in niso naše delo. Drugo pot do
paketa lahko nastavite z okoljsko spremenljivko `FL_COMBINEFL`.
