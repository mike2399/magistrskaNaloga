# Analiza tehnik lokalizacije napak v programih

Naloga na 343 resničnih napakah iz zbirke Defects4J primerja devet klasičnih tehnik lokalizacije
napak na nivoju metod, jih združi z naučenim modelom (večplastni perceptron, MLP) in preveri, koliko
k lokalizaciji prispeva ocena sumljivosti, ki jo za vsako kandidatno metodo da velik jezikovni model.

Cevovod sestavlja 16 skript v Pythonu, razvrščenih v mape po vsebini. Vsaka skripta prebere rezultate
prejšnjih iz mape `izhod/` in vanjo zapiše svoje. Vrstni red zagona je v tabeli spodaj. Manjši rezultati
so priloženi (tabele JSON, besedila, pozivi, surovi odgovori jezikovnega modela, shranjene napovedi
modelov), zato je mogoče vse tabele in slike naloge obnoviti brez ponovnega učenja in brez klicev
jezikovnega modela.

## Struktura

```
nastavitve.py            poti, konstante in imena izhodnih datotek na enem mestu
skupno/                  skupne funkcije: metrike, podatki, MLP, rankSVM, Defects4J, meje metod (tree-sitter)
priprava_podatkov/       matrika ocen na nivoju stavkov in metod, krive metode iz popravkov
izhodisca/               Ochiai in DStar kot izhodišče na stavkih in metodah
jezikovni_model/         pozivi za jezikovni model in klici prek vmesnika API
modeli/                  naučeni modeli: MLP, rankSVM, prečno preverjanje po projektih
analize/                 polni pogled, primerjava družin tehnik, kvalitativni primeri
porocila/                tabele in slike za besedilo naloge
izhod/                   rezultati, v podmapah z enakimi imeni kot skripte
combinefl/               paket CombineFL (ni v gitu, glej combinefl/README.md)
requirements.txt         različice paketov Python
```

## Vrstni red zagona

| # | skripta | kaj naredi | potrebuje | trajanje |
|---|---|---|---|---|
| 1 | `priprava_podatkov/matrika_stavkov.py` | iz `release.json` (CombineFL) sestavi tabelo ocen devetih tehnik na nivoju stavkov | CombineFL | pod minuto |
| 2 | `izhodisca/ochiai_stavki.py` | izhodišče Ochiai in DStar na nivoju stavkov (348 napak) | 1 | pod minuto |
| 3 | `priprava_podatkov/matrika_metod.py` | delovne kopije Defects4J, krive metode iz razvijalčevega popravka, združevanje stavkov v metode, 343 lokalizabilnih napak | 1, Defects4J | več ur |
| 4 | `izhodisca/ochiai_metode.py` | izhodišče Ochiai in DStar na nivoju metod | 2, 3 | sekunde |
| 5 | `modeli/mlp_h9.py` | MLP nad devetimi klasičnimi ocenami, 60 epoh | 3 | 2 do 3 minute |
| 6 | `jezikovni_model/pozivi.py` | pozivi za jezikovni model (sled sklada in koda do 50 kandidatnih metod na napako) | 3, Defects4J | okoli ure |
| 7 | `jezikovni_model/klici.py` | klici modela Claude prek paketnega vmesnika, štetje žetonov, pošiljanje, prevzem ocen | 6, ključ API | odvisno od paketa |
| 8 | `modeli/mlp_llm.py` | glavni poskus: MLP H9, MLP H9+LLM, kontrolni preizkus s premešanimi ocenami, kaskada, izbira epoh, pet semen, statistični testi, rezultati po projektih | 3, 7 | s predpomnilnikom sekunde, sicer ure (sama izbira epoh okoli 50 minut) |
| 9 | `analize/polni_pogled.py` | napake, pri katerih je model videl vse metode, proti napakam z okrnjenim izborom | 8 | sekunde |
| 10 | `modeli/ranksvm_h9.py` | izhodišče rankSVM (CombineFL) na istih gubah in z isto metriko | 3, 8, svm_rank | s predpomnilnikom sekunde, sicer okoli minute |
| 11 | `modeli/mlp_lopo.py` | prečno preverjanje po projektih (leave-one-project-out) | 3, 7, 8 | s predpomnilnikom sekunde, sicer ure |
| 12 | `modeli/ranksvm_llm.py` | rankSVM z oceno jezikovnega modela kot deseto značilko | 3, 7, 8, 10, svm_rank | s predpomnilnikom sekunde, sicer ure |
| 13 | `analize/kvalitativni_primeri.py` | izbor 20 napak za kvalitativno analizo z dokaznimi izpisi | 6, 7, 8, 10 | sekunde |
| 14 | `analize/primerjava_druzin.py` | primerjava posameznih tehnik in družin, uporabnost, komplementarnost | 3, 8 | pod minuto |
| 15 | `porocila/tabele_poglavje6.py` | tabele šestega poglavja (Markdown) iz rezultatov | 8 do 14 | sekunde |
| 16 | `porocila/slike.py` | slike 5.2, 6.1 in 6.2 (PDF in PNG) iz rezultatov | 8, 14 | sekunde |

Skripte poženete s potjo do datoteke, iz katere koli mape, na primer:

```bash
python modeli/mlp_llm.py
```

Skripte z dolgim zagonom (3, 6, 8, 10, 11, 12) rezultate zapisujejo sproti in prekinjen zagon
nadaljujejo tam, kjer je ostal. Skripti 3 in 6 sprejmeta tudi izbor napak za poskusni zagon, na primer
`python priprava_podatkov/matrika_metod.py lang1,lang23`.

## Okolje

Rezultati so bili izračunani v Ubuntu pod WSL2 (Windows 11) s Pythonom 3.11 v okolju conda.
Različice paketov so v `requirements.txt`, PyTorch je različica za CPU. Učenje MLP teče na eni niti
(`torch.set_num_threads(1)`), da so seštevanja ponovljiva med računalniki.

Poleg paketov Python so potrebni:

- **Defects4J 3.0.1** (https://github.com/rjust/defects4j) z Javo 11. Privzeta pot je `~/defects4j`,
  drugo nastavite z okoljsko spremenljivko `DEFECTS4J_HOME`. Delovne kopije programov gredo v
  `~/d4j_checkouts` (spremenljivka `D4J_CHECKOUTS`). Potrebujeta ju skripti 3 in 6.
- **Paket CombineFL** v mapi `combinefl/` (glej `combinefl/README.md`). Datoteka `release.json` je vhod
  skripte 1, programa `svm_rank_learn` in `svm_rank_classify` pa uporabljata skripti 10 in 12.
- **Ključ API Anthropic** za skripto 7 (okoljska spremenljivka `ANTHROPIC_API_KEY`).

Mapo z rezultati lahko preusmerite s spremenljivko `FL_IZHOD`.

## Kaj je priloženo v `izhod/` in kaj se izračuna znova

```
izhod/podatki/           matriki ocen na nivoju stavkov in metod, razvrstitev vseh 357 napak
izhod/izhodisca/         rangi in povzetki za Ochiai in DStar
izhod/jezikovni_model/   pozivi (pozivi/), manifest, žetoni, surovi odgovori, izluščene ocene, seznama ponovitev
izhod/modeli/            rezultati in napovedi modelov, izbrano število epoh, predpomnilniki napovedi (*_cache/)
izhod/analize/           polni pogled, primerjava družin, dokazni izpisi dvajsetih primerov
izhod/porocila/          tabele šestega poglavja (Markdown) in slike (PDF, PNG)
```

Velike tabele napovedi po metodah (`*_napovedi.csv`, 13 do 21 MB) niso v gitu. Skripte jih zapišejo
znova, s predpomnilnikom v nekaj sekundah. V gitu prav tako niso začasne datoteke LETOR (`modeli/*_tmp/`)
in sprotni zapis skripte 3 (`podatki/matrika_metod_sprotno.csv`).

Predpomnilnik napovedi je označen z zgoščeno vrednostjo nastavitev in vhodnih podatkov (skripte 10, 11
in 12), zato spremenjeni podatki samodejno sprožijo ponoven izračun. Za učenje iz nič izbrišite
ustrezno mapo `*_cache/`.

## Ponovljivost

- Delitev napak v deset gub je določena s semenom 2018 (kot v CombineFL), vse metode iste napake so
  v isti gubi. Uteži MLP se inicializirajo s semeni 2018, 7, 101, 2024 in 31337. Glavne tabele so pri
  prvem semenu, ostala štiri merijo razpršenost.
- Merilo je povsod enako. Znotraj napake metode uredimo padajoče po oceni, enake ocene dobijo povprečni
  rang, rang napake je rang njene najbolje uvrščene krive metode. Top-1 poročamo tudi pri optimistični
  in pesimistični obravnavi enakih ocen.
- Klici jezikovnega modela (model `claude-sonnet-5`, prilagodljivo razmišljanje, paketni vmesnik,
  julij 2026) niso deterministični. Priloženi so vsi surovi odgovori, tako da so vsi nadaljnji koraki
  ponovljivi natančno. Ocene so bile zbrane v treh paketih. Prvi je zajel vse napake, pri 102 napakah
  pa je model ves proračun `max_tokens = 1500` porabil za razmišljanje in vrnil prazen odgovor. Te
  napake so bile ponovno poslane z mejo 8000 (seznam `ponovitev_1.txt`), preostale 22 pa z mejo
  32000 (`ponovitev_2.txt`). Ena napaka (time22) je ostala brez odgovora. Skupni strošek je bil
  13,07 USD.
- Sedem od 357 napak zbirke CombineFL v Defects4J 3.0.1 ni več podprtih (checkout ne uspe), štiri
  popravek odpravi z dodajanjem nove metode in tri s spremembami zunaj metod. Te napake so izločene,
  ostane 343 napak s 459 krivimi metodami in 111.348 kandidatnimi metodami.

## Povezava z nalogo

| del naloge | skripta |
|---|---|
| tabela učinka združevanja v metode (4. poglavje) | `izhodisca/ochiai_stavki.py`, `izhodisca/ochiai_metode.py` |
| tabela 6.1 do 6.4 in slika 6.1 (posamezne tehnike, družine, komplementarnost) | `analize/primerjava_druzin.py` |
| tabela 6.5 (polni in okrnjeni pogled) | `analize/polni_pogled.py` |
| tabela 6.6, 6.7, 6.12 in 6.13, slika 5.2 in 6.2 (glavni poskus, testi, rezultati po projektih) | `modeli/mlp_llm.py` |
| tabela 6.8 in 6.15 (rankSVM z oceno LLM, obravnava enakih ocen) | `modeli/ranksvm_h9.py`, `modeli/ranksvm_llm.py` |
| tabela 6.9, 6.10 in 6.14 (prečno preverjanje po projektih, semena) | `modeli/mlp_lopo.py` |
| poglavje 7 in dodatek B (kvalitativna analiza primerov) | `analize/kvalitativni_primeri.py` |
| vse tabele in slike v obliki za nalogo | `porocila/tabele_poglavje6.py`, `porocila/slike.py` |
