# Tabele šestega poglavja (samodejno iz rezultatov, porocila/tabele_poglavje6.py)

<!-- TABELA 6.1 -->
| Tehnika | Družina | Uporabnost (strogo) | Top-1 (netrivialne) | Top-3 | Top-5 | Top-10 | MFR | MAR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DStar | SBFL | 328 | 92 (83) | 184 | 210 | 243 | 26,10 | 32,37 |
| Ochiai | SBFL | 328 | 89 (80) | 182 | 209 | 245 | 26,07 | 32,30 |
| Metallaxis | MBFL | 283 | 82 (73) | 152 | 182 | 217 | 72,38 | 75,22 |
| MUSE | MBFL | 208 | 60 (51) | 129 | 153 | 187 | 87,59 | 91,79 |
| sled sklada | sled sklada | 83 | 42 (33) | 91 | 109 | 130 | 149,40 | 150,42 |
| slicing_count (frekvenca) | rezanje | 167 | 29 (20) | 84 | 103 | 139 | 148,93 | 149,98 |
| slicing_intersection (presek) | rezanje | 147 | 27 (18) | 78 | 94 | 127 | 152,45 | 152,53 |
| slicing (unija) | rezanje | 165 | 26 (17) | 78 | 96 | 137 | 149,51 | 150,31 |
| menjava predikatov | PS | 121 | 23 (14) | 82 | 99 | 128 | 146,88 | 149,05 |

<!-- TABELA 6.2 -->
| Tehnika | n (podmnožica) | MFR: tehnika / Ochiai | Mediana prvega ranga: tehnika / Ochiai | Top-1: tehnika / Ochiai | Zmage / porazi / vezi (parni n) | Holm p (družina 8) | Sodba |
|---|---:|---:|---:|---:|---:|---:|---|
| DStar | 328 | 27,17 / 27,14 | 3,0 / 3,0 | 83 / 80 | 18 / 9 / 301 (n = 328) | 0,90 | ni značilne razlike |
| Metallaxis | 283 | 19,98 / 23,83 | 3,5 / 3,0 | 73 / 69 | 93 / 112 / 75 (n = 280) | 0,75 | ni značilne razlike |
| MUSE | 208 | 42,27 / 30,83 | 4,75 / 4,25 | 51 / 37 | 94 / 88 / 23 (n = 205) | 0,90 | ni značilne razlike |
| sled sklada | 83 | 30,96 / 11,33 | 2,0 / 2,5 | 33 / 21 | 39 / 22 / 20 (n = 81) | 0,90 | ni značilne razlike |
| slicing (unija) | 165 | 20,23 / 11,65 | 8,0 / 2,5 | 17 / 39 | 38 / 110 / 13 (n = 161) | 6,2 · 10⁻⁸ | Ochiai značilno boljši |
| slicing_count (frekvenca) | 167 | 18,84 / 11,52 | 7,5 / 2,5 | 20 / 41 | 41 / 108 / 14 (n = 163) | 5,3 · 10⁻⁷ | Ochiai značilno boljši |
| slicing_intersection (presek) | 147 | 22,95 / 11,53 | 8,0 / 2,5 | 18 / 37 | 34 / 96 / 13 (n = 143) | 5,3 · 10⁻⁷ | Ochiai značilno boljši |
| menjava predikatov | 121 | 205,42 / 34,84 | 35,5 / 7,5 | 14 / 18 | 29 / 76 / 15 (n = 120) | 2,8 · 10⁻⁹ | Ochiai značilno boljši |

<!-- TABELA 6.3 -->
| Par (A – B) | Predstavnika | Vseh 343: W | Holm p | zmage / porazi / vezi (za A) | smer v prid | Presek: n | W | Holm p | zmage / porazi / vezi (za A) | smer v prid | Opozorilo |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|---|
| SBFL – MBFL | Ochiai – Metallaxis | 10.857,5 | 7,2 · 10⁻⁶ | 156 / 100 / 87 | SBFL | 280 | 9.438 | 0,94 (n. z.) | 112 / 93 / 75 | SBFL | izgubi značilnost |
| SBFL – PS | Ochiai – menjava predikatov | 4.515 | 6,0 · 10⁻³⁵ | 269 / 44 / 30 | SBFL | 120 | 820,5 | 3,6 · 10⁻⁹ | 76 / 29 / 15 | SBFL | — |
| SBFL – rezanje | Ochiai – slicing_count | 5.416 | 4,2 · 10⁻³³ | 264 / 54 / 25 | SBFL | 163 | 2.778,5 | 7,0 · 10⁻⁷ | 108 / 41 / 14 | SBFL | — |
| SBFL – sled sklada | Ochiai – sled sklada | 4.672,5 | 1,3 · 10⁻³³ | 254 / 55 / 34 | SBFL | 81 | 827,5 | 1,0 (n. z.) | 22 / 39 / 20 | sled sklada | izgubi značilnost |
| MBFL – PS | Metallaxis – menjava predikatov | 4.855,5 | 1,4 · 10⁻³⁰ | 251 / 46 / 46 | MBFL | 100 | 732,5 | 2,1 · 10⁻⁷ | 67 / 26 / 7 | MBFL | — |
| MBFL – rezanje | Metallaxis – slicing_count | 4.833 | 3,2 · 10⁻²⁶ | 225 / 51 / 67 | MBFL | 160 | 2.375 | 4,6 · 10⁻⁸ | 110 / 36 / 14 | MBFL | — |
| MBFL – sled sklada | Metallaxis – sled sklada | 4.217,5 | 3,5 · 10⁻²⁶ | 210 / 55 / 78 | MBFL | 75 | 583,5 | 1,0 (n. z.) | 18 / 31 / 26 | sled sklada | izgubi značilnost |
| PS – rezanje | menjava predikatov – slicing_count | 8.609 | 5,0 · 10⁻⁷ | 62 / 176 / 105 | rezanje | 47 | 447 | 1,0 (n. z.) | 25 / 19 / 3 | PS | izgubi značilnost |
| PS – sled sklada | menjava predikatov – sled sklada | 5.332 | 0,0017 | 47 / 126 / 170 | sled sklada | 26 | 91 | 0,99 (n. z.) | 8 / 14 / 4 | sled sklada | izgubi značilnost |
| rezanje – sled sklada | slicing_count – sled sklada | 5.975 | 0,0074 | 102 / 74 / 167 | rezanje | 58 | 176 | 0,0034 | 8 / 34 / 16 | sled sklada | OBRNITEV SMERI |
<!-- 6.3 kontrola: izgubi značilnost = 5, obrnitev smeri = 1 -->

<!-- TABELA 6.4 -->
| Količina (Top-1) | Vrednost |
|---|---:|
| unikatno rešene napake: MUSE | 17 |
| unikatno rešene napake: Metallaxis | 11 |
| unikatno rešene napake: sled sklada | 8 |
| unikatno rešene napake: menjava predikatov | 4 |
| unikatno rešene napake: DStar | 2 |
| unikatno rešene napake: slicing_count | 1 |
| unikatno rešene napake skupaj | 43 |
| Ochiai ∪ DStar (kombiniranje znotraj SBFL) | 92 |
| najboljša posamezna tehnika (DStar) | 92 |
| zgornja meja popolnega izbiralnika (ne rezultat) | 156 |
| dosežena naučena kombinacija (razdelek 6.3) | 141 |
| delež zgornje meje | 141 / 156 ≈ 90 % |

<!-- TABELA 6.5 -->
| Pogled | Model | Top-1 | Top-3 | Top-5 | Top-10 | MFR |
|---|---|---:|---:|---:|---:|---:|
| polni pogled (n = 151, pokritost izbora 100 %) | LLM sam | 122 | 145 | 146 | 151 | 1,39 |
| polni pogled (n = 151, pokritost izbora 100 %) | klasična kombinacija H9 | 86 | 127 | 142 | 148 | 2,16 |
| polni pogled (n = 151, pokritost izbora 100 %) | H9+LLM | 127 | 146 | 146 | 150 | 1,40 |
| polni pogled (n = 151, pokritost izbora 100 %) | kaskada | 122 | 145 | 146 | 151 | 1,39 |
| okrnjeni pogled (n = 192, pokritost izbora 92,2 %) | LLM sam | 126 | 158 | 165 | 170 | 39,67 |
| okrnjeni pogled (n = 192, pokritost izbora 92,2 %) | klasična kombinacija H9 | 55 | 97 | 119 | 135 | 26,49 |
| okrnjeni pogled (n = 192, pokritost izbora 92,2 %) | H9+LLM | 128 | 159 | 165 | 169 | 20,53 |
| okrnjeni pogled (n = 192, pokritost izbora 92,2 %) | kaskada | 126 | 158 | 165 | 170 | 21,72 |

<!-- TABELA 6.6 -->
| Model | Top-1 | Top-3 | Top-5 | Top-10 | MFR | MAR |
|---|---:|---:|---:|---:|---:|---:|
| Ochiai | 89 | 182 | 209 | 245 | 26,07 | 32,30 |
| rankSVM (surove značilke) | 149 | 228 | 250 | 275 | 14,93 | 20,12 |
| MLP H9 (kontrola) | 141 | 224 | 261 | 283 | 15,78 | 21,39 |
| **MLP H9+LLM** | 255 | 305 | 311 | 319 | 12,11 | 17,61 |
| MLP H9 s premešanimi ocenami LLM (kontrolni preizkus, 6.4) | 124 | 204 | 251 | 278 | 16,13 | 21,35 |
| kaskada LLM→H9 (brez dodatnega učenja, 6.5) | 248 | 303 | 311 | 321 | 12,77 | 18,23 |
| LLM sam (ožji izbor + preureditev) | 248 | 303 | 311 | 321 | 22,81 | 32,05 |

<!-- TABELA 6.7 (n = 327) -->
| Model | MFR | Top-1 |
|---|---:|---:|
| LLM sam | 1,76 | 248 |
| kaskada | 1,76 | 248 |
| **H9+LLM** | 1,91 | 255 |
| H9 | 5,15 | 141 |
| H9 s premešanimi ocenami | 6,00 | 124 |
| Ochiai | 15,32 | 89 |
<!-- 6.7 MAR samo pokrite: H9+LLM 7,07, kaskada 7,06, LLM sam 11,45 -->

<!-- TABELA 6.8 -->
| Model | Top-1 | Top-3 | Top-5 | Top-10 | MFR | MAR | Top-1: optimistična / povprečna / pesimistična |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ochiai | 89 | 182 | 209 | 245 | 26,07 | 32,30 | 162 / 89 / 89 |
| rankSVM, surove značilke | 149 | 228 | 250 | 275 | 14,93 | 20,12 | 160 / 149 / 149 |
| rankSVM, min-max | 147 | 227 | 248 | 276 | 14,96 | 20,14 | 159 / 147 / 147 |
| **rankSVM+LLM, surove značilke** | 256 | 302 | 310 | 320 | 11,80 | 17,10 | 259 / 256 / 256 |
| rankSVM+LLM, min-max | 239 | 291 | 303 | 320 | 12,11 | 17,42 | 242 / 239 / 239 |
| MLP H9 | 141 | 224 | 261 | 283 | 15,78 | 21,39 | 154 / 141 / 141 |
| MLP H9+LLM | 255 | 305 | 311 | 319 | 12,11 | 17,61 | 257 / 255 / 255 |
<!-- utež LLM: surove 0,0691 ± 0,0138, min-max 1,938 ± 0,0853 -->

<!-- TABELA 6.9a (LOPO po projektih, primarno seme) -->
| Projekt (napak) | MFR: H9 → H9+LLM | Top-1: H9 → H9+LLM |
|---|---:|---:|
| Chart (25) | 12,06 → 23,82 | 9 → 19 |
| Closure (130) | 80,62 → 26,69 | 32 → 92 |
| Lang (59) | 1,70 → 1,05 | 43 → 56 |
| Math (104) | 6,10 → 4,08 | 46 → 73 |
| Time (25) | 6,82 → 2,36 | 8 → 18 |

<!-- TABELA 6.9b (LOPO združeno, primarno seme) -->
| Model | Top-1 | MFR |
|---|---:|---:|
| Ochiai (nenaučena referenca) | 89 | 26,07 |
| H9 | 138 | 34,08 |
| H9+LLM | 258 | 13,44 |

<!-- TABELA 6.10 -->
| Model | MFR, desetkratna CV | MFR, LOPO | Sprememba | MFR po semenih (LOPO): razpon | povprečje ± st. odklon |
|---|---:|---:|---:|---:|---:|
| H9 | 15,78 | 34,08 | +116 % | 23,1 … 68,0 | 37,18 ± 17,91 |
| H9+LLM | 12,11 | 13,44 | +11 % | 11,9 … 24,0 | 16,32 ± 5,35 |
<!-- razlika MFR: primarno seme -20,63, povprečje -20,87 ± 21,01, vrednosti ['-20,63', '0,90', '-14,37', '-55,65', '-14,57'], izboljšanih semen 4 | razlika Top-1: [120, 119, 117, 116, 113] povprečje 117,0 ± 2,7, izboljšanih 5 -->
<!-- LOPO p (napake): 2018: 2,8 · 10⁻²⁰ (175 / 32 / 136), 7: 1,6 · 10⁻¹⁸ (179 / 32 / 132), 101: 1,1 · 10⁻¹⁸ (174 / 33 / 136), 2024: 1,6 · 10⁻²² (178 / 27 / 138), 31337: 5,5 · 10⁻²⁴ (178 / 26 / 139) -->
<!-- LOPO p (projekti): 2018: 0.21875 (4/5), 7: 0.3125 (4/5), 101: 0.03125 (5/5), 2024: 0.03125 (5/5), 31337: 0.03125 (5/5) -->

<!-- TABELA 6.12 -->
| Projekt (napak) | MFR: H9 → H9+LLM | Top-1: H9 → H9+LLM |
|---|---:|---:|
| Chart (25) | 25,10 → 11,30 | 11 → 20 |
| Closure (130) | 30,13 → 25,43 | 34 → 90 |
| Lang (59) | 1,78 → 1,05 | 42 → 56 |
| Math (104) | 5,71 → 4,22 | 46 → 72 |
| Time (25) | 6,78 → 2,56 | 8 → 17 |

<!-- TABELA 6.13 -->
| Primerjava | Test | p | Zmage / porazi / vezi |
|---|---|---:|---:|
| H9+LLM proti H9 (glavna primerjava, desetkratna CV) | enostranski | 2,8 · 10⁻²¹ | 169 / 31 / 143 |
| isto na ravni projektov (desetkratna CV; n = 5 parov MFR) | enostranski | 0,03125 | — |
| H9+LLM proti H9 (LOPO, raven napak, primarno seme) | enostranski | 2,8 · 10⁻²⁰ | 175 / 32 / 136 |
| isto na ravni projektov (LOPO, primarno seme) | enostranski | 0,219 | — |
| H9 proti H9 s premešanimi ocenami (kontrolni preizkus) | enostranski | 1,9 · 10⁻¹⁰ | 141 / 37 / 165 |
| H9+LLM proti LLM samemu | enostranski | 0,027 | 41 / 22 / 280 |
| H9+LLM proti kaskadi | enostranski | 0,27 (ni značilno) | 34 / 22 / 287 |
| rankSVM proti Ochiai | enostranski | 1,5 · 10⁻¹⁴ | 168 / 53 / 122 |
| H9+LLM proti rankSVM | enostranski | 1,2 · 10⁻²⁰ | 163 / 28 / 152 |
| MLP H9 proti rankSVM (brez LLM) | dvostranski | 0,14 | 73 / 62 / 208 |
| rankSVM+LLM proti rankSVM (surove značilke) | enostranski | 3,1 · 10⁻²⁰ | 163 / 30 / 150 |
| rankSVM+LLM proti rankSVM (min-max) | enostranski | 6,0 · 10⁻²¹ | 163 / 28 / 152 |
| rankSVM+LLM proti MLP H9+LLM (surove značilke) | dvostranski | 0,83 | 18 / 20 / 305 |
| rankSVM+LLM proti MLP H9+LLM (min-max) | dvostranski | 0,0053 | 23 / 55 / 265 |
| rankSVM+LLM: surove značilke proti min-max (isti model) | dvostranski | 0,0005 | 49 / 19 / 275 |
| LLM sam proti H9 (polni pogled, n = 151) | enostranski | 1,4 · 10⁻⁷ | 58 / 14 / 79 |
| H9+LLM proti LLM samemu (polni pogled, n = 151) | enostranski | 0,55 (test skoraj brez moči) | 9 / 5 / 137 |
<!-- LOPO raven napak, razpon p čez semena: 5,5 · 10⁻²⁴ … 1,6 · 10⁻¹⁸ -->

<!-- TABELA 6.14 -->
| Protokol | Količina | Povprečje ± st. odklon | Izboljšanje po semenih |
|---|---|---:|---|
| desetkratna CV | Δ Top-1 (H9+LLM − H9) | +111,0 ± 2,1 | 5 od 5 |
| desetkratna CV | Δ MFR | −3,60 ± 0,11 | 5 od 5 |
| LOPO | Δ Top-1 (H9+LLM − H9) | +117,0 ± 2,7 | 5 od 5 (vrednosti 113–120) |
| LOPO | Δ MFR | −20,87 ± 21,01 | 4 od 5 (pri enem semenu se predznak obrne) |

<!-- TABELA 6.15 -->
| Model | optimistična (najmanjši rang) | povprečna | pesimistična (največji rang) |
|---|---:|---:|---:|
| Ochiai | 162 | 89 | 89 |
| rankSVM, surove značilke | 160 | 149 | 149 |
| rankSVM, min-max | 159 | 147 | 147 |
| MLP H9 | 154 | 141 | 141 |
| **MLP H9+LLM** | 257 | 255 | 255 |
| rankSVM+LLM, surove značilke | 259 | 256 | 256 |
| rankSVM+LLM, min-max | 242 | 239 | 239 |

<!-- TABELA 6.12z (desetkratna CV po projektih + združeno) -->
| Projekt (napak) | MFR: H9 → H9+LLM | Top-1: H9 → H9+LLM |
|---|---:|---:|
| Chart (25) | 25,10 → 11,30 | 11 → 20 |
| Closure (130) | 30,13 → 25,43 | 34 → 90 |
| Lang (59) | 1,78 → 1,05 | 42 → 56 |
| Math (104) | 5,71 → 4,22 | 46 → 72 |
| Time (25) | 6,78 → 2,56 | 8 → 17 |
| **Združeno (343)** | 15,78 → 12,11 | 141 → 255 |

<!-- TABELA 6.9z (LOPO po projektih + združeno, primarno seme) -->
| Projekt (napak) | MFR: H9 → H9+LLM | Top-1: H9 → H9+LLM |
|---|---:|---:|
| Chart (25) | 12,06 → 23,82 | 9 → 19 |
| Closure (130) | 80,62 → 26,69 | 32 → 92 |
| Lang (59) | 1,70 → 1,05 | 43 → 56 |
| Math (104) | 6,10 → 4,08 | 46 → 73 |
| Time (25) | 6,82 → 2,36 | 8 → 18 |
| **Združeno (343)** | 34,08 → 13,44 | 138 → 258 |
