# -*- coding: utf-8 -*-
"""Večplastni perceptron (MLP), ki se nauči kombinirati ocene posameznih tehnik v eno oceno sumljivosti."""
import numpy as np
import torch
import torch.nn as nn

from skupno.metrike import metrike_po_napakah

torch.set_num_threads(1)      # ena nit, da so seštevanja ponovljiva med računalniki


class MLP(nn.Module):
    """Dva skrita sloja (64 in 32 nevronov), ReLU, izpuščanje 0,2, en izhodni logit."""

    def __init__(self, d, h1=64, h2=32, p=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h1), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(p),
                                 nn.Linear(h2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def uci_z_mejniki(Xtr, ytr, Xev, d, seme, mejniki):
    """Nauči en MLP do največjega mejnika in vrne napovedi na Xev pri vsakem mejniku ({epohe: napovedi})."""
    torch.manual_seed(seme)
    model = MLP(d=d)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    pos = float(ytr.sum())
    neg = float(len(ytr) - pos)
    # utež pozitivnega razreda uravna neuravnoteženost (okoli 459 krivih od 111.348 metod)
    izguba = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)]))
    zeleni, out, najvec = set(mejniki), {}, max(mejniki)
    for ep in range(1, najvec + 1):
        model.train()
        opt.zero_grad()
        loss = izguba(model(Xtr), ytr)
        loss.backward()
        opt.step()
        if ep in zeleni:
            model.eval()
            with torch.no_grad():
                out[ep] = model(Xev).numpy().astype(np.float64)
    return out


def napovedi_cv(dfn, df, znacilke, gube, napaka_vrstice, seme, epohe):
    """Prečno preverjanje po napakah, guba k dobi seme (seme + k). Vrne napovedi, poravnane z vrsticami df."""
    X = dfn[znacilke].to_numpy(dtype=np.float32)
    y = df["faulty"].to_numpy(dtype=np.float32)
    napoved = np.full(len(df), np.nan, dtype=np.float64)
    for k, testne in enumerate(gube, 1):
        te = np.isin(napaka_vrstice, list(testne))
        out = uci_z_mejniki(torch.from_numpy(X[~te]), torch.from_numpy(y[~te]),
                            torch.from_numpy(X[te]), len(znacilke), seme + k, [epohe])
        napoved[te] = out[epohe]
    return napoved


def izberi_epohe(dfn, df, znacilke, gube, napaka_vrstice, seme, mreza, delez_validacije, oznaka):
    """Izbere število epoh po MFR na notranji validacijski gubi, testne napake ostanejo nedotaknjene.

    Vrne (izbrane epohe, {epohe: validacijski MFR}).
    """
    X = dfn[znacilke].to_numpy(dtype=np.float32)
    y = df["faulty"].to_numpy(dtype=np.float32)
    val_napoved = {ep: np.full(len(df), np.nan) for ep in mreza}
    rng = np.random.default_rng(seme)
    for k, testne in enumerate(gube, 1):
        te = np.isin(napaka_vrstice, list(testne))
        ucne_napake = np.array(sorted(set(napaka_vrstice[~te])))
        perm = rng.permutation(len(ucne_napake))
        n_val = max(1, int(round(delez_validacije * len(ucne_napake))))
        validacijske = set(ucne_napake[perm[:n_val]])
        iv = np.isin(napaka_vrstice, list(validacijske))
        it = (~te) & (~iv)
        out = uci_z_mejniki(torch.from_numpy(X[it]), torch.from_numpy(y[it]),
                            torch.from_numpy(X[iv]), len(znacilke), seme + k, mreza)
        for ep, p in out.items():
            val_napoved[ep][iv] = p
    krivulja = {}
    for ep in mreza:
        tmp = df.copy()
        tmp["_v"] = val_napoved[ep]
        podmnozica = set(tmp.loc[tmp["_v"].notna(), "fault_id"])
        _, mfr, _, _ = metrike_po_napakah(tmp[tmp["_v"].notna()], "_v", podmnozica=podmnozica)
        krivulja[ep] = mfr
    izbrane = min(krivulja, key=krivulja.get)
    print(f"  [{oznaka}] validacijski MFR po epohah: "
          + ", ".join(f"{ep}: {krivulja[ep]:.2f}" for ep in mreza) + f"  -> izbrano {izbrane}")
    return izbrane, krivulja
