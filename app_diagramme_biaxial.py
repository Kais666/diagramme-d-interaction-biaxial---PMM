"""
Interface Streamlit — Diagramme d'interaction biaxial N-My-Mz (Eurocode 2)
===========================================================================

Interface graphique permettant de configurer une section (rectangulaire ou
circulaire), son ferraillage et ses matériaux, puis de :
  - visualiser la section et sa discrétisation,
  - calculer et visualiser la surface d'interaction 3D (N, My, Mz),
  - explorer une coupe interactive à N constant (My-Mz),
  - vérifier un point de sollicitation (N_Ed, My_Ed, Mz_Ed),
  - exporter les résultats en Excel et un rapport en PDF.

Lancer avec :  streamlit run app_diagramme_biaxial.py
"""

import io
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.backends.backend_pdf import PdfPages
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# MOTEUR DE CALCUL (fonctions pures, paramétrées — pas de globales)
# ============================================================

def sigma_beton(eps, fcd, e0, ebu):
    """Loi parabole-rectangle du béton (EC2), vectorisée."""
    eps = np.atleast_1d(eps)
    sig = np.zeros_like(eps, dtype=float)
    m_parab = (eps <= 0) & (eps >= e0)
    sig[m_parab] = fcd * (1 - (1 - eps[m_parab] / e0) ** 2)
    m_rect = (eps < e0) & (eps >= ebu)
    sig[m_rect] = fcd
    return sig


def sigma_acier(eps, Es, fyd):
    """Loi bilinéaire élasto-plastique de l'acier, vectorisée."""
    return np.clip(Es * eps, -fyd, fyd)


def generer_fibres_beton(type_section, b, h, D, nx_beton, ny_beton):
    if type_section == "rectangulaire":
        dx, dy = b / nx_beton, h / ny_beton
        xs = np.linspace(-b/2 + dx/2, b/2 - dx/2, nx_beton)
        ys = np.linspace(-h/2 + dy/2, h/2 - dy/2, ny_beton)
        Xc, Yc = np.meshgrid(xs, ys)
        Xc, Yc = Xc.ravel(), Yc.ravel()
        dA = np.full_like(Xc, dx * dy)
        return Xc, Yc, dA
    else:  # circulaire
        R = D / 2
        dx = dy = (2 * R) / nx_beton
        xs = np.linspace(-R + dx/2, R - dx/2, nx_beton)
        ys = np.linspace(-R + dy/2, R - dy/2, ny_beton)
        Xc, Yc = np.meshgrid(xs, ys)
        Xc, Yc = Xc.ravel(), Yc.ravel()
        mask = Xc**2 + Yc**2 <= R**2
        Xc, Yc = Xc[mask], Yc[mask]
        dA = np.full_like(Xc, dx * dy)
        return Xc, Yc, dA


def generer_armatures_perimetre_rect(b, h, d_p, nx_barres, ny_barres, diam_mm):
    A_barre = np.pi * (diam_mm / 1000 / 2) ** 2
    xs, ys = [], []
    x_horiz = np.linspace(-b/2 + d_p, b/2 - d_p, nx_barres)
    xs += list(x_horiz); ys += [h/2 - d_p] * nx_barres
    xs += list(x_horiz); ys += [-h/2 + d_p] * nx_barres
    if ny_barres > 2:
        y_vert = np.linspace(-h/2 + d_p, h/2 - d_p, ny_barres)[1:-1]
        xs += [-b/2 + d_p] * len(y_vert); ys += list(y_vert)
        xs += [b/2 - d_p] * len(y_vert);  ys += list(y_vert)
    x, y = np.array(xs), np.array(ys)
    A = np.full(len(x), A_barre)
    return x, y, A


def generer_armatures_multi_couronnes_circ(D, nappes):
    """nappes : liste de 1 à 4 dicts {n_barres, diam_mm, d_p}, chacun formant
    une couronne de barres à son propre enrobage."""
    xs, ys, As = [], [], []
    for nappe in nappes:
        n_barres, diam_mm, d_p = nappe["n_barres"], nappe["diam_mm"], nappe["d_p"]
        A_barre = np.pi * (diam_mm / 1000 / 2) ** 2
        r_s = D / 2 - d_p
        theta = np.linspace(0, 2*np.pi, n_barres, endpoint=False)
        xs.append(r_s * np.cos(theta))
        ys.append(r_s * np.sin(theta))
        As.append(np.full(n_barres, A_barre))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(As)


def get_U_extremes(theta, type_section, b, h, D):
    if type_section == "rectangulaire":
        corners = [(-b/2, -h/2), (b/2, -h/2), (b/2, h/2), (-b/2, h/2)]
        u_vals = [cx*np.cos(theta) + cy*np.sin(theta) for cx, cy in corners]
        return max(u_vals), min(u_vals)
    else:
        R = D / 2
        return R, -R


def calcul_N_My_Mz(eps_bc, eps_bt, theta, U_other, H_equiv,
                    Xc, Yc, dA, Xa, Ya, DA, fcd, e0, ebu, Es, fyd):
    eps_bc = np.atleast_1d(np.asarray(eps_bc, dtype=float))
    eps_bt = np.atleast_1d(np.asarray(eps_bt, dtype=float))

    ct, st_ = np.cos(theta), np.sin(theta)

    # ---- béton ----
    u_c = Xc * ct + Yc * st_
    w_c = u_c - U_other
    eps_c = eps_bt[:, None] + (eps_bc[:, None] - eps_bt[:, None]) * (w_c[None, :] / H_equiv)
    sig_c = sigma_beton(eps_c.ravel(), fcd, e0, ebu).reshape(eps_c.shape)

    N_b  = np.sum(sig_c * dA[None, :], axis=1)
    My_b = np.sum(sig_c * dA[None, :] * Xc[None, :], axis=1)
    Mz_b = np.sum(sig_c * dA[None, :] * Yc[None, :], axis=1)

    # ---- acier ----
    u_a = Xa * ct + Ya * st_
    w_a = u_a - U_other
    eps_a = eps_bt[:, None] + (eps_bc[:, None] - eps_bt[:, None]) * (w_a[None, :] / H_equiv)
    sigA = sigma_acier(eps_a, Es, fyd)
    sigB = sigma_beton(eps_a.ravel(), fcd, e0, ebu).reshape(eps_a.shape)
    contrib = (-sigA - sigB) * DA[None, :]

    N_a  = np.sum(contrib, axis=1)
    My_a = np.sum(contrib * Xa[None, :], axis=1)
    Mz_a = np.sum(contrib * Ya[None, :], axis=1)

    N, My, Mz = N_b + N_a, My_b + My_a, Mz_b + Mz_a
    if N.shape[0] == 1:
        return N[0], My[0], Mz[0]
    return N, My, Mz


def calcul_branche_theta(theta, type_section, b, h, D, Xc, Yc, dA, Xa, Ya, DA,
                          fcd, e0, ebu, Es, fyd, esu, ecu, N_PTS, EPS_GEOM):
    U_comp, U_other = get_U_extremes(theta, type_section, b, h, D)
    H_equiv = U_comp - U_other

    ct, st_ = np.cos(theta), np.sin(theta)
    u_a = Xa * ct + Ya * st_
    u_t = np.min(u_a)
    y_t = u_t - U_other

    if H_equiv < EPS_GEOM or (H_equiv - y_t) < EPS_GEOM:
        raise ValueError(f"Géométrie dégénérée pour theta={theta:.3f} rad")

    yc_piv = H_equiv - 3 * H_equiv / 7

    def NMM(eps_bc, eps_bt):
        return calcul_N_My_Mz(eps_bc, eps_bt, theta, U_other, H_equiv,
                               Xc, Yc, dA, Xa, Ya, DA, fcd, e0, ebu, Es, fyd)

    # ---- Pivot A ----
    eps_comp_range = np.linspace(esu, ebu, N_PTS)
    pente_A = (eps_comp_range - esu) / (H_equiv - y_t)
    eps_bt_A = esu + pente_A * (0 - y_t)
    eps_bc_A = esu + pente_A * (H_equiv - y_t)
    N_A, My_A, Mz_A = NMM(eps_bc_A, eps_bt_A)
    eps_other_A_end = eps_bt_A[-1]

    pente_depart_C = (ecu - ebu) / (yc_piv - H_equiv)
    eps_other_C_start = ecu + pente_depart_C * (0 - yc_piv)

    # ---- Pivot B ----
    eps_other_range = np.linspace(eps_other_A_end, eps_other_C_start, N_PTS)
    eps_bc_B = np.full(N_PTS, ebu)
    eps_bt_B = eps_other_range
    N_B, My_B, Mz_B = NMM(eps_bc_B, eps_bt_B)

    # ---- Pivot C ----
    pente_range = np.linspace(pente_depart_C, 0, N_PTS)
    eps_bc_C = ecu + pente_range * (H_equiv - yc_piv)
    eps_bt_C = ecu + pente_range * (0 - yc_piv)
    N_C, My_C, Mz_C = NMM(eps_bc_C, eps_bt_C)

    N_full  = np.concatenate([N_A, N_B, N_C])
    My_full = np.concatenate([My_A, My_B, My_C])
    Mz_full = np.concatenate([Mz_A, Mz_B, Mz_C])
    return N_full, My_full, Mz_full


def coupe_a_N_constant(N_cible, branches_N, branches_My, branches_Mz, n_theta):
    My_coupe, Mz_coupe = [], []
    for i in range(n_theta):
        N_i = branches_N[i]
        if N_cible < N_i.min() or N_cible > N_i.max():
            continue
        My_i = np.interp(N_cible, N_i, branches_My[i])
        Mz_i = np.interp(N_cible, N_i, branches_Mz[i])
        My_coupe.append(My_i)
        Mz_coupe.append(Mz_i)
    return np.array(My_coupe), np.array(Mz_coupe)


def trouver_croisements_axe(vals_axe, vals_autre):
    n = len(vals_axe)
    pts = []
    for i in range(n):
        j = (i + 1) % n
        if vals_axe[i] == 0:
            pts.append(vals_autre[i])
        elif vals_axe[i] * vals_axe[j] < 0:
            t = -vals_axe[i] / (vals_axe[j] - vals_axe[i])
            pts.append(vals_autre[i] + t * (vals_autre[j] - vals_autre[i]))
    return pts


def tableau_points_critiques(N_cible_kN, branches_N, branches_My, branches_Mz, n_theta):
    My_c, Mz_c = coupe_a_N_constant(N_cible_kN * 1e3, branches_N, branches_My, branches_Mz, n_theta)
    if len(My_c) < 3:
        return None
    My_c_kNm, Mz_c_kNm = My_c / 1e3, Mz_c / 1e3

    croisements_Mz0 = trouver_croisements_axe(Mz_c_kNm, My_c_kNm)
    croisements_My0 = trouver_croisements_axe(My_c_kNm, Mz_c_kNm)

    lignes = [
        ("My max", My_c_kNm.max(), Mz_c_kNm[np.argmax(My_c_kNm)]),
        ("My min", My_c_kNm.min(), Mz_c_kNm[np.argmin(My_c_kNm)]),
        ("Mz max", My_c_kNm[np.argmax(Mz_c_kNm)], Mz_c_kNm.max()),
        ("Mz min", My_c_kNm[np.argmin(Mz_c_kNm)], Mz_c_kNm.min()),
    ]
    for i, v in enumerate(croisements_Mz0):
        lignes.append((f"Mz=0 (#{i+1})", v, 0.0))
    for i, v in enumerate(croisements_My0):
        lignes.append((f"My=0 (#{i+1})", 0.0, v))

    df = pd.DataFrame(lignes, columns=["Point", "My (kN.m)", "Mz (kN.m)"])
    df.insert(0, "N (kN)", N_cible_kN)
    return df


def tableau_contour_complet(N_cible_kN, branches_N, branches_My, branches_Mz, n_theta):
    My_c, Mz_c = coupe_a_N_constant(N_cible_kN * 1e3, branches_N, branches_My, branches_Mz, n_theta)
    df = pd.DataFrame({"My (kN.m)": My_c / 1e3, "Mz (kN.m)": Mz_c / 1e3})
    df.insert(0, "N (kN)", N_cible_kN)
    return df


# ============================================================
# CALCUL COMPLET (mis en cache — ne se relance que si les paramètres changent)
# ============================================================

@st.cache_data(show_spinner="Calcul de la surface d'interaction (balayage angulaire)...")
def calculer_tout(cfg):
    fcd = cfg["fck"] * 1e6 / cfg["gamma_c"]
    fyd = cfg["fyk"] * 1e6 / cfg["gamma_s"]
    Es = cfg["Es"] * 1e9

    Xc, Yc, dA = generer_fibres_beton(
        cfg["type_section"], cfg["b"], cfg["h"], cfg["D"], cfg["nx_beton"], cfg["ny_beton"]
    )

    if cfg["type_section"] == "rectangulaire":
        Xa, Ya, DA = generer_armatures_perimetre_rect(
            cfg["b"], cfg["h"], cfg["d_p"], cfg["nx_barres"], cfg["ny_barres"], cfg["diam_mm"]
        )
    else:
        Xa, Ya, DA = generer_armatures_multi_couronnes_circ(cfg["D"], cfg["nappes"])

    esu, ebu, e0, ecu = 10e-3, -3.5e-3, -2e-3, -2e-3
    EPS_GEOM = 1e-9
    N_PTS = cfg["n_pts"]
    N_THETA = cfg["n_theta"]

    thetas = np.linspace(0, 2*np.pi, N_THETA, endpoint=False)
    branches_N, branches_My, branches_Mz = [], [], []
    for th in thetas:
        N_th, My_th, Mz_th = calcul_branche_theta(
            th, cfg["type_section"], cfg["b"], cfg["h"], cfg["D"],
            Xc, Yc, dA, Xa, Ya, DA, fcd, e0, ebu, Es, fyd,
            esu, ecu, N_PTS, EPS_GEOM
        )
        branches_N.append(N_th)
        branches_My.append(My_th)
        branches_Mz.append(Mz_th)

    branches_N = np.array(branches_N)
    branches_My = np.array(branches_My)
    branches_Mz = np.array(branches_Mz)

    return {
        "Xc": Xc, "Yc": Yc, "dA": dA, "Xa": Xa, "Ya": Ya, "DA": DA,
        "branches_N": branches_N, "branches_My": branches_My, "branches_Mz": branches_Mz,
        "fcd": fcd, "fyd": fyd, "Es": Es,
    }


# ============================================================
# FIGURES
# ============================================================

def figure_section(cfg, res):
    fig, ax = plt.subplots(figsize=(5, 5))
    if cfg["type_section"] == "rectangulaire":
        ax.add_patch(plt.Rectangle((-cfg["b"]/2, -cfg["h"]/2), cfg["b"], cfg["h"],
                                    facecolor='#212121', edgecolor='black', lw=1.5, zorder=1))
    else:
        ax.add_patch(plt.Circle((0, 0), cfg["D"]/2, facecolor='#212121',
                                 edgecolor='black', lw=1.5, zorder=1))
    ax.scatter(res["Xc"], res["Yc"], s=2, color='#555555', alpha=0.3, zorder=2, label='Fibres béton')
    ax.scatter(res["Xa"], res["Ya"], s=80, color='#B0BEC5', edgecolor='black', zorder=3, label='Barres acier')
    ax.set_aspect('equal')
    ax.set_title("Section et discrétisation")
    ax.legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    return fig


def figure_surface_3d(res):
    X = res["branches_My"] / 1e3
    Y = res["branches_Mz"] / 1e3
    Z = res["branches_N"] / 1e3

    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Viridis',
                                      colorbar=dict(title="N (kN)"))])
    fig.update_layout(
        scene=dict(xaxis_title="My (kN.m)", yaxis_title="Mz (kN.m)", zaxis_title="N (kN)"),
        title="Surface d'interaction N-My-Mz",
        margin=dict(l=0, r=0, b=0, t=40),
        height=650,
    )
    return fig


def figure_coupe(N_cible_kN, res, n_theta):
    My_c, Mz_c = coupe_a_N_constant(N_cible_kN * 1e3, res["branches_N"], res["branches_My"],
                                     res["branches_Mz"], n_theta)
    fig, ax = plt.subplots(figsize=(6, 6))
    if len(My_c) >= 3:
        My_f = np.append(My_c, My_c[0]) / 1e3
        Mz_f = np.append(Mz_c, Mz_c[0]) / 1e3
        ax.plot(My_f, Mz_f, '-o', color='#2196F3', ms=3)
        ax.fill(My_f, Mz_f, color='#90CAF9', alpha=0.3)
    else:
        ax.text(0.5, 0.5, "N hors de la plage atteignable\npour cette section",
                ha='center', va='center', transform=ax.transAxes)
    ax.axhline(0, color='black', lw=0.6, alpha=0.4)
    ax.axvline(0, color='black', lw=0.6, alpha=0.4)
    ax.set_xlabel('My (kN.m)')
    ax.set_ylabel('Mz (kN.m)')
    ax.set_title(f"Coupe à N = {N_cible_kN:.0f} kN")
    ax.set_aspect('equal')
    ax.grid(alpha=0.3, ls='--')
    plt.tight_layout()
    return fig, My_c, Mz_c


def figure_verification(N_Ed, My_Ed, Mz_Ed, res, n_theta):
    My_c, Mz_c = coupe_a_N_constant(N_Ed * 1e3, res["branches_N"], res["branches_My"],
                                     res["branches_Mz"], n_theta)
    if len(My_c) < 3:
        return None, None

    poly_pts = np.column_stack([My_c/1e3, Mz_c/1e3])
    path = Path(poly_pts)
    est_dedans = path.contains_point((My_Ed, Mz_Ed))

    fig, ax = plt.subplots(figsize=(6, 6))
    poly_ferme = np.vstack([poly_pts, poly_pts[0]])
    ax.plot(poly_ferme[:, 0], poly_ferme[:, 1], '-', color='#2196F3')
    ax.fill(poly_ferme[:, 0], poly_ferme[:, 1], color='#90CAF9', alpha=0.3)
    couleur_pt = '#4CAF50' if est_dedans else '#E53935'
    ax.plot(My_Ed, Mz_Ed, 'o', color=couleur_pt, ms=12, markeredgecolor='black', zorder=5)
    ax.annotate(f"({My_Ed:.1f} ; {Mz_Ed:.1f})", (My_Ed, Mz_Ed),
                textcoords="offset points", xytext=(10, 10), fontweight='bold', color=couleur_pt)
    ax.axhline(0, color='black', lw=0.6, alpha=0.4)
    ax.axvline(0, color='black', lw=0.6, alpha=0.4)
    ax.set_xlabel('My (kN.m)'); ax.set_ylabel('Mz (kN.m)')
    ax.set_title(f"Vérification à N_Ed = {N_Ed:.0f} kN")
    ax.set_aspect('equal'); ax.grid(alpha=0.3, ls='--')
    plt.tight_layout()
    return fig, est_dedans


# ============================================================
# EXPORTS (Excel + PDF, en mémoire pour st.download_button)
# ============================================================

def exporter_excel_bytes(N_cible_kN, res, n_theta):
    df_crit = tableau_points_critiques(N_cible_kN, res["branches_N"], res["branches_My"], res["branches_Mz"], n_theta)
    df_contour = tableau_contour_complet(N_cible_kN, res["branches_N"], res["branches_My"], res["branches_Mz"], n_theta)
    if df_crit is None:
        return None
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_contour.round(3).to_excel(writer, sheet_name="Contour complet", index=False)
        df_crit.round(3).to_excel(writer, sheet_name="Points critiques", index=False)
    buffer.seek(0)
    return buffer


def generer_rapport_pdf_bytes(cfg, res, N_cible_kN, n_theta):
    df_crit = tableau_points_critiques(N_cible_kN, res["branches_N"], res["branches_My"], res["branches_Mz"], n_theta)
    if df_crit is None:
        return None

    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:

        # ---- Page 1 : résumé géométrie / matériaux ----
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis('off')
        ax.text(0.97, 0.99, "Cet outil de calcul est développé par Charfi Kaies",
                fontsize=7, color='gray', style='italic', ha='right', va='top',
                transform=ax.transAxes)
        texte = "RAPPORT — Diagramme d'interaction biaxial N-My-Mz\n"
        texte += f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}\n"
        texte += "=" * 55 + "\n\nGÉOMÉTRIE\n" + "-" * 55 + "\n"
        texte += f"Type de section : {cfg['type_section']}\n"
        if cfg["type_section"] == "rectangulaire":
            texte += f"b x h = {cfg['b']*100:.0f} x {cfg['h']*100:.0f} cm\n"
            texte += f"Enrobage d_p = {cfg['d_p']*100:.1f} cm\n"
            texte += f"Ferraillage : {cfg['nx_barres']} barres/face horiz., {cfg['ny_barres']} barres/face vert., Ø{cfg['diam_mm']} mm\n"
        else:
            texte += f"Diamètre D = {cfg['D']*100:.0f} cm\n"
            for i, nappe in enumerate(cfg["nappes"]):
                texte += (f"  Nappe {i+1} : {nappe['n_barres']} x Ø{nappe['diam_mm']} mm, "
                          f"enrobage = {nappe['d_p']*100:.1f} cm\n")

        texte += f"\nAcier total : {np.sum(res['DA'])*1e4:.2f} cm²   "
        texte += f"(taux = {np.sum(res['DA'])/np.sum(res['dA'])*100:.2f} %)\n"

        texte += "\n" + "MATÉRIAUX\n" + "-" * 55 + "\n"
        texte += f"Béton : fck = {cfg['fck']:.0f} MPa   gamma_c = {cfg['gamma_c']}   fcd = {res['fcd']/1e6:.2f} MPa\n"
        texte += f"Acier : fyk = {cfg['fyk']:.0f} MPa   gamma_s = {cfg['gamma_s']}   fyd = {res['fyd']/1e6:.1f} MPa\n"
        texte += f"        Es = {cfg['Es']:.0f} GPa\n"

        texte += "\n" + "RÉSOLUTION NUMÉRIQUE\n" + "-" * 55 + "\n"
        texte += f"Grille béton : {cfg['nx_beton']} x {cfg['ny_beton']}   |   {cfg['n_pts']} pts/pivot   |   {cfg['n_theta']} angles\n"

        texte += "\n" + "=" * 55 + f"\nNIVEAU ÉTUDIÉ : N = {N_cible_kN:.0f} kN\n" + "=" * 55

        ax.text(0.03, 0.97, texte, fontsize=10, family='monospace', va='top', ha='left', transform=ax.transAxes)
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Page 2 : schéma de la section ----
        fig, ax = plt.subplots(figsize=(8.27, 8.27))
        if cfg["type_section"] == "rectangulaire":
            ax.add_patch(plt.Rectangle((-cfg['b']/2, -cfg['h']/2), cfg['b'], cfg['h'],
                                        facecolor='#212121', edgecolor='black', lw=1.5, zorder=1))
        else:
            ax.add_patch(plt.Circle((0, 0), cfg['D']/2, facecolor='#212121',
                                     edgecolor='black', lw=1.5, zorder=1))
        ax.scatter(res["Xa"], res["Ya"], s=80, color='#B0BEC5', edgecolor='black', zorder=3)
        ax.scatter(res["Xc"], res["Yc"], s=2, color='#555555', alpha=0.3, zorder=2)
        ax.set_aspect('equal')
        ax.set_title("Section et discrétisation", fontsize=13, fontweight='bold')
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Page 3 : diagramme My-Mz à N constant ----
        My_c, Mz_c = coupe_a_N_constant(N_cible_kN * 1e3, res["branches_N"], res["branches_My"],
                                         res["branches_Mz"], n_theta)
        fig, ax = plt.subplots(figsize=(8.27, 8.27))
        My_f = np.append(My_c, My_c[0]) / 1e3
        Mz_f = np.append(Mz_c, Mz_c[0]) / 1e3
        ax.plot(My_f, Mz_f, '-o', color='#2196F3', ms=3)
        ax.fill(My_f, Mz_f, color='#90CAF9', alpha=0.3)
        ax.axhline(0, color='black', lw=0.6, alpha=0.4)
        ax.axvline(0, color='black', lw=0.6, alpha=0.4)
        ax.set_xlabel('My (kN.m)'); ax.set_ylabel('Mz (kN.m)')
        ax.set_title(f"Diagramme d'interaction My-Mz à N = {N_cible_kN:.0f} kN", fontsize=13, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(alpha=0.3, ls='--')
        pdf.savefig(fig)
        plt.close(fig)

        # ---- Page 4 : tableau des points critiques ----
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis('off')
        ax.set_title("Points critiques du diagramme", fontsize=14, fontweight='bold', pad=30)
        tbl = ax.table(cellText=df_crit.round(2).values, colLabels=df_crit.columns,
                        loc='upper center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 2.0)
        pdf.savefig(fig)
        plt.close(fig)

    buffer.seek(0)
    return buffer


# ============================================================
# INTERFACE STREAMLIT
# ============================================================

st.set_page_config(page_title="Diagramme biaxial N-My-Mz (EC2)", layout="wide")

col_titre, col_credit = st.columns([5, 1.3])
with col_titre:
    st.title("Diagramme d'interaction biaxial N-My-Mz (Eurocode 2)")
    st.caption("Calcul par balayage de l'angle de l'axe neutre, discrétisation en grille de fibres.")
with col_credit:
    st.markdown(
        "<div style='text-align:right; font-size:0.75rem; color:gray; padding-top:1.6rem;'>"
        "Cet outil de calcul est<br>développé par <b>Charfi Kaies</b></div>",
        unsafe_allow_html=True,
    )

# Résolution numérique fixée au maximum pour garantir un calcul le plus précis
# possible (plus de sliders exposés à l'utilisateur : le temps de calcul reste
# raisonnable et on privilégie la robustesse du résultat).
NX_BETON_MAX = 100
NY_BETON_MAX = 100
N_PTS_MAX = 80
N_THETA_MAX = 96

with st.sidebar:
    st.header("1. Section")
    type_section = st.radio("Type de section", ["rectangulaire", "circulaire"], horizontal=True)

    if type_section == "rectangulaire":
        b = st.number_input("Largeur b (m)", 0.10, 3.0, 0.30, 0.01)
        h = st.number_input("Hauteur h (m)", 0.10, 3.0, 0.60, 0.01)
        D = 0.50  # non utilisé
    else:
        D = st.number_input("Diamètre D (m)", 0.10, 3.0, 0.50, 0.01)
        b = h = 0.30  # non utilisés

    st.header("2. Ferraillage")
    if type_section == "rectangulaire":
        d_p = st.number_input("Enrobage jusqu'au centre des barres d_p (m)", 0.01, 0.20, 0.05, 0.005)
        nx_barres = st.number_input("Barres par face horizontale (haut/bas)", 2, 20, 4)
        ny_barres = st.number_input("Barres par face verticale (coins inclus)", 2, 20, 3)
        diam_mm = st.number_input("Diamètre des barres (mm)", 6, 40, 20)
        nappes = []  # non utilisé
    else:
        d_p = 0.05  # non utilisé (chaque nappe a son propre enrobage)
        nx_barres = ny_barres = 4  # non utilisés
        diam_mm = 20  # non utilisé (chaque nappe a son propre diamètre)
        n_nappes = st.number_input("Nombre de nappes d'acier", 1, 4, 1)
        nappes = []
        for i in range(int(n_nappes)):
            with st.expander(f"Nappe {i+1}", expanded=(i == 0)):
                d_p_i = st.number_input(f"Enrobage nappe {i+1} (m)", 0.01, 0.20, 0.05, 0.005, key=f"dp_circ_{i}")
                n_barres_i = st.number_input(f"Nombre de barres — nappe {i+1}", 3, 40, 12, key=f"nb_circ_{i}")
                diam_i = st.number_input(f"Diamètre des barres (mm) — nappe {i+1}", 6, 40, 20, key=f"diam_circ_{i}")
                nappes.append(dict(n_barres=int(n_barres_i), diam_mm=diam_i, d_p=d_p_i))

    st.header("3. Matériaux")
    fck = st.number_input("fck béton (MPa)", 12, 90, 25)
    gamma_c = st.number_input("gamma_c", 1.0, 2.0, 1.5, 0.05)
    fyk = st.number_input("fyk acier (MPa)", 300, 600, 500)
    gamma_s = st.number_input("gamma_s", 1.0, 2.0, 1.15, 0.05)
    Es = st.number_input("Es acier (GPa)", 150, 220, 200)

    calculer_btn = st.button("Calculer / Recalculer la surface", type="primary", use_container_width=True)

cfg = dict(
    type_section=type_section, b=b, h=h, D=D, d_p=d_p,
    nx_barres=int(nx_barres), ny_barres=int(ny_barres), diam_mm=diam_mm, nappes=nappes,
    fck=fck, gamma_c=gamma_c, fyk=fyk, gamma_s=gamma_s, Es=Es,
    nx_beton=NX_BETON_MAX, ny_beton=NY_BETON_MAX, n_pts=N_PTS_MAX, n_theta=N_THETA_MAX,
)

if "res" not in st.session_state:
    st.session_state.res = None
    st.session_state.cfg = None

if calculer_btn or st.session_state.res is None:
    res = calculer_tout(cfg)
    st.session_state.res = res
    st.session_state.cfg = cfg

res = st.session_state.res
cfg_actif = st.session_state.cfg

if cfg_actif != cfg:
    st.info("Les paramètres ont changé — clique sur **Calculer / Recalculer la surface** pour les appliquer.")

col1, col2, col3 = st.columns(3)
col1.metric("Fibres béton", len(res["Xc"]))
col2.metric("Barres acier", len(res["Xa"]), f"{np.sum(res['DA'])*1e4:.1f} cm²")
col3.metric("Taux de ferraillage", f"{np.sum(res['DA'])/np.sum(res['dA'])*100:.2f} %")

tab_section, tab_3d, tab_coupe, tab_verif, tab_export = st.tabs(
    ["Section", "Surface 3D", "Coupe à N constant", "Vérification d'un point", "Export"]
)

with tab_section:
    st.pyplot(figure_section(cfg_actif, res), use_container_width=False)

with tab_3d:
    st.plotly_chart(figure_surface_3d(res), use_container_width=True)
    st.caption(
        f"Plage N : [{res['branches_N'].min()/1e3:.0f} ; {res['branches_N'].max()/1e3:.0f}] kN — "
        f"Plage My : [{res['branches_My'].min()/1e3:.1f} ; {res['branches_My'].max()/1e3:.1f}] kN.m — "
        f"Plage Mz : [{res['branches_Mz'].min()/1e3:.1f} ; {res['branches_Mz'].max()/1e3:.1f}] kN.m"
    )

with tab_coupe:
    n_min = float(res["branches_N"].min() / 1e3)
    n_max = float(res["branches_N"].max() / 1e3)
    N_cible = st.slider("N (kN)", min_value=n_min, max_value=n_max, value=0.0, step=(n_max - n_min) / 200)
    fig_coupe, My_c, Mz_c = figure_coupe(N_cible, res, cfg_actif["n_theta"])
    st.pyplot(fig_coupe, use_container_width=False)

with tab_verif:
    st.write("Vérifie si un point de sollicitation (N_Ed, My_Ed, Mz_Ed) est à l'intérieur de la surface de résistance.")
    vc1, vc2, vc3 = st.columns(3)
    N_Ed = vc1.number_input("N_Ed (kN)", value=800.0, step=10.0)
    My_Ed = vc2.number_input("My_Ed (kN.m)", value=60.0, step=1.0)
    Mz_Ed = vc3.number_input("Mz_Ed (kN.m)", value=40.0, step=1.0)

    if st.button("Vérifier ce point"):
        fig_v, est_dedans = figure_verification(N_Ed, My_Ed, Mz_Ed, res, cfg_actif["n_theta"])
        if fig_v is None:
            st.warning("N_Ed est hors de la plage résistante de la section (aucune coupe valide).")
        else:
            st.pyplot(fig_v, use_container_width=False)
            if est_dedans:
                st.success(f"Section VÉRIFIÉE : le point (N_Ed={N_Ed:.0f} kN, My_Ed={My_Ed:.1f}, "
                           f"Mz_Ed={Mz_Ed:.1f} kN.m) est à l'intérieur de la surface de résistance.")
            else:
                st.error(f"Section NON VÉRIFIÉE : le point (N_Ed={N_Ed:.0f} kN, My_Ed={My_Ed:.1f}, "
                         f"Mz_Ed={Mz_Ed:.1f} kN.m) est en dehors de la surface de résistance.")

with tab_export:
    st.write("Génère un tableau des points critiques et un rapport pour un niveau d'effort normal N donné.")
    n_min = float(res["branches_N"].min() / 1e3)
    n_max = float(res["branches_N"].max() / 1e3)
    N_export = st.number_input("N à exporter (kN)", min_value=n_min, max_value=n_max, value=min(max(800.0, n_min), n_max))

    df_crit = tableau_points_critiques(N_export, res["branches_N"], res["branches_My"], res["branches_Mz"], cfg_actif["n_theta"])
    df_contour = tableau_contour_complet(N_export, res["branches_N"], res["branches_My"], res["branches_Mz"], cfg_actif["n_theta"])
    if df_crit is None:
        st.warning("N hors de la plage résistante de la section.")
    else:
        st.write(f"**Tous les points du contour** ({len(df_contour)} points) :")
        st.dataframe(df_contour.round(3), use_container_width=True, hide_index=True)

        st.write("**Points critiques** (extrêmes et croisements d'axes) :")
        st.dataframe(df_crit.round(2), use_container_width=True, hide_index=True)

        ec1, ec2 = st.columns(2)
        excel_buffer = exporter_excel_bytes(N_export, res, cfg_actif["n_theta"])
        ec1.download_button(
            "Télécharger Excel (.xlsx)", data=excel_buffer,
            file_name=f"resultats_N{N_export:.0f}kN.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

        pdf_buffer = generer_rapport_pdf_bytes(cfg_actif, res, N_export, cfg_actif["n_theta"])
        ec2.download_button(
            "Télécharger le rapport (.pdf)", data=pdf_buffer,
            file_name=f"rapport_diagramme_N{N_export:.0f}kN.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
