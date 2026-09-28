"""Step 08: facts about the Safia network in Tashkent for the one-page report.

F1  Branches per 100k residents by district (official population, step 06b).
F2  Coverage gaps: share of residential cells whose centre is more than 1 km from a Safia.
F3  Network overlap: branches with another Safia within 500 m; competitors next to Safia.
Side stats: 24/7 share, site data-quality issues.
Writes figures to outputs/figures/08_*.png and the numbers to docs/findings.md.
"""
import json
import sys

import geopandas as gpd
import h3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CRS_METRIC, CRS_WGS84, DATA_INTERIM, DATA_PROCESSED, DATA_RAW, DOCS, OUTPUTS_FIGURES

COVER_M = 1000        # "within reach" of a branch
OVERLAP_M = 500       # two branches this close compete for the same customers
RESIDENTIAL_MAX_NONRES = 0.5
SAFIA_RED, GREY = "#c0392b", "#bdc3c7"


def pairwise_min(a: np.ndarray, b: np.ndarray, same: bool = False) -> np.ndarray:
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    if same:
        np.fill_diagonal(d, np.inf)
    return d.min(1)


def xy(df: pd.DataFrame) -> np.ndarray:
    g = gpd.GeoSeries(gpd.points_from_xy(df["lon"], df["lat"]), crs=CRS_WGS84).to_crs(CRS_METRIC)
    return np.c_[g.x, g.y]


def main() -> int:
    safia = pd.read_csv(DATA_INTERIM / "safia_tashkent.csv")
    ordinary = safia[safia["special_site"].isna()].copy()
    pop = pd.read_csv(DATA_PROCESSED / "district_population.csv").dropna(subset=["population"])
    cells = pd.read_parquet(DATA_PROCESSED / "hex_features.parquet")
    comp = pd.read_csv(DATA_INTERIM / "competitors_geo.csv")
    all_branches = pd.read_csv(DATA_INTERIM / "safia_branches.csv")
    lines = ["# Topilmalar (08-qadam)", "",
             "Manba: Safia sayti (2026-09-28), rasmiy tuman aholisi (2023-04-01), OSM, Cake Lab sayti. "
             "Faqat oddiy filiallar (Duty-free va Fabrika hisobga olinmagan). Yangi Toshkent tumani kiritilmagan.", ""]

    # F1: branches per 100k residents
    per = ordinary.groupby("district").size().rename("branches").to_frame().join(
        pop.set_index("district")[["population", "density_per_km2"]], how="right").fillna({"branches": 0})
    per["per_100k"] = per["branches"] / per["population"] * 1e5
    per["residents_per_branch"] = per["population"] / per["branches"].replace(0, np.nan)
    per = per.sort_values("per_100k")
    city_rate = per["branches"].sum() / per["population"].sum() * 1e5
    print("F1. Branches per 100k residents:")
    print(per.round(1).to_string())
    lo, hi = per.iloc[0], per.iloc[-1]
    print(f"   City: {city_rate:.1f} per 100k | lowest {lo.name} {lo.per_100k:.1f}, highest {hi.name} {hi.per_100k:.1f} "
          f"(x{hi.per_100k / lo.per_100k:.1f})")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = [SAFIA_RED if r < city_rate * 0.75 else GREY for r in per["per_100k"]]
    ax.barh(per.index, per["per_100k"], color=colors)
    ax.axvline(city_rate, color="black", linestyle="--", linewidth=1)
    ax.text(city_rate, len(per) - 0.4, f" city average {city_rate:.1f}", va="bottom", fontsize=9)
    for i, (b, p, r) in enumerate(zip(per["branches"], per["population"], per["per_100k"])):
        ax.text(r + 0.08, i, f"{int(b)} branches / {p / 1000:.0f}k people", va="center", fontsize=8, color="#333")
    ax.set_xlim(0, per["per_100k"].max() * 1.45)
    ax.set_xlabel("Safia branches per 100,000 residents")
    ax.set_title("F1. Safia coverage per resident differs strongly by district")
    fig.savefig(OUTPUTS_FIGURES / "08_f1_branches_per_100k.png", dpi=150, bbox_inches="tight")

    # F2: residential cells far from Safia
    res = cells[cells["nonres_share"] < RESIDENTIAL_MAX_NONRES].copy()
    res["far"] = res["dist_safia_m"] > COVER_M
    gap = res.groupby("district").agg(cells=("h3", "size"), far=("far", "sum"))
    gap["far_share"] = gap["far"] / gap["cells"]
    gap = gap.join(pop.set_index("district")["density_per_km2"])
    cell_km2 = res["h3"].map(lambda c: h3.cell_area(c, "km^2")).mean()  # ~0.88 at Tashkent's latitude
    gap["residents_far_est"] = gap["far"] * cell_km2 * gap["density_per_km2"]
    gap = gap.sort_values("far_share", ascending=False)
    print(f"\nF2. Residential cells (nonres_share < {RESIDENTIAL_MAX_NONRES}) with centre > {COVER_M} m from Safia:")
    print(gap.round(2).to_string())
    total_far = gap["far"].sum() / gap["cells"].sum()
    print(f"   City: {total_far:.0%} of residential cells; rough residents there "
          f"{gap['residents_far_est'].sum():,.0f} (district density x cell area, an approximation)")

    fig, ax = plt.subplots(figsize=(8, 8))
    grid = gpd.GeoDataFrame(res, geometry=gpd.points_from_xy(res["lon"], res["lat"]), crs=CRS_WGS84)
    districts = gpd.read_file(DATA_RAW / "tashkent_districts.geojson")
    districts = districts[districts["district"] != "Yangi Toshkent"]
    districts.boundary.plot(ax=ax, color="#555", linewidth=0.6)
    grid[~grid["far"]].plot(ax=ax, color="#d5e8d4", markersize=28, marker="h")
    grid[grid["far"]].plot(ax=ax, color="#f39c12", markersize=28, marker="h", label=f"> {COVER_M} m from Safia")
    ax.scatter(ordinary["lon"], ordinary["lat"], s=8, color=SAFIA_RED, label="Safia branch", zorder=3)
    for r in districts.itertuples():
        pt = r.geometry.representative_point()
        ax.annotate(r.district, (pt.x, pt.y), fontsize=7, ha="center", color="#333")
    ax.legend(loc="lower left")
    ax.set_title(f"F2. Residential areas more than {COVER_M / 1000:.0f} km from a Safia branch")
    ax.set_axis_off()
    fig.savefig(OUTPUTS_FIGURES / "08_f2_coverage_gaps.png", dpi=150, bbox_inches="tight")

    # F3: overlap between branches, and with competitors
    sxy, cxy = xy(ordinary), xy(comp)
    ordinary["nearest_safia_m"] = pairwise_min(sxy, sxy, same=True)
    close = (ordinary["nearest_safia_m"] <= OVERLAP_M).sum()
    comp["nearest_safia_m"] = pairwise_min(cxy, sxy)
    comp_close = comp.groupby("brand")["nearest_safia_m"].apply(lambda s: (s <= OVERLAP_M).mean())
    ordinary["nearest_comp_m"] = pairwise_min(sxy, cxy)
    print(f"\nF3. Branches with another Safia within {OVERLAP_M} m: {close}/{len(ordinary)} ({close / len(ordinary):.0%}); "
          f"median distance to nearest Safia {ordinary['nearest_safia_m'].median():.0f} m")
    by_d = ordinary.assign(close=ordinary["nearest_safia_m"] <= OVERLAP_M).groupby("district")["close"].agg(["sum", "size"])
    print(by_d.sort_values("sum", ascending=False).to_string())
    print(f"   Competitor branches with a Safia within {OVERLAP_M} m: {dict(comp_close.round(2))} "
          f"(all: {(comp['nearest_safia_m'] <= OVERLAP_M).mean():.0%})")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.arange(0, 3001, 250)
    ax.hist(ordinary["nearest_safia_m"].clip(upper=3000), bins=bins, color=SAFIA_RED, alpha=0.85)
    ax.axvline(OVERLAP_M, color="black", linestyle="--", linewidth=1)
    ax.text(OVERLAP_M, ax.get_ylim()[1] * 0.92, f"  {close} of {len(ordinary)} branches have another Safia "
            f"within {OVERLAP_M} m", fontsize=9)
    share_comp = (comp["nearest_safia_m"] <= OVERLAP_M).mean()
    ax.text(0.98, 0.6, f"{share_comp:.0%} of competitor branches\n(Bon!, Cake Lab, Breadly)\nhave a Safia within {OVERLAP_M} m",
            transform=ax.transAxes, ha="right", fontsize=10, bbox={"facecolor": "white", "edgecolor": "#999"})
    ax.set_xlabel("Distance from a Safia branch to the nearest other Safia, m (3000+ grouped)")
    ax.set_ylabel("Branches")
    ax.set_title(f"F3. Competition is concentrated where Safia already is (median gap between branches {ordinary['nearest_safia_m'].median():.0f} m)")
    fig.savefig(OUTPUTS_FIGURES / "08_f3_nearest_branch.png", dpi=150, bbox_inches="tight")

    # Side stats
    tk_24 = ordinary["is_24h"].mean()
    late = all_branches[all_branches["hours"].str.extract(r"-(\d{2}):", expand=False).astype(int) > 24]
    print(f"\nSide: 24/7 branches in Tashkent {ordinary['is_24h'].sum()}/{len(ordinary)} ({tk_24:.0%}); "
          f"S express {ordinary['is_s_express'].sum()}; opening hours after 24:00 on the site: "
          f"{late[['name', 'hours']].values.tolist()}")

    # findings.md
    lines += [
        f"## F1. 100 ming aholiga to'g'ri keladigan filiallar soni tumanlar bo'yicha {hi.per_100k / lo.per_100k:.1f} baravar farq qiladi",
        f"- Shahar bo'yicha: {city_rate:.1f} filial / 100 ming aholi ({int(per['branches'].sum())} filial, {per['population'].sum():,.0f} aholi).",
        f"- Eng past: **{lo.name}** — {lo.per_100k:.1f} ({int(lo.branches)} filial, {lo.population:,.0f} aholi, "
        f"bir filialga {lo.residents_per_branch:,.0f} kishi).",
        f"- Eng yuqori: **{hi.name}** — {hi.per_100k:.1f} ({int(hi.branches)} filial, {hi.population:,.0f} aholi).",
        "- Jadval: " + "; ".join(f"{d} {r.per_100k:.1f}" for d, r in per.iterrows()) + ".",
        "- Rasm: `outputs/figures/08_f1_branches_per_100k.png`.",
        "- Cheklov: aholi 2023 yil apreliga oid; markaziy tumanlarga kunduzi ishlash va o'qish uchun ko'p odam keladi, shuning uchun ular yuqori bo'lishi tabiiy.", "",
        f"## F2. Turar-joy hududlarining {total_far:.0%} i eng yaqin Safia'dan {COVER_M / 1000:.0f} km dan uzoqda",
        f"- Turar-joy katagi: noturar hudud ulushi < {RESIDENTIAL_MAX_NONRES}. Masofa — katak markazidan to'g'ri chiziq bo'yicha.",
        "- Tumanlar bo'yicha uzoq kataklar ulushi: " + "; ".join(f"{d} {r.far_share:.0%}" for d, r in gap.iterrows()) + ".",
        f"- U yerlarda taxminan {gap['residents_far_est'].sum():,.0f} kishi yashaydi (tuman zichligi × katak maydoni; qo'pol baho).",
        "- Rasm: `outputs/figures/08_f2_coverage_gaps.png`.", "",
        f"## F3. Raqobatchilar filiallarining {(comp['nearest_safia_m'] <= OVERLAP_M).mean():.0%} i Safia'dan {OVERLAP_M} m ichida: raqobat markazda, janub bo'sh",
        f"- {close} ta Safia filialidan ({close / len(ordinary):.0%}) {OVERLAP_M} m ichida boshqa Safia bor.",
        f"- Eng yaqin boshqa Safia'gacha masofa medianasi: {ordinary['nearest_safia_m'].median():.0f} m.",
        "- Tumanlar bo'yicha (yaqin juftli / jami): " + "; ".join(f"{d} {r['sum']}/{r['size']}" for d, r in by_d.sort_values('sum', ascending=False).iterrows()) + ".",
        f"- Raqobatchilarning {(comp['nearest_safia_m'] <= OVERLAP_M).mean():.0%} ida {OVERLAP_M} m ichida Safia bor "
        f"({', '.join(f'{b} {v:.0%}' for b, v in comp_close.items())}).",
        "- Rasm: `outputs/figures/08_f3_nearest_branch.png`.",
        "- Talqin: bu kannibalizatsiya bo'lishi mumkin yoki yuqori talabli joylarda ataylab zich joylashtirish. Ichki savdo ma'lumotisiz ajratib bo'lmaydi — suhbatda so'raladigan savol.", "",
        "## Qo'shimcha",
        f"- Toshkentdagi 24/7 filiallar: {ordinary['is_24h'].sum()} / {len(ordinary)} ({tk_24:.0%}); S express: {ordinary['is_s_express'].sum()}.",
        "- Saytdagi ma'lumot xatolari: " + ", ".join(f"{n} {h}" for n, h in late[['name', 'hours']].values) +
        "; 10 ta filialda ichki UUID yo'q, 6 tasida takrorlangan (02-qadam).",
        "- Global aholi xaritalari (WorldPop, GHSL) Toshkent janubiy tumanlari aholisini 4–7 baravar oshirib ko'rsatadi (06b-qadam, `outputs/figures/06b_population.png`).",
    ]
    (DOCS / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Machine-readable copy for the report (step 12): one source for every number
    facts = {
        "branches": int(per["branches"].sum()), "population": int(per["population"].sum()),
        "per_100k_city": round(city_rate, 1),
        "per_100k": {d: round(v, 1) for d, v in per["per_100k"].items()},
        "branches_by_district": {d: int(v) for d, v in per["branches"].items()},
        "population_by_district": {d: int(v) for d, v in per["population"].items()},
        "per_100k_ratio_max_min": round(hi.per_100k / lo.per_100k, 1),
        "far_share_city": round(total_far, 3), "far_share": {d: round(v, 3) for d, v in gap["far_share"].items()},
        "residents_far_est": int(round(gap["residents_far_est"].sum(), -4)),
        "cover_m": COVER_M, "overlap_m": OVERLAP_M,
        "safia_close_pairs": int(close), "safia_nearest_median_m": int(ordinary["nearest_safia_m"].median()),
        "competitors_near_safia_share": round((comp["nearest_safia_m"] <= OVERLAP_M).mean(), 3),
        "competitors_near_safia_by_brand": {b: round(v, 3) for b, v in comp_close.items()},
        "branches_24h": int(ordinary["is_24h"].sum()),
    }
    (DATA_PROCESSED / "facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSaved docs/findings.md and outputs/figures/08_f1..f3 png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
