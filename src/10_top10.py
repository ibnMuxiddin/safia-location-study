"""Step 10: top-10 candidate areas = cells that look like Safia locations but have no branch nearby.

Score: spatial out-of-fold score of the chosen model (step 09), so no cell is scored by a model
that saw it. Rules: nearest ordinary Safia >= 1 km; candidates >= 1.5 km apart (greedy by score).
Reasons: SHAP values of the chosen model summed per concept (cell + neighbour features together),
because in-cell and neighbour coefficients have opposite signs and read badly on their own.
"""
import sys

import folium
import geopandas as gpd
import h3
import numpy as np
import pandas as pd

from config import CRS_METRIC, CRS_WGS84, DATA_INTERIM, DATA_PROCESSED, DATA_RAW, OUTPUTS_MAPS
from map_utils import add_basemaps, safe_text

CHECK_MAP = OUTPUTS_MAPS / "10_candidates_check.html"
CHECKS = DATA_INTERIM / "candidate_checks.csv"  # user's satellite checks: verdict ok / rejected


def hex_latlon(cell: str) -> list[tuple[float, float]]:
    return [tuple(p) for p in h3.cell_to_boundary(cell)]


def check_map(top: pd.DataFrame, coverage: pd.DataFrame | None = None) -> None:
    """Satellite map for the manual check: candidate cell (the area to check, ~450 m from its centre),
    the 7-cell reason area (~1.5 km), and ordinary Safia branches.
    At Tashkent's latitude a res-8 cell is ~0.88 km2: ~500 m from centre to edge, ~580 m to a corner."""
    m = folium.Map(location=[top["lat"].mean(), top["lon"].mean()], zoom_start=12, tiles=None)
    add_basemaps(m, default="satellite")
    reason_layer = folium.FeatureGroup(name="Reason area: cell + 6 neighbours (~1.5 km)")
    layers = [reason_layer]
    for frame, title, line, badge in [(top, "Top-10 'similarity' (check ~500 m around the centre)", "#00e5ff", "#c0392b"),
                                      (coverage, "Coverage list C1-C5 (under-served south)", "#ff9f1a", "#8e44ad")]:
        if frame is None:
            continue
        cell_layer = folium.FeatureGroup(name=title)
        for r in frame.itertuples():
            for n in h3.grid_disk(r.h3, 1):
                folium.Polygon(hex_latlon(n), color="#f1c40f", weight=1, fill=False, dash_array="4").add_to(reason_layer)
            folium.Polygon(hex_latlon(r.h3), color=line, weight=3, fill=True, fill_opacity=0.08,
                           tooltip=safe_text(f"{r.rank} {r.district}, score {r.score:.2f}")).add_to(cell_layer)
            folium.Marker([r.lat, r.lon], icon=folium.DivIcon(
                html=f'<div style="font:bold 14px sans-serif;color:#fff;background:{badge};border-radius:10px;'
                     f'padding:1px 6px;transform:translate(-50%,-50%);display:inline-block">{r.rank}</div>'),
                popup=folium.Popup(safe_text(f"{r.rank} {r.district} | centre {r.lat:.5f}, {r.lon:.5f} | "
                                             f"{r.reason}"), max_width=300)).add_to(cell_layer)
        layers.append(cell_layer)
    safia = pd.read_csv(DATA_INTERIM / "safia_tashkent.csv")
    s_layer = folium.FeatureGroup(name="Safia branches")
    for r in safia[safia["special_site"].isna()].itertuples():
        folium.CircleMarker([r.lat, r.lon], radius=4, color="#e74c3c", fill=True, fill_opacity=1,
                            tooltip=safe_text(f"Safia: {r.name}")).add_to(s_layer)
    for layer in layers + [s_layer]:
        layer.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(CHECK_MAP)

MIN_DIST_SAFIA_M = 1000
MIN_SPACING_M = 1500
TOP_N = 10
# "Coverage" list (agreed with the user 2026-09-28): under-served districts from step 08
COVERAGE_DISTRICTS = ["Yangihayot", "Sergeli", "Bektemir", "Yashnobod"]
COVERAGE_N = 5
COVERAGE_MAX_NONRES = 0.5
# concept -> (feature columns, count column shown in the text, Uzbek label, Russian label)
CONCEPTS = {
    "cafe": (["n_cafe", "n_cafe_k1"], "n_cafe_k1", "kafe", "кафе"),
    "school": (["n_school", "n_school_k1"], "n_school_k1", "maktab", "школ"),
    "bus_stop": (["n_bus_stop", "n_bus_stop_k1"], "n_bus_stop_k1", "avtobus bekati", "остановок"),
    "supermarket": (["n_supermarket", "n_supermarket_k1"], "n_supermarket_k1", "supermarket", "супермаркетов"),
    "university": (["n_university", "n_university_k1"], "n_university_k1", "universitet/kollej", "вузов/колледжей"),
    "hospital": (["n_hospital", "n_hospital_k1"], "n_hospital_k1", "shifoxona/klinika", "больниц/клиник"),
    "metro": (["n_metro", "n_metro_k1"], "n_metro_k1", "metro bekati", "станций метро"),
    "mall": (["n_mall", "n_mall_k1"], "n_mall_k1", "savdo markazi", "ТЦ"),
    "office": (["n_office", "n_office_k1"], "n_office_k1", "ofis", "офисов"),
    "apartments": (["n_apartments", "n_apartments_k1", "n_apartment_levels", "n_apartment_levels_k1"],
                   "n_apartments_k1", "ko'p qavatli uy", "многоэтажных домов"),
    "density": (["district_density"], None, "zich tuman", "плотный район"),
    "centre": (["dist_centre_km"], None, "markazga yaqin", "близко к центру"),
    "landuse": (["nonres_share"], None, "sanoat zonasi yo'q", "нет промзоны"),
}


def concept_shap(shap_row: pd.Series) -> pd.Series:
    return pd.Series({k: shap_row[[c for c in cols if c in shap_row.index]].sum() for k, (cols, *_) in CONCEPTS.items()})


def reason(row: pd.Series, contrib: pd.Series, lang: str) -> str:
    parts = []
    for concept in contrib.sort_values(ascending=False).index:
        if contrib[concept] <= 0 or len(parts) == 3:
            break
        _, count_col, uz, ru = CONCEPTS[concept]
        if count_col is not None and row[count_col] == 0:
            continue  # "0 cafes" can raise the score (negative neighbour coefficient) but is not a reason
        if count_col is None:
            text = uz if lang == "uz" else ru
            if concept == "density":
                text += f" ({row['district_density']:,.0f} {'kishi/km²' if lang == 'uz' else 'чел./км²'})"
        else:
            text = f"{int(row[count_col])} {uz if lang == 'uz' else ru}"
        parts.append(text)
    prefix = "Atrofda (~1.5 km): " if lang == "uz" else "В радиусе ~1.5 км: "
    return prefix + ", ".join(parts)


def main() -> int:
    feats = pd.read_parquet(DATA_PROCESSED / "hex_features.parquet")
    scores = pd.read_parquet(DATA_PROCESSED / "oof_scores.parquet")
    shap_vals = pd.read_parquet(DATA_PROCESSED / "shap_values.parquet").set_index("h3")
    df = feats.merge(scores.drop(columns="has_safia"), on="h3")

    cand = df[(df["has_safia"] == 0) & (df["dist_safia_m"] >= MIN_DIST_SAFIA_M)].sort_values("score", ascending=False)
    print(f"Cells without Safia: {(df['has_safia'] == 0).sum()} | of these >= {MIN_DIST_SAFIA_M} m from Safia: {len(cand)}")
    # Cells the user checked on satellite imagery and rejected (e.g. cropland) are not candidates
    checks = pd.read_csv(CHECKS) if CHECKS.exists() else pd.DataFrame(columns=["h3", "verdict", "note"])
    rejected = checks.loc[checks["verdict"] == "rejected", "h3"]
    cand = cand[~cand["h3"].isin(rejected)]
    print(f"Manual checks: {checks['verdict'].value_counts().to_dict()} | rejected cells removed: {len(rejected)}")
    pois = gpd.read_file(DATA_RAW / "osm_pois.geojson")
    named = pois[pois["category"].isin(["metro", "mall", "university"]) & pois["name"].notna()].to_crs(CRS_METRIC)

    top = select(cand, TOP_N, blocked=None)
    out = describe(top, named, shap_vals)
    out.to_csv(DATA_PROCESSED / "top10.csv", index=False, encoding="utf-8")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    print(f"\nTop {len(out)} (score = spatial out-of-fold probability):")
    print(out[["rank", "district", "score", "dist_safia_m", "landmark", "landmark_dist_m"]].to_string(index=False))
    print()
    for r in out.itertuples():
        print(f"{r.rank:>2}. {r.reason}")
    print("\nBy district:", out["district"].value_counts().to_dict())
    print(f"Score of chosen candidates: {out['score'].min():.2f}-{out['score'].max():.2f} | "
          f"median score of cells that have Safia: {df.loc[df['has_safia'] == 1, 'score'].median():.2f}")

    # Robustness: would LightGBM (step 09, same features) pick the same areas?
    alt = df.set_index("h3")["oof_lgbm__no_competitor"]
    alt_rank = alt[cand["h3"]].rank(ascending=False)
    in_top_alt = [h for h in out["h3"] if alt_rank[h] <= 30]
    print(f"Robustness: {len(in_top_alt)}/{len(out)} of the top-10 are also in LightGBM's top-30 eligible cells")

    # Neighbours of candidates within the study grid (a sanity check that candidates are not edge artefacts)
    grid_cells = set(df["h3"])
    edge = [int(sum(n in grid_cells for n in h3.grid_disk(h, 1)) - 1) for h in out["h3"]]
    print(f"Neighbours inside the grid per candidate (6 = not on the edge): {edge}")

    # Coverage list: best cells inside the under-served districts (step 08, F1/F2). The model learns
    # where Safia already is, so these scores are low by construction; they need internal sales data.
    cov_cand = cand[cand["district"].isin(COVERAGE_DISTRICTS) & (cand["nonres_share"] < COVERAGE_MAX_NONRES)
                    & ~cand["h3"].isin(out["h3"])]
    cov = describe(select(cov_cand, COVERAGE_N, blocked=out), named, shap_vals)
    cov["rank"] = [f"C{i}" for i in range(1, len(cov) + 1)]
    cov.to_csv(DATA_PROCESSED / "coverage5.csv", index=False, encoding="utf-8")
    print(f"\nCoverage list ({', '.join(COVERAGE_DISTRICTS)}; nonres_share < {COVERAGE_MAX_NONRES}; "
          f">= {MIN_SPACING_M} m from each other and from the top-10): eligible cells {len(cov_cand)}")
    print(cov[["rank", "district", "score", "dist_safia_m", "landmark", "landmark_dist_m"]].to_string(index=False))
    for r in cov.itertuples():
        print(f"{r.rank:>3}. {r.reason}")
    pct = [(df["score"] < s).mean() for s in cov["score"]]
    print(f"Score percentile among all cells: {[f'{p:.0%}' for p in pct]}")

    check_map(out, cov)
    print(f"\nSaved {DATA_PROCESSED.name}/top10.csv, coverage5.csv and {CHECK_MAP.relative_to(CHECK_MAP.parents[2])}")
    return 0


def select(cand: pd.DataFrame, n: int, blocked: pd.DataFrame | None) -> pd.DataFrame:
    """Greedy by score: keep a cell if it is >= MIN_SPACING_M from every chosen and blocked cell."""
    def metric(frame):
        return gpd.GeoSeries(gpd.points_from_xy(frame["lon"], frame["lat"]), crs=CRS_WGS84,
                             index=frame.index).to_crs(CRS_METRIC)
    g = metric(cand)
    fixed = list(metric(blocked)) if blocked is not None else []
    chosen = []
    for idx in cand.index:
        pt = g[idx]
        if all(pt.distance(q) >= MIN_SPACING_M for q in fixed + [g[j] for j in chosen]):
            chosen.append(idx)
        if len(chosen) == n:
            break
    sel = cand.loc[chosen].copy()
    sel.insert(0, "rank", range(1, len(sel) + 1))
    return sel


def describe(sel: pd.DataFrame, named: gpd.GeoDataFrame, shap_vals: pd.DataFrame) -> pd.DataFrame:
    g = gpd.GeoSeries(gpd.points_from_xy(sel["lon"], sel["lat"]), crs=CRS_WGS84, index=sel.index).to_crs(CRS_METRIC)
    lm, lm_dist = [], []
    for idx in sel.index:  # landmark: nearest named metro / mall / university
        d = named.distance(g[idx])
        j = d.idxmin()
        lm.append(f"{named.at[j, 'name']} ({named.at[j, 'category']})")
        lm_dist.append(round(d[j]))
    sel["landmark"], sel["landmark_dist_m"] = lm, lm_dist
    contribs = [concept_shap(shap_vals.loc[h]) for h in sel["h3"]]
    sel["reason"] = [reason(r, c, "uz") for (_, r), c in zip(sel.iterrows(), contribs)]
    sel["reason_ru"] = [reason(r, c, "ru") for (_, r), c in zip(sel.iterrows(), contribs)]
    # pt= puts a pin exactly on the cell centre; z=16 shows roughly the cell and its neighbours
    sel["yandex"] = [f"https://yandex.uz/maps/?pt={lon},{lat}&z=16&l=sat,skl" for lat, lon in zip(sel["lat"], sel["lon"])]
    cols = ["rank", "h3", "lat", "lon", "score", "district", "dist_safia_m", "landmark", "landmark_dist_m",
            "reason", "reason_ru", "yandex"]
    return sel[cols].round({"score": 3, "dist_safia_m": 0, "lat": 6, "lon": 6})


if __name__ == "__main__":
    sys.exit(main())
