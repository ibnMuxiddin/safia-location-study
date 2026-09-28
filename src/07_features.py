"""Step 07: H3 grid over the study area and one feature row per cell.

Study area: the 12 established districts of Tashkent. Yangi Toshkent is excluded (no Safia
branches, no official population, mostly construction; decided with the user 2026-09-28).
Target: at least one ordinary Safia branch in the cell. Special sites (airport duty free,
factory store) are excluded from the target.
Leakage: `safia_count` and `dist_safia_m` describe the target. They are stored for step 10
(choosing candidates) and must never be used as model features.
"""
import sys

import geopandas as gpd
import h3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, mapping

from config import (CRS_METRIC, CRS_WGS84, DATA_INTERIM, DATA_PROCESSED, DATA_RAW, H3_CV_RES,
                    H3_RES, OUTPUTS_FIGURES)

EXCLUDED_DISTRICTS = ["Yangi Toshkent"]
CITY_CENTRE = (41.3111, 69.2797)  # Amir Temur square (lat, lon)
COMPETITOR_RADIUS_M = 1000
BRAND_PATTERN = r"safia|сафия|bon!|^bon$|cake ?lab|кейк ?лаб|breadly|брэдли"
POI_CATEGORIES = ["university", "school", "metro", "bus_stop", "office", "mall", "supermarket",
                  "hospital", "apartments", "cafe"]
OUT = DATA_PROCESSED / "hex_features.parquet"
DICT = DATA_PROCESSED / "feature_dictionary.md"
FIG = OUTPUTS_FIGURES / "07_grid_check.png"


def cell_polygon(c: str) -> Polygon:
    return Polygon([(lon, lat) for lat, lon in h3.cell_to_boundary(c)])


def points(df: pd.DataFrame, lat="lat", lon="lon") -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon], df[lat]), crs=CRS_WGS84)


def main() -> int:
    districts = gpd.read_file(DATA_RAW / "tashkent_districts.geojson")
    study = districts[~districts["district"].isin(EXCLUDED_DISTRICTS)]
    area = study.union_all()

    # Grid: every res-8 cell whose centre lies in the study area, plus any cell that holds an
    # ordinary branch (a branch near the boundary can sit in a cell whose centre is outside)
    safia = pd.read_csv(DATA_INTERIM / "safia_tashkent.csv")
    ordinary = safia[safia["special_site"].isna() & ~safia["district"].isin(EXCLUDED_DISTRICTS)]
    ordinary_h3 = pd.Series([h3.latlng_to_cell(a, b, H3_RES) for a, b in zip(ordinary["lat"], ordinary["lon"])])
    base = set(h3.geo_to_cells(mapping(area), H3_RES))
    added = set(ordinary_h3) - base
    cells = sorted(base | added)
    grid = gpd.GeoDataFrame({"h3": cells}, geometry=[cell_polygon(c) for c in cells], crs=CRS_WGS84)
    centres = [h3.cell_to_latlng(c) for c in cells]
    grid["lat"] = [c[0] for c in centres]
    grid["lon"] = [c[1] for c in centres]
    grid["cv_block"] = [h3.cell_to_parent(c, H3_CV_RES) for c in cells]
    centre_pts = points(grid[["h3", "lat", "lon"]])
    grid["district"] = gpd.sjoin(centre_pts, study[["district", "geometry"]], how="left",
                                 predicate="within")["district"].values
    no_district = grid["district"].isna()
    if no_district.any():  # centre just outside the study area: take the nearest district
        nearest = gpd.sjoin_nearest(centre_pts[no_district].to_crs(CRS_METRIC),
                                    study[["district", "geometry"]].to_crs(CRS_METRIC))
        grid.loc[no_district, "district"] = nearest["district"].groupby(level=0).first()
    print(f"Study area: {study.to_crs(CRS_METRIC).area.sum() / 1e6:.1f} km2 "
          f"({len(study)} districts, excluded: {EXCLUDED_DISTRICTS})")
    print(f"H3 res {H3_RES} cells: {len(grid)} ({len(base)} with centre inside + {len(added)} added because "
          f"they hold a branch) | cells whose district came from the nearest polygon: {no_district.sum()}")
    print(f"CV blocks (res {H3_CV_RES}): {grid['cv_block'].nunique()}, "
          f"cells per block: min {grid['cv_block'].value_counts().min()}, max {grid['cv_block'].value_counts().max()}")

    # POI counts: in the cell, and in the cell plus its 6 neighbours (k=1 disk)
    pois = gpd.read_file(DATA_RAW / "osm_pois.geojson")
    # Safia and competitor cafes are in OSM's cafe category: counting them would leak the target
    # (Safia) or duplicate the competitor feature. Found in step 09.
    brand = (pois["category"] == "cafe") & pois["name"].fillna("").str.contains(BRAND_PATTERN, case=False, regex=True)
    print(f"Brand cafes removed from POI counts: {brand.sum()} "
          f"(Safia: {pois.loc[brand, 'name'].str.contains('safia|сафия', case=False).sum()})")
    pois = pois[~brand]
    pois["h3"] = [h3.latlng_to_cell(p.y, p.x, H3_RES) for p in pois.geometry]
    counts = pois.pivot_table(index="h3", columns="category", values="osm_id", aggfunc="size", fill_value=0)
    counts = counts.reindex(columns=POI_CATEGORIES, fill_value=0)
    apt_levels = pois[pois["category"] == "apartments"].groupby("h3")["levels"].sum()
    in_cell = counts.reindex(grid["h3"]).fillna(0)
    in_cell["apartment_levels"] = apt_levels.reindex(grid["h3"]).fillna(0).values
    disks = [h3.grid_disk(c, 1) for c in grid["h3"]]
    all_counts = counts.assign(apartment_levels=apt_levels).fillna(0)
    ring = np.vstack([all_counts.reindex(d).fillna(0).sum().values for d in disks])
    for i, col in enumerate(all_counts.columns):
        grid[f"n_{col}"] = in_cell[col].values
        grid[f"n_{col}_k1"] = ring[:, i]

    # Official district density (step 06b)
    pop = pd.read_csv(DATA_PROCESSED / "district_population.csv")
    grid["district_density"] = grid["district"].map(pop.set_index("district")["density_per_km2"])

    # Metric geometry for distances and shares
    grid_m = grid.to_crs(CRS_METRIC)
    centres_m = centre_pts.to_crs(CRS_METRIC).geometry
    ctr = gpd.GeoSeries([Point(CITY_CENTRE[1], CITY_CENTRE[0])], crs=CRS_WGS84).to_crs(CRS_METRIC).iloc[0]
    grid["dist_centre_km"] = centres_m.distance(ctr).values / 1000

    # Share of the cell covered by non-residential land use (step 06b)
    nonres = gpd.read_file(DATA_RAW / "osm_nonresidential.geojson").to_crs(CRS_METRIC)
    nonres_union = nonres.union_all()
    grid["nonres_share"] = (grid_m.geometry.intersection(nonres_union).area / grid_m.area).clip(0, 1).values

    # Competitors within 1 km of the cell centre
    comp = points(pd.read_csv(DATA_INTERIM / "competitors_geo.csv")).to_crs(CRS_METRIC)
    cxy = np.c_[comp.geometry.x, comp.geometry.y]
    gxy = np.c_[centres_m.x, centres_m.y]
    dmat = np.sqrt(((gxy[:, None, :] - cxy[None, :, :]) ** 2).sum(-1))
    grid["competitors_1km"] = (dmat <= COMPETITOR_RADIUS_M).sum(1)
    grid["has_competitor_1km"] = (grid["competitors_1km"] > 0).astype(int)

    # Target and leakage columns
    grid["safia_count"] = grid["h3"].map(ordinary_h3.value_counts()).fillna(0).astype(int)
    grid["has_safia"] = (grid["safia_count"] > 0).astype(int)
    s_m = points(ordinary).to_crs(CRS_METRIC)
    sxy = np.c_[s_m.geometry.x, s_m.geometry.y]
    grid["dist_safia_m"] = np.sqrt(((gxy[:, None, :] - sxy[None, :, :]) ** 2).sum(-1)).min(1)
    outside = (~ordinary_h3.isin(grid["h3"])).sum()

    out = pd.DataFrame(grid.drop(columns="geometry"))
    out.to_parquet(OUT, index=False)
    print(f"Saved {OUT.relative_to(OUT.parents[2])}: {out.shape[0]} rows x {out.shape[1]} columns")

    print(f"\nOrdinary Safia branches: {len(ordinary)} (special sites excluded: "
          f"{safia['special_site'].notna().sum()}) | branches whose cell is outside the grid: {outside}")
    print(f"Cells with Safia (target=1): {out['has_safia'].sum()} ({out['has_safia'].mean():.1%}) | "
          f"cells with 2+ branches: {(out['safia_count'] >= 2).sum()}")
    print(f"Missing values: {int(out.isna().sum().sum())}")
    feats = [c for c in out.columns if c.startswith("n_") or c in
             ["district_density", "dist_centre_km", "nonres_share", "competitors_1km", "has_competitor_1km"]]
    desc = out[feats].describe().T[["mean", "50%", "max"]]
    desc["share_nonzero"] = (out[feats] > 0).mean()
    print("\nFeature summary:")
    print(desc.round(2).to_string())
    print("\nMean of features, cells with Safia vs without:")
    cmp_ = out.groupby("has_safia")[feats].mean().T
    cmp_.columns = ["no_safia", "safia"]
    cmp_["ratio"] = cmp_["safia"] / cmp_["no_safia"].replace(0, np.nan)
    print(cmp_.round(2).sort_values("ratio", ascending=False).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    panels = [("has_safia", "Target: cell has a Safia branch", "Reds"),
              ("n_apartment_levels_k1", "Apartment floors, cell + neighbours (OSM)", "viridis"),
              ("nonres_share", "Non-residential share of the cell", "Greys")]
    for ax, (col, title, cmap) in zip(axes, panels):
        grid.assign(v=out[col].values).plot(column="v", ax=ax, cmap=cmap, legend=True, edgecolor="none",
                                            legend_kwds={"shrink": 0.6})
        study.boundary.plot(ax=ax, color="#333", linewidth=0.5)
        ax.set_title(title)
        ax.set_axis_off()
    fig.savefig(FIG, dpi=110, bbox_inches="tight")
    print(f"Figure -> {FIG.relative_to(FIG.parents[2])}")

    write_dictionary(feats)
    print(f"Dictionary -> {DICT.relative_to(DICT.parents[2])}")
    return 0


def write_dictionary(feats: list[str]) -> None:
    lines = [
        "# Xususiyatlar lug'ati (`hex_features.parquet`)", "",
        f"Bir qator = bitta H3 katak (resolution {H3_RES}; Toshkentda ~0.88 km², markazdan chekkasigacha ~500 m). Hudud: 12 tuman, Yangi Toshkent kiritilmagan.", "",
        "| Ustun | Ma'no | Modelda |", "|---|---|---|",
        "| `h3`, `lat`, `lon` | katak id va markazi | yo'q |",
        f"| `cv_block` | H3 resolution {H3_CV_RES} ota-katak, fazoviy CV guruhi | guruh |",
        "| `district` | katak markazi tushgan tuman | yo'q |",
    ]
    for c in POI_CATEGORIES + ["apartment_levels"]:
        what = "ko'p qavatli uylar qavatlari yig'indisi" if c == "apartment_levels" else f"OSM `{c}` obyektlari soni"
        if c == "cafe":
            what += " (Safia va raqobatchi brendlar kafelari chiqarilgan — leakage)"
        lines.append(f"| `n_{c}` | {what}, katak ichida | ha |")
        lines.append(f"| `n_{c}_k1` | {what}, katak + 6 qo'shni | ha |")
    lines += [
        "| `district_density` | tumanning rasmiy aholi zichligi, kishi/km² (2023-04-01) | ha |",
        "| `dist_centre_km` | Amir Temur xiyobonigacha masofa, km | ha |",
        "| `nonres_share` | katakning noturar hudud (sanoat, aeroport, qabriston...) bilan qoplangan ulushi | ha (07-qadamda qo'shilgan) |",
        f"| `competitors_1km` | markazdan {COMPETITOR_RADIUS_M} m ichidagi raqobatchilar soni (Bon!, Cake Lab, Breadly) | faqat tushuntirish uchun |",
        f"| `has_competitor_1km` | {COMPETITOR_RADIUS_M} m ichida raqobatchi bor (1/0); Bon!/Breadly to'liq emas | ha |",
        "| `has_safia` | **target**: katakda oddiy Safia filiali bor (1/0) | target |",
        "| `safia_count` | katakdagi oddiy filiallar soni | **yo'q — leakage** |",
        "| `dist_safia_m` | markazdan eng yaqin oddiy Safia'gacha, m | **yo'q — leakage**, faqat 10-qadamda |",
    ]
    DICT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
