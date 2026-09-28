"""Step 06b: population for Tashkent.

Outcome: gridded population products (GHSL 2025, WorldPop 2025) disagree with official district
totals (Spearman ~0), even after masking non-residential land use, so the model uses official
district totals (data/processed/district_population.csv). This script keeps the evidence:

1. Download the GHSL GHS-POP 2025 tile (cached, not in git) and clip it to the city.
2. Compare GHSL district sums with official district populations (and with WorldPop, rejected
   earlier, if its clipped raster from src/06b_worldpop.py exists).
3. Mask cells inside non-residential land use from OSM (industrial, railway, military, cemetery,
   garages, aerodrome). GHSL allocates people by built volume, so large industrial buildings in the
   southern districts received residents. The masked comparison tests that hypothesis.
4. Check our district polygons against official units with OSM counts (schools, supermarkets...).
5. Save official population and density per district. Yangi Toshkent (created after 2023) has none.
"""
import sys
import zipfile

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests
from matplotlib.colors import LogNorm
from shapely.geometry import LineString, Polygon
from shapely.ops import linemerge, polygonize, unary_union
from rasterio.features import rasterize
from rasterio.mask import mask

from config import (CRS_METRIC, CRS_WGS84, DATA_INTERIM, DATA_PROCESSED, DATA_RAW, GHSL_URL, OUTPUTS_FIGURES,
                    TASHKENT_BBOX, USER_AGENT)
from osm_utils import overpass

ZIP = DATA_RAW / "cache" / "ghsl" / GHSL_URL.rsplit("/", 1)[1]
CLIPPED = DATA_RAW / "ghsl_pop_tashkent.tif"
WORLDPOP_CLIPPED = DATA_RAW / "worldpop_tashkent.tif"
OFFICIAL = DATA_INTERIM / "district_population_official.csv"
OUT_CSV = DATA_PROCESSED / "district_population.csv"
OUT_FIG = OUTPUTS_FIGURES / "06b_population.png"
NONRES = DATA_RAW / "osm_nonresidential.geojson"
NONRES_TAGS = {"landuse": ["industrial", "railway", "military", "cemetery", "garages"], "aeroway": ["aerodrome"]}


def build_polygons(elements: list[dict]) -> list:
    """Closed ways -> Polygon; multipolygon relations -> polygons from their outer member rings."""
    out = []
    for el in elements:
        if el["type"] == "way" and len(el.get("geometry", [])) >= 4:
            ring = [(p["lon"], p["lat"]) for p in el["geometry"]]
            if ring[0] == ring[-1]:
                out.append((el, Polygon(ring)))
        elif el["type"] == "relation":
            lines = [LineString([(p["lon"], p["lat"]) for p in m["geometry"]])
                     for m in el.get("members", []) if m.get("role") == "outer" and m.get("geometry")]
            polys = list(polygonize(linemerge(lines))) if lines else []
            if polys:
                out.append((el, unary_union(polys)))
    return out


def load_nonresidential() -> gpd.GeoDataFrame:
    if NONRES.exists():
        return gpd.read_file(NONRES)
    s_, w, n, e = TASHKENT_BBOX
    rows = []
    for key, values in NONRES_TAGS.items():
        rx = "^(" + "|".join(values) + ")$"
        query = (f"[out:json][timeout:180][bbox:{s_},{w},{n},{e}];"
                 f'(way["{key}"~"{rx}"];relation["{key}"~"{rx}"]["type"="multipolygon"];);out geom;')
        print(f"  querying {key} ...")
        for el, geom in build_polygons(overpass(query)["elements"]):
            rows.append({"osm_id": f"{el['type'][0]}{el['id']}", "kind": el["tags"][key], "geometry": geom})
    g = gpd.GeoDataFrame(rows, crs=CRS_WGS84)
    g["geometry"] = g.geometry.make_valid()
    g.to_file(NONRES, driver="GeoJSON")
    return g


def download() -> None:
    if ZIP.exists():
        print(f"Using cached {ZIP.name} ({ZIP.stat().st_size / 1e6:.1f} MB)")
        return
    ZIP.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {GHSL_URL}")
    tmp = ZIP.with_suffix(".part")
    with requests.get(GHSL_URL, headers={"User-Agent": USER_AGENT}, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(1 << 20):
                f.write(chunk)
    tmp.rename(ZIP)


def clip(city: gpd.GeoDataFrame) -> None:
    with zipfile.ZipFile(ZIP) as z:
        tif = next(n for n in z.namelist() if n.endswith(".tif"))
    with rasterio.open(f"zip://{ZIP.as_posix()}!/{tif}") as src:
        data, transform = mask(src, city.to_crs(src.crs).geometry, crop=True, nodata=src.nodata)
        meta = src.meta | {"height": data.shape[1], "width": data.shape[2], "transform": transform,
                           "compress": "deflate"}
    with rasterio.open(CLIPPED, "w", **meta) as dst:
        dst.write(data)


def read_with_districts(path, districts: gpd.GeoDataFrame):
    """Return population array (NaN outside data), district-id array (0 = none), profile."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        arr[arr < 0] = np.nan
        d = districts.to_crs(src.crs)
        ids = rasterize(((g, i + 1) for i, g in enumerate(d.geometry)), out_shape=arr.shape,
                        transform=src.transform, fill=0, dtype="int32")
        return arr, ids, src.profile, [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]


def district_stats(arr, ids, names) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(names, start=1):
        vals = arr[(ids == i) & ~np.isnan(arr)]
        populated = vals[vals > 0]
        cv = populated.std() / populated.mean() if len(populated) else np.nan
        rows.append({"district": name, "sum": vals.sum(), "within_cv": cv})
    return pd.DataFrame(rows)


def compare(stats: pd.DataFrame, official: pd.DataFrame, label: str) -> pd.DataFrame:
    t = stats.merge(official[["district", "population"]], on="district")
    t["ratio"] = t["sum"] / t["population"]
    rho = t["sum"].rank().corr(t["population"].rank())
    rel_err = (t["sum"] / t["sum"].sum() - t["population"] / t["population"].sum()).abs() / (
        t["population"] / t["population"].sum())
    worst = t.loc[rel_err.idxmax()]
    print(f"{label}: total {t['sum'].sum():,.0f} vs official {t['population'].sum():,.0f} | "
          f"Spearman {rho:.2f} | worst district share error {rel_err.max():.0%} ({worst['district']}) | "
          f"median within-district CV {stats['within_cv'].median():.2f}")
    return t.assign(source=label)


def main() -> int:
    refresh = "--refresh" in sys.argv
    city = gpd.read_file(DATA_RAW / "tashkent_boundary.geojson")
    districts = gpd.read_file(DATA_RAW / "tashkent_districts.geojson")
    official = pd.read_csv(OFFICIAL)
    names = districts["district"].tolist()
    if refresh or not CLIPPED.exists():  # the clipped raster is in git; the 34 MB tile is not
        download()
        clip(city)
    else:
        print(f"Using clipped raster {CLIPPED.name} (pass --refresh to download the GHSL tile again)")

    ghsl, ids, profile, extent = read_with_districts(CLIPPED, districts)
    print(f"GHSL clipped: {ghsl.shape[1]}x{ghsl.shape[0]} cells, city total {np.nansum(ghsl):,.0f}\n")
    ghsl_stats = district_stats(ghsl, ids, names)

    print("Before calibration (12 districts with official figures):")
    tables = [compare(ghsl_stats, official, "GHSL 2025")]
    wp = None
    if WORLDPOP_CLIPPED.exists():
        wp, wp_ids, _, wp_extent = read_with_districts(WORLDPOP_CLIPPED, districts)
        tables.append(compare(district_stats(wp, wp_ids, names), official, "WorldPop 2025"))

    # Mask non-residential land use (hypothesis test: does agreement with official figures improve?)
    nonres = load_nonresidential()
    nonres = nonres[nonres.intersects(city.geometry.iloc[0])]
    with rasterio.open(CLIPPED) as src:
        nonres_mask = rasterize(nonres.to_crs(src.crs).geometry, out_shape=ghsl.shape,
                                transform=src.transform, fill=0, default_value=1, dtype="uint8").astype(bool)
    masked = np.where(nonres_mask, 0.0, ghsl)
    area = nonres.to_crs("EPSG:32642").area.groupby(nonres["kind"]).sum() / 1e6
    print(f"  OSM non-residential polygons: {len(nonres)} ({dict(area.round(1))} km2)")
    print(f"  GHSL people inside them: {np.nansum(ghsl[nonres_mask]):,.0f} "
          f"({np.nansum(ghsl[nonres_mask]) / np.nansum(ghsl):.0%} of the city total)")
    masked_stats = district_stats(masked, ids, names)
    tables.append(compare(masked_stats, official, "GHSL masked"))
    print("  (within-district CV: higher = more variation inside a district; a flat 'block' is close to 0)")

    both = pd.concat(tables).pivot_table(index="district", columns="source", values="ratio")
    both.insert(0, "official", official.set_index("district")["population"])
    print("\nRatio model / official by district:")
    print(both.sort_values("official", ascending=False).round(2).to_string())

    # Independent check that our district polygons match the official units:
    # OSM counts that scale with residents should rank districts like the official figures do.
    pois = gpd.read_file(DATA_RAW / "osm_pois.geojson")
    in_d = gpd.sjoin(pois, districts[["district", "geometry"]], predicate="within")
    counts = in_d.pivot_table(index="district", columns="category", values="osm_id", aggfunc="size", fill_value=0)
    off = official.set_index("district")["population"]
    rhos = {c: round(counts.reindex(off.index)[c].rank().corr(off.rank()), 2)
            for c in ["supermarket", "school", "bus_stop", "apartments"]}
    print(f"\nOSM counts vs official population (Spearman): {rhos}")

    # Decision (step 06b): gridded products are rejected; population = official district totals
    d = districts[["district", "geometry"]].copy()
    d["km2"] = d.to_crs(CRS_METRIC).area / 1e6
    d = d.merge(official[["district", "population", "as_of"]], on="district", how="left")
    d["density_per_km2"] = d["population"] / d["km2"]
    out = d.drop(columns="geometry")
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nSaved {OUT_CSV.relative_to(OUT_CSV.parents[2])} (Yangi Toshkent has no official figure -> empty)")
    print(out.sort_values("density_per_km2", ascending=False).round(1).to_string(index=False))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1, 1.1]})
    ratio = both.drop(columns="official").sort_values("GHSL 2025")
    y = np.arange(len(ratio))
    for k, (col, color) in enumerate(zip(ratio.columns, ["#e67e22", "#c0392b", "#2980b9"])):
        ax1.barh(y + (k - 1) * 0.27, ratio[col], height=0.27, color=color, label=col)
    ax1.axvline(1, color="black", linewidth=1)
    ax1.set_xscale("log")
    ticks = [0.25, 0.5, 1, 2, 4, 8]
    ax1.set_xticks(ticks, [f"×{t:g}" for t in ticks])
    ax1.xaxis.set_minor_formatter(plt.NullFormatter())
    ax1.set_yticks(y, ratio.index)
    ax1.set_xlabel("Model population / official population (log scale; 1 = correct)")
    ax1.set_title("Global population grids vs official district totals (2023)")
    ax1.legend(loc="lower right")
    d.plot(column="density_per_km2", ax=ax2, cmap="viridis", legend=True, edgecolor="white",
           missing_kwds={"color": "#dddddd", "label": "no official figure"},
           legend_kwds={"label": "People per km2 (official, 2023-04-01)", "shrink": 0.7})
    for r in d.itertuples():
        pt = r.geometry.representative_point()
        ax2.annotate(r.district, (pt.x, pt.y), color="white", fontsize=7, ha="center")
    ax2.set_title("Population density by district (used in the model)")
    ax2.set_axis_off()
    fig.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    print(f"Figure -> {OUT_FIG.relative_to(OUT_FIG.parents[2])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
