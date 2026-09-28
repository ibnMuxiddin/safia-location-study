"""Step 06b: WorldPop population raster for Tashkent.

Downloads the national 100 m raster once (cached, not in git), clips it to the city polygon
and reports population by district. The district ranking is compared with the OSM
apartment-building density from step 06 as a sanity check of both sources.
"""
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests
from matplotlib.colors import LogNorm
from rasterio.mask import mask

from config import (CRS_METRIC, DATA_RAW, OUTPUTS_FIGURES, USER_AGENT, WORLDPOP_URL,
                    WORLDPOP_YEAR)

NATIONAL = DATA_RAW / "cache" / "worldpop" / WORLDPOP_URL.rsplit("/", 1)[1]
CLIPPED = DATA_RAW / "worldpop_tashkent.tif"
OUT_FIG = OUTPUTS_FIGURES / "06b_population.png"


def download() -> None:
    if NATIONAL.exists():
        print(f"Using cached {NATIONAL.name} ({NATIONAL.stat().st_size / 1e6:.1f} MB)")
        return
    NATIONAL.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {WORLDPOP_URL}")
    tmp = NATIONAL.with_suffix(".part")
    with requests.get(WORLDPOP_URL, headers={"User-Agent": USER_AGENT}, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(1 << 20):
                f.write(chunk)
    tmp.rename(NATIONAL)


def clip(city: gpd.GeoDataFrame) -> None:
    with rasterio.open(NATIONAL) as src:
        data, transform = mask(src, city.to_crs(src.crs).geometry, crop=True, nodata=src.nodata)
        meta = src.meta | {"height": data.shape[1], "width": data.shape[2], "transform": transform,
                           "compress": "deflate"}
    with rasterio.open(CLIPPED, "w", **meta) as dst:
        dst.write(data)


def zone_sum(src, geom) -> float:
    data, _ = mask(src, [geom], crop=True, nodata=src.nodata)
    arr = data[0]
    return float(arr[(arr != src.nodata) & ~np.isnan(arr)].sum())


def main() -> int:
    city = gpd.read_file(DATA_RAW / "tashkent_boundary.geojson")
    districts = gpd.read_file(DATA_RAW / "tashkent_districts.geojson")
    if "--refresh" in sys.argv or not CLIPPED.exists():  # clipped raster is in git; the 36 MB file is not
        download()
        clip(city)

    with rasterio.open(CLIPPED) as src:
        print(f"Clipped raster: {src.width}x{src.height} cells, res {src.res[0]:.6f} deg, crs {src.crs}")
        total = zone_sum(src, city.to_crs(src.crs).geometry.iloc[0])
        d = districts.to_crs(src.crs)
        d["population"] = [zone_sum(src, g) for g in d.geometry]
        arr = src.read(1, masked=True)
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

    print(f"\nTashkent population, WorldPop {WORLDPOP_YEAR}: {total:,.0f}")
    d["km2"] = districts.to_crs(CRS_METRIC).area / 1e6
    d["density_per_km2"] = d["population"] / d["km2"]
    print(f"Sum over districts: {d['population'].sum():,.0f}")

    # Cross-check with OSM apartment buildings (step 06)
    pois = gpd.read_file(DATA_RAW / "osm_pois.geojson")
    apts = gpd.sjoin(pois[pois["category"] == "apartments"], districts[["district", "geometry"]], predicate="within")
    d["osm_apts_per_km2"] = d["district"].map(apts["district"].value_counts()).fillna(0) / d["km2"]
    table = d[["district", "population", "km2", "density_per_km2", "osm_apts_per_km2"]].sort_values(
        "density_per_km2", ascending=False)
    print("\nBy district:")
    print(table.round({"population": 0, "km2": 1, "density_per_km2": 0, "osm_apts_per_km2": 1}).to_string(index=False))
    rho = table["density_per_km2"].rank().corr(table["osm_apts_per_km2"].rank())
    print(f"\nSpearman rank correlation, WorldPop density vs OSM apartments/km2: {rho:.2f}")

    fig, ax = plt.subplots(figsize=(8, 8))
    img = ax.imshow(np.ma.masked_less_equal(arr, 0), extent=extent, cmap="viridis",
                    norm=LogNorm(vmin=1, vmax=float(arr.max())))
    d.boundary.plot(ax=ax, color="white", linewidth=0.6)
    for r in d.itertuples():
        pt = r.geometry.representative_point()
        ax.annotate(r.district, (pt.x, pt.y), color="white", fontsize=7, ha="center")
    fig.colorbar(img, ax=ax, shrink=0.7, label="People per 100 m cell (log scale)")
    ax.set_title(f"Tashkent population, WorldPop {WORLDPOP_YEAR} (100 m, constrained)")
    ax.set_axis_off()
    fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    print(f"Figure -> {OUT_FIG.relative_to(OUT_FIG.parents[2])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
