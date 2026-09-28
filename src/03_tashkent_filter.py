"""Step 03: download the Tashkent city boundary and districts, keep branches inside the city.

Filtering is done by polygon, not by address text: step 02 showed the text is unreliable
(missing city names, Yangihayot district written without "Ташкент").
"""
import argparse
import sys

import folium
import geopandas as gpd
import osmnx as ox
import pandas as pd

from map_utils import add_basemaps
from config import (CRS_METRIC, CRS_WGS84, DATA_INTERIM, DATA_RAW, OSM_CACHE, OUTPUTS_MAPS,
                    TASHKENT_DISTRICTS, TASHKENT_RELATION, USER_AGENT)

BOUNDARY = DATA_RAW / "tashkent_boundary.geojson"
DISTRICTS = DATA_RAW / "tashkent_districts.geojson"
IN_CSV = DATA_INTERIM / "safia_branches.csv"
OUT_CSV = DATA_INTERIM / "safia_tashkent.csv"
OUT_MAP = OUTPUTS_MAPS / "03_tashkent_branches.html"

# Branches that do not serve ordinary city foot traffic. Kept in the data, flagged for step 07.
SPECIAL_SITES = {
    "Дьюти-фри": "airport airside (duty free)",
    "Фабрика": "store at the central factory",
}


def load_boundaries(refresh: bool) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if BOUNDARY.exists() and DISTRICTS.exists() and not refresh:
        print("Using cached boundaries")
        return gpd.read_file(BOUNDARY), gpd.read_file(DISTRICTS)

    ox.settings.http_user_agent = USER_AGENT
    ox.settings.cache_folder = str(OSM_CACHE)
    city = ox.geocode_to_gdf(f"R{TASHKENT_RELATION}", by_osmid=True)[["osm_id", "geometry"]]
    city["name"] = "Toshkent"
    ids = list(TASHKENT_DISTRICTS)
    districts = ox.geocode_to_gdf([f"R{i}" for i in ids], by_osmid=True)[["osm_id", "geometry"]]
    districts["district"] = districts["osm_id"].map(TASHKENT_DISTRICTS)
    city.to_file(BOUNDARY, driver="GeoJSON")
    districts.to_file(DISTRICTS, driver="GeoJSON")
    return city, districts


def show(df: pd.DataFrame) -> str:
    return df.to_string(index=False) if len(df) else "  none"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="download boundaries again")
    args = parser.parse_args()

    city, districts = load_boundaries(args.refresh)
    city_m, districts_m = city.to_crs(CRS_METRIC), districts.to_crs(CRS_METRIC)
    print(f"City area: {city_m.area.sum() / 1e6:.1f} km2")
    districts_m["km2"] = districts_m.area / 1e6
    print(districts_m[["district", "km2"]].round(1).sort_values("km2").to_string(index=False))
    uncovered = city_m.geometry.iloc[0].difference(districts_m.union_all()).area / 1e6
    print(f"City area not covered by districts: {uncovered:.2f} km2")

    df = pd.read_csv(IN_CSV)
    pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=CRS_WGS84)
    pts = gpd.sjoin(pts, districts[["district", "geometry"]], how="left", predicate="within")
    pts = pts.drop(columns="index_right")
    in_city = pts["district"].notna()
    text_tashkent = pts["city"].eq("Ташкент")

    print(f"\nBranches total: {len(pts)} | inside city polygon: {in_city.sum()} | text says Ташкент: {text_tashkent.sum()}")
    cols = ["name", "city", "address"]
    print("\nInside polygon, but address text is not 'Ташкент':")
    print(show(pts.loc[in_city & ~text_tashkent, cols + ["district"]]))
    print("\nAddress text is 'Ташкент', but outside polygon:")
    print(show(pts.loc[~in_city & text_tashkent, cols + ["lat", "lon"]]))

    out = pts.loc[in_city].drop(columns="geometry")
    out["special_site"] = out["name"].map(SPECIAL_SITES)
    print("\nFlagged special sites:")
    print(show(out.loc[out["special_site"].notna(), ["name", "address", "special_site"]]))
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nSaved {len(out)} branches -> {OUT_CSV.name}")
    print("\nBranches by district:")
    print(out["district"].value_counts().to_string())

    m = folium.Map(location=[41.30, 69.27], zoom_start=11, tiles=None)
    add_basemaps(m)
    folium.GeoJson(districts, name="Districts", tooltip=folium.GeoJsonTooltip(["district"]),
                   style_function=lambda _: {"color": "#555", "weight": 1, "fillOpacity": 0.03}).add_to(m)
    for r in pts.itertuples():
        color = "#c0392b" if pd.notna(r.district) else "#7f8c8d"
        folium.CircleMarker([r.lat, r.lon], radius=4, color=color, fill=True, fill_opacity=0.9,
                            tooltip=f"{r.name} | {r.address} | {r.hours}").add_to(m)
    folium.LayerControl().add_to(m)
    m.save(OUT_MAP)
    print(f"Map -> {OUT_MAP.relative_to(OUT_MAP.parents[2])} (red = inside city, grey = outside)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
