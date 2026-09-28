"""Step 06: points of interest that signal demand, from OpenStreetMap.

One Overpass query per category (big combined queries time out). Raw responses are kept
in data/raw/overpass/ (in git), the combined result is data/raw/osm_pois.geojson.
Ways and relations are reduced to their centre point. Only objects inside the city polygon are kept.
"""
import argparse
import json
import sys

import folium
import geopandas as gpd
import pandas as pd

from config import CRS_METRIC, CRS_WGS84, DATA_RAW, OUTPUTS_MAPS, TASHKENT_BBOX
from map_utils import add_basemaps, safe_text
from osm_utils import overpass

CACHE = DATA_RAW / "overpass"  # tracked in git: OSM changes over time, results must be reproducible
OUT = DATA_RAW / "osm_pois.geojson"
OUT_MAP = OUTPUTS_MAPS / "06_pois_check.html"

# category -> Overpass filters (each filter is applied to nodes, ways and relations)
CATEGORIES = {
    "university": ['["amenity"~"^(university|college)$"]'],
    "school": ['["amenity"="school"]'],
    "metro": ['["railway"="station"]["station"="subway"]', '["public_transport"="station"]["subway"="yes"]'],
    "bus_stop": ['["highway"="bus_stop"]', '["public_transport"="platform"]["bus"="yes"]'],
    "office": ['["office"]', '["building"="office"]'],
    "mall": ['["shop"="mall"]'],
    "supermarket": ['["shop"="supermarket"]'],
    "hospital": ['["amenity"~"^(hospital|clinic)$"]'],
    "apartments": ['["building"="apartments"]'],
    "cafe": ['["amenity"="cafe"]'],
}
MAP_CHECK = {"metro": "#e74c3c", "university": "#2980b9", "mall": "#27ae60"}
DEDUP_M = 30  # the same object mapped twice within one category (e.g. two metro station tags)


def fetch(category: str, filters: list[str], refresh: bool) -> list[dict]:
    path = CACHE / f"{category}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))["elements"]
    s, w, n, e = TASHKENT_BBOX
    body = "".join(f"nwr{f};" for f in filters)
    query = f"[out:json][timeout:120][bbox:{s},{w},{n},{e}];({body});out center tags;"
    print(f"  querying {category} ...")
    data = overpass(query)
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data["elements"]


def to_rows(category: str, elements: list[dict]) -> list[dict]:
    rows = []
    for el in elements:
        centre = el.get("center", {})
        lat, lon = el.get("lat", centre.get("lat")), el.get("lon", centre.get("lon"))
        if lat is None:
            continue
        tags = el.get("tags", {})
        rows.append({"category": category, "osm_id": f"{el['type'][0]}{el['id']}", "name": tags.get("name"),
                     "levels": pd.to_numeric(tags.get("building:levels"), errors="coerce"), "lat": lat, "lon": lon})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="query Overpass again")
    args = parser.parse_args()

    rows = []
    for cat, filters in CATEGORIES.items():
        rows += to_rows(cat, fetch(cat, filters, args.refresh))
    df = pd.DataFrame(rows).drop_duplicates(["category", "osm_id"])
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=CRS_WGS84)

    city = gpd.read_file(DATA_RAW / "tashkent_boundary.geojson").geometry.iloc[0]
    in_bbox = len(gdf)
    gdf = gdf[gdf.within(city)]

    # Same object tagged twice in one category (e.g. a station node and its building)
    m = gdf.to_crs(CRS_METRIC)
    m["cell"] = (m.geometry.x // DEDUP_M).astype(int).astype(str) + "_" + (m.geometry.y // DEDUP_M).astype(int).astype(str)
    before = len(m)
    m = m.drop_duplicates(["category", "cell"])
    gdf = gdf.loc[m.index]
    print(f"\nObjects in bbox: {in_bbox} | inside city: {before} | after {DEDUP_M} m de-duplication: {len(gdf)}")

    gdf.drop(columns=["lat", "lon"]).to_file(OUT, driver="GeoJSON")
    print(f"Saved {OUT.relative_to(OUT.parents[2])}")

    summary = gdf.groupby("category").agg(count=("osm_id", "size"), named=("name", lambda s: s.notna().sum()))
    summary = summary.reindex(CATEGORIES.keys()).fillna(0).astype(int)
    print("\nObjects per category (inside city):")
    print(summary.to_string())
    empty = summary.index[summary["count"] == 0].tolist()
    print(f"Empty categories: {empty or 'none'}")
    apts = gdf[gdf["category"] == "apartments"]
    print(f"Apartment buildings with building:levels: {apts['levels'].notna().sum()}/{len(apts)} "
          f"(median levels {apts['levels'].median():.0f})")

    fmap = folium.Map(location=[41.30, 69.27], zoom_start=11, tiles=None)
    add_basemaps(fmap)
    for cat, color in MAP_CHECK.items():
        layer = folium.FeatureGroup(name=f"{cat} ({(gdf['category'] == cat).sum()})")
        for r in gdf[gdf["category"] == cat].itertuples():
            folium.CircleMarker([r.geometry.y, r.geometry.x], radius=4, color=color, fill=True, fill_opacity=0.9,
                                tooltip=safe_text(f"{cat}: {r.name or '-'}")).add_to(layer)
        layer.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.save(OUT_MAP)
    print(f"Map (metro, university, mall) -> {OUT_MAP.relative_to(OUT_MAP.parents[2])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
