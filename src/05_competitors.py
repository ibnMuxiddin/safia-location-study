"""Step 05: competitor branches (Cake Lab, Bon!, Breadly) inside Tashkent.

Sources, in order of trust:
  1. Cake Lab website: the /ru/branches page embeds a "terminals" list with coordinates.
  2. OpenStreetMap: Bon!, Breadly (and Cake Lab as a cross-check).
  3. Optional manual additions: data/interim/competitors_manual.csv
     (columns: brand,name,address,lat,lon,source), e.g. branches the user copied from Yandex.
2GIS is not scraped: its terms forbid automated collection.
"""
import json
import re
import sys

import folium
import geopandas as gpd
import pandas as pd
import requests

from map_utils import add_basemaps
from config import (CRS_METRIC, CRS_WGS84, DATA_INTERIM, DATA_RAW, OUTPUTS_MAPS, TASHKENT_BBOX,
                    USER_AGENT)
from osm_utils import overpass

CAKELAB_URL = "https://cakelab.uz/ru/branches"
CAKELAB_HTML = DATA_RAW / "cakelab_branches.html"
CAKELAB_CSV = DATA_RAW / "cakelab_branches.csv"  # extracted table; kept when the HTML is not published
OSM_JSON = DATA_RAW / "osm_competitors.json"
MANUAL_CSV = DATA_INTERIM / "competitors_manual.csv"
OUT_CSV = DATA_INTERIM / "competitors_geo.csv"
OUT_MAP = OUTPUTS_MAPS / "05_competitors.html"

DEDUP_M = 150     # an OSM point this close to a website point of the same brand is the same branch
SAME_SHOP_M = 50  # two OSM points of the same brand this close are one shop
BRAND_PATTERNS = {
    "Cake Lab": r"cake ?lab|кейк ?лаб",
    "Bon!": r"^\s*bon\s*!?\s*$",
    "Breadly": r"breadly|брэдли|бредли",
}
BRAND_COLORS = {"Cake Lab": "#27ae60", "Bon!": "#8e44ad", "Breadly": "#d35400"}


def cakelab_from_site() -> pd.DataFrame:
    if not CAKELAB_HTML.exists() and CAKELAB_CSV.exists() and "--refresh" not in sys.argv:
        # Public copy of the repo: the page's HTML is not redistributed, the extracted table is
        print(f"No {CAKELAB_HTML.name}; using the committed {CAKELAB_CSV.name}")
        return pd.read_csv(CAKELAB_CSV)
    if not CAKELAB_HTML.exists() or "--refresh" in sys.argv:
        resp = requests.get(CAKELAB_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
        resp.raise_for_status()
        CAKELAB_HTML.write_text(resp.text, encoding="utf-8")
    html = CAKELAB_HTML.read_text(encoding="utf-8")
    # Next.js streams page data as JS string literals inside self.__next_f.push([1, "..."])
    chunks = re.findall(r'self\.__next_f\.push\(\[1,("(?:[^"\\]|\\.)*")\]\)', html)
    payload = "".join(json.loads(c) for c in chunks)
    start = payload.index('"terminals":[') + len('"terminals":')
    terminals, _ = json.JSONDecoder().raw_decode(payload[start:])
    rows = [{"brand": "Cake Lab", "name": t["name_ru"], "address": t.get("address_ru") or t.get("address_uz"),
             "lat": float(t["latitude"]), "lon": float(t["longitude"]), "source": "cakelab.uz"}
            for t in terminals]
    df = pd.DataFrame(rows)
    df.to_csv(CAKELAB_CSV, index=False, encoding="utf-8")
    return df


def from_osm() -> pd.DataFrame:
    if not OSM_JSON.exists():
        s, w, n, e = TASHKENT_BBOX
        parts = "".join(f'nwr["name"~"{p}",i];nwr["brand"~"{p}",i];' for p in BRAND_PATTERNS.values())
        query = f"[out:json][timeout:60][bbox:{s},{w},{n},{e}];({parts});out center tags;"
        OSM_JSON.write_text(json.dumps(overpass(query), ensure_ascii=False), encoding="utf-8")
    rows = []
    for el in json.loads(OSM_JSON.read_text(encoding="utf-8"))["elements"]:
        tags, centre = el["tags"], el.get("center", {})
        label = tags.get("brand") or tags.get("name") or ""
        brand = next((b for b, p in BRAND_PATTERNS.items()
                      if re.search(p, label, re.I) or re.search(p, tags.get("name", ""), re.I)), None)
        if brand is None:
            continue
        street = " ".join(filter(None, [tags.get("addr:street"), tags.get("addr:housenumber")]))
        rows.append({"brand": brand, "name": tags.get("name"), "address": street or None,
                     "lat": el.get("lat", centre.get("lat")), "lon": el.get("lon", centre.get("lon")),
                     "source": f"osm:{el['type'][0]}{el['id']}"})
    return pd.DataFrame(rows)


def to_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=CRS_WGS84)


def main() -> int:
    site = to_gdf(cakelab_from_site())
    osm = to_gdf(from_osm())
    print(f"Cake Lab website: {len(site)} branches | OSM objects: {osm['brand'].value_counts().to_dict()}")

    # Cake Lab: the official website is authoritative, OSM is only a cross-check of its coordinates
    osm_cl = osm[osm["brand"] == "Cake Lab"].to_crs(CRS_METRIC)
    near = gpd.sjoin_nearest(osm_cl, site.to_crs(CRS_METRIC)[["geometry"]], distance_col="d")
    near = near[~near.index.duplicated()]
    print(f"OSM Cake Lab points matching a website branch within {DEDUP_M} m: {(near['d'] <= DEDUP_M).sum()}/{len(osm_cl)} "
          f"(distances: {sorted(near['d'].round().astype(int).tolist())})")
    for r in near[near["d"] > DEDUP_M].itertuples():
        print(f"  not on the official site (probably closed, dropped): {r.source} at {r.d:.0f} m")
    osm = osm[osm["brand"] != "Cake Lab"]

    # The same shop is sometimes mapped twice in OSM (a node and its building)
    osm_m = osm.to_crs(CRS_METRIC)
    keep = []
    for _, grp in osm_m.groupby("brand"):
        for idx, geom in grp.geometry.items():
            if all(geom.distance(osm_m.geometry[k]) > SAME_SHOP_M for k in keep if osm_m.at[k, "brand"] == osm_m.at[idx, "brand"]):
                keep.append(idx)
    dropped = osm.index.difference(keep)
    print(f"OSM duplicates within {SAME_SHOP_M} m (dropped): {osm.loc[dropped, 'source'].tolist()}")
    osm = osm.loc[keep]

    parts = [site, osm]
    if MANUAL_CSV.exists():
        manual = to_gdf(pd.read_csv(MANUAL_CSV))
        print(f"Manual additions: {len(manual)}")
        parts.append(manual)
    allc = pd.concat(parts, ignore_index=True)

    districts = gpd.read_file(DATA_RAW / "tashkent_districts.geojson")[["district", "geometry"]]
    allc = gpd.sjoin(allc, districts, how="left", predicate="within").drop(columns="index_right")
    outside = allc["district"].isna()
    print(f"Outside Tashkent city (dropped): {outside.sum()} {allc.loc[outside, ['brand', 'name']].values.tolist()}")
    allc = allc[~outside]

    out = allc[["brand", "name", "address", "lat", "lon", "source", "district"]]
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nSaved {OUT_CSV.name}: {len(out)} branches")
    summary = out.assign(src=out["source"].str.split(":").str[0]).pivot_table(
        index="brand", columns="src", values="name", aggfunc="size", fill_value=0)
    summary["total"] = summary.sum(axis=1)
    print(summary.to_string())

    m = folium.Map(location=[41.30, 69.27], zoom_start=11, tiles=None)
    add_basemaps(m)
    for brand, grp in out.groupby("brand"):
        layer = folium.FeatureGroup(name=f"{brand} ({len(grp)})")
        for r in grp.itertuples():
            folium.CircleMarker([r.lat, r.lon], radius=5, color=BRAND_COLORS[brand], fill=True,
                                fill_opacity=0.9, tooltip=f"{brand}: {r.name} | {r.address} | {r.source}").add_to(layer)
        layer.add_to(m)
    safia = pd.read_csv(DATA_INTERIM / "safia_tashkent.csv")
    s_layer = folium.FeatureGroup(name=f"Safia ({len(safia)})", show=False)
    for r in safia.itertuples():
        folium.CircleMarker([r.lat, r.lon], radius=3, color="#c0392b", fill=True, fill_opacity=0.6,
                            tooltip=f"Safia: {r.name}").add_to(s_layer)
    s_layer.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(OUT_MAP)
    print(f"Map -> {OUT_MAP.relative_to(OUT_MAP.parents[2])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
