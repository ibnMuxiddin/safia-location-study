"""Step 04: check that the coordinates published on the Safia site are plausible.

Automatic checks:
  A. district written in the address text vs district the point falls into
  B. independent source: Safia shops mapped by OpenStreetMap volunteers vs site coordinates
  C. suspiciously close pairs of branches
  D. coordinate precision
Manual check: 10 branches (one per district where possible) plus every flagged branch,
with map links, for the user.

Note: forward geocoding with Nominatim was tried and rejected. It found only 6/123
addresses and matched unrelated shops on the same street.
"""
import json
import re
import sys

import folium
import geopandas as gpd
import numpy as np
import pandas as pd

from map_utils import add_basemaps
from config import CRS_METRIC, CRS_WGS84, DATA_INTERIM, DATA_RAW, OUTPUTS_MAPS, TASHKENT_BBOX
from osm_utils import overpass

IN_CSV = DATA_INTERIM / "safia_tashkent.csv"
OUT_CSV = DATA_INTERIM / "coords_check.csv"
OUT_MAP = OUTPUTS_MAPS / "04_branches_check.html"
OSM_SAFIA = DATA_RAW / "osm_safia.json"
MANUAL_CHECKS = DATA_INTERIM / "manual_checks.csv"  # results of the user's check on Yandex maps

OSM_MATCH_M = 150       # an OSM Safia shop this close confirms the site point
CLOSE_PAIR_M = 100      # two branches closer than this are flagged
SHOP_KINDS = {"confectionery", "bakery", "pastry", "cafe"}

RU_DISTRICTS = {
    "Алмазарский": "Olmazor", "Бектемирский": "Bektemir", "Мирабадский": "Mirobod",
    "Мирзо-Улугбекский": "Mirzo Ulug'bek", "Сергелийский": "Sergeli", "Учтепинский": "Uchtepa",
    "Чиланзарский": "Chilonzor", "Шайхантахурский": "Shayxontohur", "Юнусабадский": "Yunusobod",
    "Яккасарайский": "Yakkasaroy", "Яшнабадский": "Yashnobod", "Янгихаётский": "Yangihayot",
}


def text_district(address: str) -> str | None:
    m = re.search(r"([А-Яа-яЁё\-]+) район", address)
    return RU_DISTRICTS.get(m.group(1)) if m else None


def load_osm_safia() -> gpd.GeoDataFrame:
    if not OSM_SAFIA.exists():
        s, w, n, e = TASHKENT_BBOX
        query = (f'[out:json][timeout:60][bbox:{s},{w},{n},{e}];'
                 '(nwr["name"~"safia|сафия",i];nwr["brand"~"safia",i];);out center tags;')
        OSM_SAFIA.write_text(json.dumps(overpass(query), ensure_ascii=False), encoding="utf-8")
    rows = []
    for el in json.loads(OSM_SAFIA.read_text(encoding="utf-8"))["elements"]:
        tags, centre = el["tags"], el.get("center", {})
        rows.append({"osm_id": f"{el['type'][0]}{el['id']}", "osm_name": tags.get("name"),
                     "kind": tags.get("shop") or tags.get("amenity"),
                     "lat": el.get("lat", centre.get("lat")), "lon": el.get("lon", centre.get("lon"))})
    osm = pd.DataFrame(rows)
    osm = osm[osm["kind"].isin(SHOP_KINDS)]
    return gpd.GeoDataFrame(osm, geometry=gpd.points_from_xy(osm["lon"], osm["lat"]), crs=CRS_WGS84)


def decimals(x: float) -> int:
    s = repr(float(x))
    return len(s.split(".")[1]) if "." in s else 0


def main() -> int:
    df = pd.read_csv(IN_CSV)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=CRS_WGS84).to_crs(CRS_METRIC)

    # A. district in text vs spatial district
    df["district_text"] = df["address"].map(text_district)
    has_text = df["district_text"].notna()
    mismatch = has_text & (df["district_text"] != df["district"])
    print(f"A. Address names a district: {has_text.sum()} | matches: {(has_text & ~mismatch).sum()} | mismatches: {mismatch.sum()}")
    if mismatch.any():
        print(df.loc[mismatch, ["name", "address", "district_text", "district"]].to_string(index=False))

    # B. OSM Safia shops as an independent source
    osm = load_osm_safia().to_crs(CRS_METRIC)
    nearest = gpd.sjoin_nearest(gdf[["branch_id", "geometry"]], osm[["osm_id", "geometry"]],
                                how="left", distance_col="osm_dist_m")
    nearest = nearest.drop_duplicates("branch_id")
    df["osm_dist_m"] = nearest["osm_dist_m"].values
    confirmed = df["osm_dist_m"] <= OSM_MATCH_M
    print(f"\nB. Safia shops in OSM (confectionery/bakery/pastry/cafe): {len(osm)}")
    print(f"   Site branches with an OSM Safia shop within {OSM_MATCH_M} m: {confirmed.sum()}/{len(df)} "
          f"| median distance of these: {df.loc[confirmed, 'osm_dist_m'].median():.0f} m")
    buckets = pd.cut(df["osm_dist_m"], [0, 50, 150, 300, 1000, np.inf], right=True, include_lowest=True)
    print("   Distance to nearest OSM Safia shop:")
    print(buckets.value_counts().sort_index().to_string())

    # Reverse direction: OSM shops inside the city with no site branch nearby
    city = gpd.read_file(DATA_RAW / "tashkent_boundary.geojson").to_crs(CRS_METRIC).geometry.iloc[0]
    osm_city = osm[osm.within(city)]
    back = gpd.sjoin_nearest(osm_city, gdf[["name", "geometry"]], distance_col="site_dist_m")
    back = back.drop_duplicates("osm_id")
    orphan = back[back["site_dist_m"] > OSM_MATCH_M]
    print(f"   OSM Safia shops inside the city: {len(osm_city)} | with a site branch within {OSM_MATCH_M} m: "
          f"{len(osm_city) - len(orphan)} | without (closed or moved?): {len(orphan)}")
    for r in orphan.itertuples():
        print(f"     {r.osm_id} {r.osm_name} ({r.kind}): nearest site branch {r.name} at {r.site_dist_m:.0f} m")

    # C. close pairs
    xy = np.c_[gdf.geometry.x, gdf.geometry.y]
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    df["nearest_branch_m"] = d.min(1)
    i, j = np.where(np.triu(d < CLOSE_PAIR_M))
    print(f"\nC. Pairs closer than {CLOSE_PAIR_M} m: {len(i)}")
    for a, b in zip(i, j):
        print(f"   {df.name[a]} <-> {df.name[b]}: {d[a, b]:.0f} m | {df.address[a]} | {df.address[b]}")
    print(f"   Nearest-branch distance: median {df['nearest_branch_m'].median():.0f} m")

    # D. precision
    df["coord_decimals"] = df[["lat", "lon"]].apply(lambda r: min(decimals(r.lat), decimals(r.lon)), axis=1)
    print(f"\nD. Coordinate decimals (min of lat/lon): {df['coord_decimals'].value_counts().sort_index().to_dict()}"
          f"  (4 decimals ~ 10 m)")

    # Manual check: one random branch per district, topped up to 10, plus all flagged
    df["flag"] = np.select([mismatch], ["district text differs"], default="")
    sample = df.groupby("district").sample(1, random_state=7)
    if len(sample) < 10:
        sample = pd.concat([sample, df.drop(sample.index).sample(10 - len(sample), random_state=7)])
    df["manual_check"] = df.index.isin(sample.head(10).index) | (df["flag"] != "")
    print("\nManual check list (open the link, compare the pin with the address):")
    for r in df[df["manual_check"]].itertuples():
        note = f"  <-- {r.flag}" if r.flag else ""
        print(f"   {r.name} | {r.district} | {r.address} | OSM Safia {r.osm_dist_m:.0f} m | "
              f"https://yandex.uz/maps/?pt={r.lon},{r.lat}&z=18&l=map{note}")

    if MANUAL_CHECKS.exists():
        manual = pd.read_csv(MANUAL_CHECKS)
        df = df.merge(manual[["branch_id", "result"]].rename(columns={"result": "manual_result"}),
                      on="branch_id", how="left")
        print(f"\nUser's manual check: {manual['result'].value_counts().to_dict()}")
        bad = manual[manual["result"] != "ok"]
        for r in bad.itertuples():
            print(f"   {r.name}: {r.result} - {r.note}")
    else:
        print("\nUser's manual check: not done yet")

    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nSaved {OUT_CSV.name}")

    m = folium.Map(location=[41.30, 69.27], zoom_start=11, tiles=None)
    add_basemaps(m)
    osm_layer = folium.FeatureGroup(name="OSM Safia shops (grey)")
    for r in osm.to_crs(CRS_WGS84).itertuples():
        folium.CircleMarker([r.lat, r.lon], radius=3, color="#7f8c8d", fill=True, fill_opacity=0.8,
                            tooltip=f"OSM: {r.osm_name} ({r.kind})").add_to(osm_layer)
    site_layer = folium.FeatureGroup(name="Site branches")
    for r in df.itertuples():
        color = "#e67e22" if r.flag else ("#2980b9" if r.manual_check else "#c0392b")
        link = f"https://yandex.uz/maps/?pt={r.lon},{r.lat}&z=18&l=map"
        folium.CircleMarker([r.lat, r.lon], radius=7 if r.manual_check else 4, color=color,
                            fill=True, fill_opacity=0.9,
                            popup=folium.Popup(f"<b>{r.name}</b><br>{r.address}<br>{r.district}<br>"
                                               f"OSM Safia: {r.osm_dist_m:.0f} m<br>"
                                               f'<a href="{link}" target="_blank">Yandex</a>',
                                               max_width=300)).add_to(site_layer)
    osm_layer.add_to(m)
    site_layer.add_to(m)
    folium.LayerControl().add_to(m)
    m.save(OUT_MAP)
    print(f"Map -> {OUT_MAP.relative_to(OUT_MAP.parents[2])} "
          "(red = site, blue = manual sample, orange = flagged, grey = OSM Safia)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
