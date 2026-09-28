"""Step 11: final interactive map for the Safia team (Uzbek + Russian labels).

Layers: similarity score per H3 cell, Safia branches, competitors, top-10 "similarity" candidates,
coverage list C1-C5, district borders. Clicking a candidate shows the reason in both languages.
"""
import sys

import folium
import geopandas as gpd
import h3
import pandas as pd
from branca.colormap import LinearColormap
from branca.element import Element

from config import DATA_INTERIM, DATA_PROCESSED, DATA_RAW, OUTPUTS_MAPS
from map_utils import add_basemaps, safe_text

OUT = OUTPUTS_MAPS / "safia_whitespace.html"
BRAND_COLORS = {"Cake Lab": "#27ae60", "Bon!": "#2c3e50", "Breadly": "#d35400"}
SCORE_COLORS = ["#f7f7f7", "#fde0c5", "#f9a870", "#e8603c", "#b2182b"]

LEGEND = """
<details id="legend" style="position:fixed;bottom:48px;left:10px;z-index:9999;max-width:min(360px,calc(100vw - 20px));
 background:rgba(255,255,255,.95);padding:8px 12px;border-radius:8px;box-shadow:0 1px 5px rgba(0,0,0,.3);
 font:12px/1.35 sans-serif;color:#222">
 <summary style="font-weight:bold;font-size:14px;cursor:pointer">Safia: yangi filial uchun hududlar · Зоны для новых филиалов</summary>
 <div style="margin-top:4px"><b>Ball / Балл</b> — hudud hozirgi Safia joylariga qanchalik o'xshashi
  (sotuv emas) · насколько зона похожа на текущие места Safia (не продажи).
  <div style="display:flex;align-items:center;gap:6px;margin-top:4px">0
   <span style="flex:1;height:10px;border-radius:3px;background:linear-gradient(90deg,__GRADIENT__)"></span>1</div></div>
 <div style="margin-top:6px">
  <span style="background:#c0392b;color:#fff;border-radius:9px;padding:0 6px">1–10</span> O'xshashlik: markazdagi bo'shliqlar · Сходство: пробелы в центре<br>
  <span style="background:#8e44ad;color:#fff;border-radius:9px;padding:0 6px">C1–C5</span> Qamrov: kam xizmat ko'rsatilgan janub · Охват: недообслуженный юг<br>
  <span style="color:#e74c3c">●</span> Safia &nbsp; <span style="color:#27ae60">●</span> Cake Lab &nbsp;
  <span style="color:#2c3e50">●</span> Bon! &nbsp; <span style="color:#d35400">●</span> Breadly
 </div>
 <div style="margin-top:6px;color:#555">Manbalar / Источники: safiabakery.uz, cakelab.uz, OpenStreetMap,
  Toshkent statistika (aholi 2023). Ochiq ma'lumot, 2026-09. Bon!/Breadly ro'yxati to'liq bo'lmasligi mumkin.</div>
</details>
<script>if (window.innerWidth > 700) { document.getElementById("legend").open = true; }</script>
"""


def hex_latlon(cell: str) -> list[tuple[float, float]]:
    return [tuple(p) for p in h3.cell_to_boundary(cell)]


def badge(text: str, color: str) -> folium.DivIcon:
    return folium.DivIcon(html=f'<div style="font:bold 13px sans-serif;color:#fff;background:{color};border-radius:10px;'
                               f'padding:1px 6px;transform:translate(-50%,-50%);display:inline-block;'
                               f'box-shadow:0 0 3px #000">{text}</div>')


def candidate_popup(r) -> folium.Popup:
    html = (f"<b>{safe_text(r.rank)} · {safe_text(r.district)}</b><br>"
            f"Ball / Балл: <b>{r.score:.2f}</b><br>"
            f"Eng yaqin Safia / Ближайшая Safia: {r.dist_safia_m:,.0f} m<br>"
            f"Mo'ljal / Ориентир: {safe_text(r.landmark)} ({r.landmark_dist_m:,.0f} m)<br><br>"
            f"{safe_text(r.reason)}<br>{safe_text(r.reason_ru)}<br><br>"
            f'<a href="{r.yandex}" target="_blank">Yandex</a>')
    return folium.Popup(html, max_width=300)


def main() -> int:
    feats = pd.read_parquet(DATA_PROCESSED / "hex_features.parquet")
    scores = pd.read_parquet(DATA_PROCESSED / "oof_scores.parquet")[["h3", "score"]]
    cells = feats[["h3", "district", "has_safia"]].merge(scores, on="h3")
    top = pd.read_csv(DATA_PROCESSED / "top10.csv")
    cov = pd.read_csv(DATA_PROCESSED / "coverage5.csv")
    safia = pd.read_csv(DATA_INTERIM / "safia_tashkent.csv")
    safia = safia[safia["special_site"].isna()]
    comp = pd.read_csv(DATA_INTERIM / "competitors_geo.csv")
    districts = gpd.read_file(DATA_RAW / "tashkent_districts.geojson")
    districts = districts[districts["district"] != "Yangi Toshkent"]

    m = folium.Map(location=[41.300, 69.275], zoom_start=12, tiles=None, control_scale=True)
    add_basemaps(m, names={"light": "Xarita / Карта", "streets": "Ko'chalar / Улицы",
                           "satellite": "Sun'iy yo'ldosh / Спутник"})

    cmap = LinearColormap(SCORE_COLORS, vmin=0, vmax=1, caption="Ball / Балл (0–1)")
    score_layer = folium.FeatureGroup(name="Ball (katak) / Балл (ячейка)")
    for r in cells.itertuples():
        folium.Polygon(hex_latlon(r.h3), color=None, weight=0, fill=True, fill_color=cmap(r.score),
                       fill_opacity=0.55,
                       tooltip=safe_text(f"{r.district}: {r.score:.2f}" + (" · Safia" if r.has_safia else ""))).add_to(score_layer)
    score_layer.add_to(m)  # colour scale is drawn inside the legend (a branca bar overflows on phones)

    folium.GeoJson(districts[["district", "geometry"]], name="Tumanlar / Районы",
                   style_function=lambda _: {"color": "#555", "weight": 1.2, "fill": False},
                   tooltip=folium.GeoJsonTooltip(["district"], labels=False)).add_to(m)

    s_layer = folium.FeatureGroup(name=f"Safia filiallari / Филиалы ({len(safia)})")
    for r in safia.itertuples():
        folium.CircleMarker([r.lat, r.lon], radius=4, color="#fff", weight=1, fill=True, fill_color="#e74c3c",
                            fill_opacity=1, tooltip=safe_text(f"Safia: {r.name} ({r.hours})")).add_to(s_layer)
    s_layer.add_to(m)

    c_layer = folium.FeatureGroup(name=f"Raqobatchilar / Конкуренты ({len(comp)})")
    for r in comp.itertuples():
        folium.CircleMarker([r.lat, r.lon], radius=4, color="#fff", weight=1, fill=True,
                            fill_color=BRAND_COLORS[r.brand], fill_opacity=1,
                            tooltip=safe_text(f"{r.brand}: {r.name}")).add_to(c_layer)
    c_layer.add_to(m)

    for frame, name, line, color in [(top, "Top-10: o'xshashlik / сходство", "#c0392b", "#c0392b"),
                                     (cov, "C1–C5: qamrov / охват", "#8e44ad", "#8e44ad")]:
        layer = folium.FeatureGroup(name=name)
        for r in frame.itertuples():
            folium.Polygon(hex_latlon(r.h3), color=line, weight=3, fill=False).add_to(layer)
            folium.Marker([r.lat, r.lon], icon=badge(str(r.rank), color), popup=candidate_popup(r),
                          tooltip=safe_text(f"{r.rank} · {r.district}")).add_to(layer)
        layer.add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)
    b = districts.total_bounds  # minx, miny, maxx, maxy
    m.fit_bounds([[b[1], b[0]], [b[3], b[2]]])
    m.get_root().html.add_child(Element(LEGEND.replace("__GRADIENT__", ",".join(SCORE_COLORS))))
    m.get_root().header.add_child(Element("<title>Safia · white space</title>"))
    m.save(OUT)
    print(f"Saved {OUT.relative_to(OUT.parents[2])} ({OUT.stat().st_size / 1e6:.2f} MB)")
    print(f"Cells: {len(cells)}, Safia: {len(safia)}, competitors: {len(comp)}, top-10: {len(top)}, coverage: {len(cov)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
