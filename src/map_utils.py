"""Shared helpers for folium maps."""
import html

import folium

# Esri basemaps. OpenStreetMap's volunteer tile servers return an "Access blocked" image when the
# request has no Referer header, which is what happens when a saved HTML file is opened from disk
# (file://). The maps are meant to be downloaded and opened that way, so OSM tiles are not used.
_ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/{}/MapServer/tile/{{z}}/{{y}}/{{x}}"
BASEMAPS = {
    "light": (_ESRI.format("Canvas/World_Light_Gray_Base"), "Esri, HERE, Garmin, © OpenStreetMap contributors"),
    "light_labels": (_ESRI.format("Canvas/World_Light_Gray_Reference"), "Esri"),
    "streets": (_ESRI.format("World_Street_Map"), "Esri, HERE, Garmin, USGS, © OpenStreetMap contributors"),
    "satellite": (_ESRI.format("World_Imagery"), "Esri, Maxar, Earthstar Geographics"),
}


def add_basemaps(m: folium.Map, default: str = "light", names: dict | None = None) -> None:
    """Add light-grey (with place labels), street and satellite basemaps; `default` is shown first."""
    names = names or {"light": "Light", "streets": "Streets", "satellite": "Satellite"}
    for key in ("light", "streets", "satellite"):
        url, attr = BASEMAPS[key]
        folium.TileLayer(url, attr=attr, name=names[key], show=(key == default), max_zoom=19).add_to(m)
    url, attr = BASEMAPS["light_labels"]
    folium.TileLayer(url, attr=attr, name="Labels", overlay=True, control=False, max_zoom=19).add_to(m)


def safe_text(text: str) -> str:
    """Escape text for folium tooltips/popups.

    Folium wraps tooltips in JS template literals, so a backtick (common in Uzbek names,
    e.g. "Ulug`bek") breaks the whole page. Use this for every OSM-sourced string.
    """
    return html.escape(str(text)).replace("`", "'")
