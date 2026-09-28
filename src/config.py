"""Shared paths and constants used by every step."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS_MAPS = ROOT / "outputs" / "maps"
OUTPUTS_FIGURES = ROOT / "outputs" / "figures"
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"

# Coordinate reference systems
CRS_WGS84 = "EPSG:4326"    # storage
CRS_METRIC = "EPSG:32642"  # UTM 42N: distances and areas in Tashkent

# H3 grid
H3_RES = 8        # modelling cells: ~0.88 km2 at Tashkent's latitude (~500 m centre to edge)
H3_CV_RES = 6     # parent cells used as spatial CV blocks

# Web requests
USER_AGENT = "safia-location-study/0.1 (research project)"
REQUEST_DELAY_S = 1.0  # at most one request per second

SAFIA_LOCATIONS_URL = "https://safiabakery.uz/en/locations"

# OSM relations: Tashkent city (admin_level 4, UZ-TK) and its districts (admin_level 6)
TASHKENT_RELATION = 2216724
TASHKENT_DISTRICTS = {
    1751444: "Yashnobod",
    2434059: "Uchtepa",
    2439529: "Shayxontohur",
    2441651: "Olmazor",
    2441810: "Chilonzor",
    2443769: "Yakkasaroy",
    2447546: "Sergeli",
    2447560: "Bektemir",
    2447634: "Mirobod",
    2448072: "Yunusobod",
    5620904: "Mirzo Ulug'bek",
    12030887: "Yangihayot",
    17389398: "Yangi Toshkent",  # new district, largely under construction
}
OSM_CACHE = DATA_RAW / "cache" / "osmnx"

# Overpass endpoints, tried in order (the main server often returns 504)
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
TASHKENT_BBOX = (41.12, 69.05, 41.48, 69.60)  # south, west, north, east (covers the city polygon)

# GHSL GHS-POP, epoch 2025, release R2023A, 3 arcsec (~90 m), WGS84. Tile R5_C25 covers Tashkent.
# Licence CC BY 4.0. Compared in step 06b and rejected like WorldPop (see README).
GHSL_URL = ("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/"
            "GHS_POP_E2025_GLOBE_R2023A_4326_3ss/V1-0/tiles/GHS_POP_E2025_GLOBE_R2023A_4326_3ss_V1_0_R5_C25.zip")

# WorldPop Global2 R2025A, constrained, 100 m. Rejected for Tashkent; kept only for comparison.
# Licence CC BY 4.0, DOI 10.5258/SOTON/WP00839
WORLDPOP_YEAR = 2025
WORLDPOP_URL = ("https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/"
                f"{WORLDPOP_YEAR}/UZB/v1/100m/constrained/uzb_pop_{WORLDPOP_YEAR}_CN_100m_R2025A_v1.tif")
