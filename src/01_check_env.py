"""Step 01: verify that every required library imports and that project folders exist."""
import importlib
import sys

from config import (DATA_INTERIM, DATA_PROCESSED, DATA_RAW, OUTPUTS_FIGURES,
                    OUTPUTS_MAPS, REPORTS)

# import name -> pip package name
LIBRARIES = {
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "geopandas": "geopandas",
    "shapely": "shapely",
    "pyproj": "pyproj",
    "requests": "requests",
    "bs4": "beautifulsoup4",
    "lxml": "lxml",
    "geopy": "geopy",
    "osmnx": "osmnx",
    "h3": "h3",
    "sklearn": "scikit-learn",
    "lightgbm": "lightgbm",
    "shap": "shap",
    "matplotlib": "matplotlib",
    "folium": "folium",
}


def main() -> int:
    print(f"Python {sys.version.split()[0]}")
    failed = []
    for module, package in LIBRARIES.items():
        try:
            mod = importlib.import_module(module)
            print(f"  OK   {package:<15} {getattr(mod, '__version__', '?')}")
        except Exception as exc:
            failed.append(package)
            print(f"  FAIL {package:<15} {exc}")

    # Functional smoke tests for the libraries the later steps depend on most
    import h3
    from pyproj import Transformer
    cell = h3.latlng_to_cell(41.3111, 69.2797, 8)  # Tashkent centre
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:32642", always_xy=True).transform(69.2797, 41.3111)
    print(f"\nH3 cell at Tashkent centre: {cell}")
    print(f"UTM 42N coordinates: x={x:.0f}, y={y:.0f}")

    missing_dirs = [d for d in (DATA_RAW, DATA_INTERIM, DATA_PROCESSED, OUTPUTS_MAPS, OUTPUTS_FIGURES, REPORTS)
                    if not d.is_dir()]
    for d in missing_dirs:
        print(f"  MISSING DIR {d}")

    ok = not failed and not missing_dirs
    print("\nRESULT:", "all checks passed" if ok else f"failed: {failed + [str(d) for d in missing_dirs]}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
