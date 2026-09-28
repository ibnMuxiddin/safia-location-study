"""Step 02: extract all Safia branches from the public locations page.

The page embeds schema.org JSON-LD (`@type: Bakery`) with name, address, coordinates
and opening hours for every branch. The HTML is cached in data/raw/ and downloaded
only when the cache is missing or --refresh is passed.
"""
import argparse
import hashlib
import json
import sys

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config import DATA_INTERIM, DATA_RAW, SAFIA_LOCATIONS_URL, USER_AGENT

RAW_HTML = DATA_RAW / "safia_locations.html"
OUT_CSV = DATA_INTERIM / "safia_branches.csv"


def load_html(refresh: bool) -> str:
    if RAW_HTML.exists() and not refresh:
        print(f"Using cached page: {RAW_HTML.name}")
        return RAW_HTML.read_text(encoding="utf-8")
    print(f"Downloading {SAFIA_LOCATIONS_URL}")
    resp = requests.get(SAFIA_LOCATIONS_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    RAW_HTML.write_text(resp.text, encoding="utf-8")
    return resp.text


def make_id(name: str, address: str) -> str:
    """Stable id that survives re-ordering of the page."""
    return "sf_" + hashlib.md5(f"{name.strip()}|{address}".encode("utf-8")).hexdigest()[:8]


def parse_branches(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")
    items = []
    for block in soup.find_all("script", type="application/ld+json"):
        data = json.loads(block.string)
        items.extend(data if isinstance(data, list) else [data])

    rows = []
    for it in items:
        if it.get("@type") != "Bakery":
            continue
        address = it["address"]["streetAddress"].strip()
        hours = it.get("openingHours", "").replace("Mo-Su ", "").strip()
        rows.append({
            "branch_id": make_id(it["name"], address),
            # The site puts an internal UUID in the "telephone" field. It is missing for some
            # branches and shared by others, so it is kept for reference only.
            "site_uuid": it.get("telephone"),
            "name": it["name"].strip(),
            "city": address.split(",")[0].strip(),
            "address": address,
            "hours": hours,
            "is_24h": hours in {"00:00-23:59", "00:00-24:00"},
            "is_s_express": it["name"].lower().startswith("s express"),
            "lat": it.get("geo", {}).get("latitude"),
            "lon": it.get("geo", {}).get("longitude"),
        })
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> None:
    print(f"\nBranches: {len(df)}")
    print(f"Missing coordinates: {df['lat'].isna().sum()}")
    print(f"Duplicate branch_id: {df['branch_id'].duplicated().sum()}")
    print(f"site_uuid missing: {df['site_uuid'].isna().sum()}, shared: {df['site_uuid'].dropna().duplicated(keep=False).sum()}")
    print(f"Duplicate coordinates: {df.duplicated(['lat', 'lon']).sum()}")
    outside = ~df["lat"].between(37, 56) | ~df["lon"].between(46, 88)  # Uzbekistan + Kazakhstan box
    print(f"Coordinates outside UZ/KZ box: {outside.sum()}")
    print(f"24/7 branches: {df['is_24h'].sum()}  |  S express: {df['is_s_express'].sum()}")
    no_city = df["city"].str.match(r"(улица|проспект|массив|переулок|проезд)", case=False)
    print(f"Addresses without a city prefix: {no_city.sum()} -> {df.loc[no_city, 'name'].tolist()}")
    late = df["hours"].str.extract(r"-(\d{2}):", expand=False).astype(int) > 24
    print(f"Closing time after 24:00 (site data issue): {df.loc[late, ['name', 'hours']].values.tolist()}")
    print("\nBranches by city (parsed from address text, not reliable for filtering):")
    print(df["city"].value_counts().to_string())
    print("\n5 random rows for manual check against the site:")
    print(df.sample(5, random_state=42)[["name", "address", "hours", "lat", "lon"]].to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="download the page again")
    args = parser.parse_args()

    if not RAW_HTML.exists() and OUT_CSV.exists() and not args.refresh:
        # Public copy of the repo: the site's HTML is not redistributed, the extracted table is
        print(f"No {RAW_HTML.name}; using the committed {OUT_CSV.name} (pass --refresh to download the page)")
        report(pd.read_csv(OUT_CSV))
        return 0
    df = parse_branches(load_html(args.refresh))
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"Saved {OUT_CSV.relative_to(OUT_CSV.parents[2])}")
    report(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
