"""Shared helper for Overpass API requests (used by steps 04 and 06)."""
import time

import requests

from config import OVERPASS_URLS, USER_AGENT


def overpass(query: str, rounds: int = 3, wait_s: int = 20) -> dict:
    """Try every endpoint in order; repeat up to `rounds` times with a growing pause."""
    for attempt in range(1, rounds + 1):
        for url in OVERPASS_URLS:
            try:
                resp = requests.post(url, data={"data": query}, headers={"User-Agent": USER_AGENT}, timeout=120)
                if resp.ok:
                    return resp.json()
                print(f"  round {attempt} {url}: HTTP {resp.status_code}")
            except requests.RequestException as exc:
                print(f"  round {attempt} {url}: {type(exc).__name__}")
        if attempt < rounds:
            time.sleep(wait_s * attempt)
    raise RuntimeError("all Overpass endpoints failed")
