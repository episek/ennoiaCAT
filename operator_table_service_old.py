#!/usr/bin/env python3
# operator_table_service.py
"""
Build a carrier/operator frequency table JSON for a given location.

Usage:
  python3 operator_table_service.py --location "Dublin, Ireland" --out_dir /tmp/ennoia_tables

Output:
  /tmp/ennoia_tables/operator_table_dublin_ireland.json

Features:
- Normalizes US states (e.g., "Dallas, TX" -> "United States")
- Aliases UK nations (England/Scotland/Wales/NI -> "United Kingdom")
- Country providers: Ireland, United Kingdom, United States, France, Germany, Spain, Italy
- Optional augmentation:
    * GSMA (set GSMA_API_KEY)
    * ComReg (set COMREG_DATASET_URL) – Ireland only
- Prints absolute output path to stdout on success
- Writes warnings/errors to stderr; non-zero exit on hard failure
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
from typing import List, Dict, Tuple, Optional

# --------------------------- Helpers ---------------------------

US_STATES = {
    "al","ak","az","ar","ca","co","ct","de","fl","ga","hi","id","il","in","ia","ks",
    "ky","la","me","md","ma","mi","mn","ms","mo","mt","ne","nv","nh","nj","nm","ny",
    "nc","nd","oh","ok","or","pa","ri","sc","sd","tn","tx","ut","vt","va","wa","wv","wi","wy",
}
US_STATE_NAMES = {
    "alabama","alaska","arizona","arkansas","california","colorado","connecticut","delaware",
    "florida","georgia","hawaii","idaho","illinois","indiana","iowa","kansas","kentucky",
    "louisiana","maine","maryland","massachusetts","michigan","minnesota","mississippi","missouri",
    "montana","nebraska","nevada","new hampshire","new jersey","new mexico","new york",
    "north carolina","north dakota","ohio","oklahoma","oregon","pennsylvania","rhode island",
    "south carolina","south dakota","tennessee","texas","utah","vermont","virginia","washington",
    "west virginia","wisconsin","wyoming",
}
UK_NATIONS = {"england","scotland","wales","northern ireland"}

def normalize_country_token(tok: str) -> str:
    k = (tok or "").strip().lower()
    if k in US_STATES or k in US_STATE_NAMES:
        return "United States"
    if k in UK_NATIONS:
        return "United Kingdom"
    return tok.strip()

def norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "").strip().lower()).strip("_")

def out_filename(city: str, country: str) -> str:
    return f"operator_table_{norm_name(city)}_{norm_name(country)}.json"

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def validate_rows(rows: List[Dict]) -> None:
    req = {"Band ", "3GPP Band", "Uplink Frequency (MHz)", "Downlink Frequency (MHz)", "Bandwidth", "Technology", "Operators"}
    if not isinstance(rows, list) or not rows:
        raise ValueError("empty rows")
    for i, r in enumerate(rows):
        missing = req - set(r.keys())
        if missing:
            raise ValueError(f"row {i} missing fields: {sorted(missing)}")

# ---------------------- Generic row builders ----------------------

def _rng(a: float, b: float) -> str:
    a, b = sorted((float(a), float(b)))
    def f(x):
        return str(int(x)) if abs(x - int(x)) < 1e-9 else f"{x}"
    return f"{f(a)} - {f(b)}"

def _row(label, gpp, ul1, ul2, dl1, dl2, bw, tech, ops):
    return {
        "Band ": label,
        "3GPP Band": gpp,
        "Uplink Frequency (MHz)": _rng(ul1, ul2),
        "Downlink Frequency (MHz)": _rng(dl1, dl2),
        "Bandwidth": bw,
        "Technology": tech,
        "Operators": ops,
    }

# ---------------------- Country providers (curated) ----------------------
# These are curated, operator-sliced summaries typical for each market.
# They are conservative and meant to be a useful baseline even without live fetch.

def build_ie(city: str) -> List[Dict]:
    rows: List[Dict] = []
    rows += [_row("700 MHz","Band 28 / n28",703,718,758,773,"10–15 MHz","LTE, 5G NR","Vodafone"),
             _row("700 MHz","Band 28 / n28",718,733,773,788,"10–15 MHz","LTE, 5G NR","Three"),
             _row("700 MHz","Band 28 / n28",733,748,788,803,"10–15 MHz","LTE, 5G NR","eir")]
    rows += [_row("800 MHz","Band 20 / n20",842,852,801,811,"5–10 MHz","LTE / NR DSS","Vodafone"),
             _row("800 MHz","Band 20 / n20",852,862,811,821,"5–10 MHz","LTE / NR DSS","Three"),
             _row("800 MHz","Band 20 / n20",832,842,791,801,"5–10 MHz","LTE / NR DSS","eir")]
    rows += [_row("900 MHz","Band 8 / n8",890,900,935,945,"5–10 MHz","LTE","Vodafone"),
             _row("900 MHz","Band 8 / n8",900,910,945,955,"5–10 MHz","LTE","Three"),
             _row("900 MHz","Band 8 / n8",910,915,955,960,"5 MHz","LTE","eir")]
    rows += [_row("1800 MHz","Band 3 / n3",1710,1735,1805,1830,"25 MHz","LTE / NR DSS","Vodafone"),
             _row("1800 MHz","Band 3 / n3",1735,1760,1830,1855,"25 MHz","LTE / NR DSS","Three"),
             _row("1800 MHz","Band 3 / n3",1760,1785,1855,1880,"25 MHz","LTE / NR DSS","eir")]
    rows += [_row("2100 MHz","Band 1 / n1",1925,1945,2115,2135,"20 MHz","LTE / NR DSS","Vodafone"),
             _row("2100 MHz","Band 1 / n1",1945,1965,2135,2155,"20 MHz","LTE / NR DSS","Three"),
             _row("2100 MHz","Band 1 / n1",1965,1980,2155,2170,"15 MHz","LTE / NR DSS","eir")]
    rows += [_row("2600 MHz","Band 7",2500,2520,2620,2640,"20 MHz","LTE","Three"),
             _row("2600 MHz","Band 7",2520,2540,2640,2660,"20 MHz","LTE","Vodafone")]
    rows += [_row("3.6 GHz","n78 (3.4–3.8 GHz)",3460,3560,3460,3560,"100 MHz","5G NR (TDD)","Vodafone"),
             _row("3.6 GHz","n78 (3.4–3.8 GHz)",3480,3660,3480,3660,"180 MHz","5G NR (TDD)","Three"),
             _row("3.6 GHz","n78 (3.4–3.8 GHz)",3560,3660,3560,3660,"100 MHz","5G NR (TDD)","eir")]
    return rows

def build_gb(city: str) -> List[Dict]:
    rows: List[Dict] = []
    rows += [_row("700 MHz","Band 28 / n28",703,733,758,788,"30 MHz","LTE, 5G NR","EE, Vodafone, O2, Three")]
    rows += [_row("800 MHz","Band 20 / n20",832,862,791,821,"30 MHz","LTE","EE, Vodafone, O2, Three")]
    rows += [_row("1800 MHz","Band 3 / n3",1710,1785,1805,1880,"75 MHz","LTE/NR DSS","EE, Vodafone, O2, Three")]
    rows += [_row("2100 MHz","Band 1 / n1",1920,1980,2110,2170,"60 MHz","LTE/NR DSS","EE, Vodafone, O2, Three")]
    rows += [_row("2600 MHz","Band 7/38",2500,2570,2620,2690,"FDD/TDD","LTE/NR","EE, Vodafone, O2, Three")]
    rows += [_row("3.4–3.8 GHz","n77/n78",3400,3800,3400,3800,"40–100 MHz","5G NR (TDD)","EE, Vodafone, O2, Three")]
    return rows

def build_us(city: str) -> List[Dict]:
    rows: List[Dict] = []
    rows += [_row("700 MHz","Band 12/13/14/17/29 / n12/n14/n29",699,716,729,746,"Varies","LTE/NR","AT&T, Verizon, others")]
    rows += [_row("850 MHz","Band 5 / n5",824,849,869,894,"Varies","LTE/NR DSS","AT&T, Verizon, UScellular")]
    rows += [_row("PCS 1900","Band 2 / n2",1850,1915,1930,1995,"Varies","LTE/NR","AT&T, T-Mobile, Verizon")]
    rows += [_row("AWS-1/3","Band 4/66 / n66",1710,1780,2110,2200,"Varies","LTE/NR","AT&T, T-Mobile, Verizon, Dish")]
    rows += [_row("CBRS","n48 (3550–3700)",3550,3700,3550,3700,"10–40 MHz","LTE/NR (TDD)","PAL+GAA")]
    rows += [_row("C-band","n77 (3700–3980)",3700,3980,3700,3980,"up to 140 MHz","5G NR (TDD)","AT&T, Verizon")]
    rows += [_row("mmWave 24–39 GHz","n258/n260/n261",24250,39950,24250,39950,"Varies","5G NR (TDD)","Various")]
    return rows

# -------- New: France (ARCEP-like summary) --------
def build_fr(city: str) -> List[Dict]:
    rows: List[Dict] = []
    # Main operators: Orange, SFR, Bouygues Telecom, Free Mobile
    rows += [_row("700 MHz","Band 28 / n28",703,733,758,788,"2x30 MHz","LTE/NR DSS","All (various allocations)")]
    rows += [_row("800 MHz","Band 20 / n20",832,862,791,821,"2x30 MHz","LTE/NR DSS","All")]
    rows += [_row("900 MHz","Band 8 / n8",890,915,935,960,"2x25 MHz","LTE/NR DSS","All")]
    rows += [_row("1800 MHz","Band 3 / n3",1710,1785,1805,1880,"2x75 MHz","LTE/NR DSS","All")]
    rows += [_row("2100 MHz","Band 1 / n1",1920,1980,2110,2170,"2x60 MHz","LTE/NR DSS","All")]
    rows += [_row("2600 MHz (FDD)","Band 7",2500,2570,2620,2690,"2x70 MHz","LTE/NR","All")]
    rows += [_row("2.6 GHz (TDD)","Band 38",2570,2620,2570,2620,"50 MHz","LTE TDD/NR","All (smaller allocations)")]
    rows += [_row("3.4–3.8 GHz","n77/n78",3400,3800,3400,3800,"80–100 MHz","5G NR (TDD)","All (assigned blocks)")]
    return rows

# -------- New: Germany (BNetzA-like summary) --------
def build_de(city: str) -> List[Dict]:
    rows: List[Dict] = []
    # Main operators: Telekom, Vodafone, Telefónica (o2), plus 3.7–3.8 GHz local licenses
    rows += [_row("700 MHz","Band 28 / n28",703,733,758,788,"2x30 MHz","LTE/NR DSS","All")]
    rows += [_row("800 MHz","Band 20 / n20",832,862,791,821,"2x30 MHz","LTE/NR DSS","All")]
    rows += [_row("900 MHz","Band 8 / n8",890,915,935,960,"2x25 MHz","LTE/NR DSS","All")]
    rows += [_row("1500 MHz (SDL)","Band 32",1452,1496,1452,1496,"Downlink 40–50 MHz","LTE SDL","Telekom/Vodafone")]
    rows += [_row("1800 MHz","Band 3 / n3",1710,1785,1805,1880,"2x75 MHz","LTE/NR DSS","All")]
    rows += [_row("2100 MHz","Band 1 / n1",1920,1980,2110,2170,"2x60 MHz","LTE/NR DSS","All")]
    rows += [_row("2600 MHz (FDD/TDD)","Band 7/38",2500,2570,2620,2690,"FDD+TDD","LTE/NR","All")]
    rows += [_row("3.4–3.8 GHz","n77/n78",3400,3800,3400,3800,"80–100 MHz","5G NR (TDD)","All (+ local 3.7–3.8 GHz)")]
    rows += [_row("26 GHz","n258",24250,27500,24250,27500,"Varies","5G NR (TDD)","Urban trials/assignments")]
    return rows

# -------- New: Spain (CNMC-like summary) --------
def build_es(city: str) -> List[Dict]:
    rows: List[Dict] = []
    # Main operators: Telefónica (Movistar), Vodafone, Orange, MásMóvil/Yoigo
    rows += [_row("700 MHz","Band 28 / n28",703,733,758,788,"2x30 MHz","LTE/NR DSS","All")]
    rows += [_row("800 MHz","Band 20 / n20",832,862,791,821,"2x30 MHz","LTE/NR DSS","All")]
    rows += [_row("900 MHz","Band 8 / n8",890,915,935,960,"2x25 MHz","LTE/NR DSS","All")]
    rows += [_row("1800 MHz","Band 3 / n3",1710,1785,1805,1880,"2x75 MHz","LTE/NR DSS","All")]
    rows += [_row("2100 MHz","Band 1 / n1",1920,1980,2110,2170,"2x60 MHz","LTE/NR DSS","All")]
    rows += [_row("2600 MHz (FDD/TDD)","Band 7/38",2500,2570,2620,2690,"FDD+TDD","LTE/NR","All")]
    rows += [_row("3.5 GHz","n78 (3.4–3.8 GHz)",3400,3800,3400,3800,"80–100 MHz","5G NR (TDD)","All (assigned blocks)")]
    rows += [_row("26 GHz","n258",24250,27500,24250,27500,"Varies","5G NR (TDD)","Trials/early assignments")]
    return rows

# -------- New: Italy (AGCOM-like summary) --------
def build_it(city: str) -> List[Dict]:
    rows: List[Dict] = []
    # Main operators: TIM, Vodafone, Wind Tre, Iliad
    rows += [_row("700 MHz","Band 28 / n28",703,733,758,788,"2x30 MHz","LTE/NR DSS","All")]
    rows += [_row("800 MHz","Band 20 / n20",832,862,791,821,"2x30 MHz","LTE/NR DSS","All")]
    rows += [_row("900 MHz","Band 8 / n8",890,915,935,960,"2x25 MHz","LTE/NR DSS","All")]
    rows += [_row("1800 MHz","Band 3 / n3",1710,1785,1805,1880,"2x75 MHz","LTE/NR DSS","All")]
    rows += [_row("2100 MHz","Band 1 / n1",1920,1980,2110,2170,"2x60 MHz","LTE/NR DSS","All")]
    rows += [_row("2600 MHz (FDD/TDD)","Band 7/38",2500,2570,2620,2690,"FDD+TDD","LTE/NR","All")]
    rows += [_row("3.6 GHz","n78 (3.4–3.8 GHz)",3400,3800,3400,3800,"80–100 MHz","5G NR (TDD)","All (assigned blocks)")]
    rows += [_row("26 GHz","n258",24250,27500,24250,27500,"Varies","5G NR (TDD)","Urban/enterprise allocations")]
    return rows

# ---------------------- Optional live augmentation ----------------------

def augment_with_gsma(rows: List[Dict], country: str) -> List[Dict]:
    """
    Try to enrich curated rows with GSMA data if GSMA_API_KEY is set.
    This is a best-effort additive overlay; on any error it just returns the original rows.
    """
    api_key = os.getenv("GSMA_API_KEY")
    if not api_key:
        return rows
    try:
        import requests
        # Pseudo-endpoint placeholder — replace with your actual GSMA endpoint if available.
        # Example idea: GET https://api.gsma.com/spectrum/v1/country?name={country}
        url = f"https://api.gsma.com/spectrum/v1/country?name={requests.utils.quote(country)}"
        headers = {"Authorization": f"Bearer {api_key}"}
        r = requests.get(url, headers=headers, timeout=8)
        if r.ok:
            data = r.json()
            # Expect a list of band items with fields we can map conservatively.
            extra: List[Dict] = []
            for item in data.get("bands", []):
                try:
                    label = item.get("label") or item.get("range") or "Unknown"
                    gpp = item.get("gpp") or item.get("band") or "Unknown"
                    ul = item.get("uplink_mhz") or [0, 0]
                    dl = item.get("downlink_mhz") or [0, 0]
                    bw = item.get("bandwidth") or item.get("bw") or "Varies"
                    tech = item.get("tech") or "LTE/NR"
                    ops = ", ".join(item.get("operators", [])) or "Various"
                    extra.append(_row(label, gpp, ul[0], ul[-1], dl[0], dl[-1], bw, tech, ops))
                except Exception:
                    continue
            # Merge: avoid crude duplicates by (3GPP Band + Operators + Uplink range)
            def sig(rw: Dict) -> Tuple[str,str,str]:
                return (rw.get("3GPP Band",""), rw.get("Operators",""), rw.get("Uplink Frequency (MHz)",""))
            existing = {sig(x) for x in rows}
            merged = rows[:]
            for e in extra:
                if sig(e) not in existing:
                    merged.append(e)
            return merged
        else:
            print(f"gsma_warn: HTTP {r.status_code}", file=sys.stderr)
            return rows
    except Exception as e:
        print(f"gsma_error: {e}", file=sys.stderr)
        return rows

def augment_with_comreg(rows: List[Dict], country: str) -> List[Dict]:
    """
    Optional: augment for Ireland if COMREG_DATASET_URL provided (CSV/JSON that lists assignments).
    Any error returns original rows.
    """
    if country.strip().lower() not in ("ireland","ie"):
        return rows
    url = os.getenv("COMREG_DATASET_URL")
    if not url:
        return rows
    try:
        import pandas as pd
        if url.lower().endswith(".csv"):
            df = pd.read_csv(url)
        else:
            import requests
            r = requests.get(url, timeout=8)
            r.raise_for_status()
            # Try JSON first
            try:
                data = r.json()
                df = pd.json_normalize(data)
            except Exception:
                # Try CSV fallback if JSON failed
                import io as _io
                df = pd.read_csv(_io.StringIO(r.text))
        # You can customize the column mapping here to match your dataset
        extra = []
        for _, row in df.iterrows():
            try:
                label = str(row.get("Band") or row.get("BandLabel") or row.get("Label") or "Unknown")
                gpp  = str(row.get("GPP") or row.get("3GPP") or row.get("Band3GPP") or "Unknown")
                ul1, ul2 = float(row.get("UL_Low", 0)), float(row.get("UL_High", 0))
                dl1, dl2 = float(row.get("DL_Low", 0)), float(row.get("DL_High", 0))
                bw   = str(row.get("BW", "Varies"))
                tech = str(row.get("Tech", "LTE/NR"))
                ops  = str(row.get("Operator", "Various"))
                extra.append(_row(label, gpp, ul1, ul2, dl1, dl2, bw, tech, ops))
            except Exception:
                continue
        # Merge with simple signature
        def sig(rw: Dict) -> Tuple[str,str,str]:
            return (rw.get("3GPP Band",""), rw.get("Operators",""), rw.get("Uplink Frequency (MHz)",""))
        existing = {sig(x) for x in rows}
        merged = rows[:]
        for e in extra:
            if sig(e) not in existing:
                merged.append(e)
        return merged
    except Exception as e:
        print(f"comreg_error: {e}", file=sys.stderr)
        return rows

# --------------------------- Core build ---------------------------

PROVIDERS = {
    "ireland": build_ie,
    "united kingdom": build_gb,
    "uk": build_gb,
    "great britain": build_gb,
    "united states": build_us,
    "usa": build_us,
    "us": build_us,
    "france": build_fr,
    "germany": build_de,
    "deutschland": build_de,
    "spain": build_es,
    "italy": build_it,
}

def parse_location(location: str) -> Tuple[str, str]:
    parts = [p.strip() for p in (location or "").split(",") if p.strip()]
    if not parts:
        raise ValueError("Location is empty")
    city = parts[0]
    raw_country = parts[-1] if len(parts) > 1 else "Unknown"
    country = normalize_country_token(raw_country)
    return city, country

def build_rows_for_location(location: str) -> List[Dict]:
    city, country = parse_location(location)
    key = country.strip().lower()
    fn = PROVIDERS.get(key)
    if fn is None:
        return []
    rows = fn(city)
    # Optional live augmentation layers
    rows = augment_with_gsma(rows, country)
    rows = augment_with_comreg(rows, country)
    return rows

# --------------------------- Main CLI ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--location", required=True, help='Location string, e.g. "Dallas, TX" or "Dublin, Ireland"')
    ap.add_argument("--out_dir", required=True, help="Output directory for JSON file")
    args = ap.parse_args()

    # Build rows
    try:
        rows = build_rows_for_location(args.location)
    except Exception as e:
        print(f"parse_or_build_error: {e}", file=sys.stderr)
        sys.exit(2)

    # Stub if empty (unsupported country)
    if not rows:
        print(f"no_data_for: {args.location}; writing stub (please add a provider)", file=sys.stderr)
        rows = [{
            "Band ": "Unknown",
            "3GPP Band": "Unknown",
            "Uplink Frequency (MHz)": "0 - 0",
            "Downlink Frequency (MHz)": "0 - 0",
            "Bandwidth": "Unknown",
            "Technology": "Unknown",
            "Operators": "Unknown"
        }]

    # Validate and write
    try:
        validate_rows(rows)
    except Exception as e:
        print(f"validation_error: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        out_dir = os.path.abspath(args.out_dir)
        ensure_dir(out_dir)
        city, country = parse_location(args.location)
        fname = out_filename(city, country)
        out_path = os.path.join(out_dir, fname)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(out_path)  # stdout for caller
    except Exception as e:
        print(f"write_error: {e}", file=sys.stderr)
        sys.exit(3)

if __name__ == "__main__":
    main()
