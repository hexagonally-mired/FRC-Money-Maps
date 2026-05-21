"""
FRC Economic Indicator Pipeline
================================
Pulls:
  - Team locations from your local season_2026.json (TBA format)
  - EPA / performance data from Statbotics API
  - Economic indicators from Census ACS 5-Year API

Outputs:
  - frc_teams_with_economics.csv  (one row per team, US only)
  - frc_unmatched_teams.csv       (teams we couldn't geocode)

Dependencies:
  pip install zipcodes addfips requests pandas

One-time setup:
  Download the Census county centroid file (used for lat/lon -> county lookup):
  https://www2.census.gov/geo/docs/reference/cenpop2020/county/CenPop2020_Mean_CO.txt
  Save it as CenPop2020_Mean_CO.txt in the same folder as this script.
"""

import json
import math
import time
import requests
import pandas as pd
import zipcodes
import addfips

# ─────────────────────────────────────────────
# CONFIGURATION — edit these
# ─────────────────────────────────────────────

CENSUS_API_KEY = "8a19b110ab12960b20c0dcaf312e715d6e50966d"

# Which years of Statbotics data to average over
STATBOTICS_YEARS = [2023, 2024, 2025]

TEAMS_JSON_PATH = "season_2026.json"
COUNTY_CENTROIDS_PATH = "CenPop2020_Mean_CO.txt"

# ─────────────────────────────────────────────
# GEOCODING HELPERS
# ─────────────────────────────────────────────

af = addfips.AddFIPS()

def zip_to_fips(postal_code: str) -> str | None:
    """Zip code -> 5-digit county FIPS via zipcodes + addfips."""
    try:
        results = zipcodes.matching(str(postal_code).strip().zfill(5))
        if not results:
            return None
        county = results[0]["county"]   # e.g. "Oakland County"
        state  = results[0]["state"]    # e.g. "MI"
        fips = af.get_county_fips(county, state=state)
        return str(fips).zfill(5) if fips else None
    except Exception:
        return None


def build_centroid_index(path: str):
    """
    Load Census county population centroids into a list for nearest-neighbor lookup.
    File: https://www2.census.gov/geo/docs/reference/cenpop2020/county/CenPop2020_Mean_CO.txt
    """
    df = pd.read_csv(path, dtype={"STATEFP": str, "COUNTYFP": str})
    df["county_fips"] = df["STATEFP"].str.zfill(2) + df["COUNTYFP"].str.zfill(3)
    # Convert to list of (lat, lon, fips) for fast iteration
    return list(zip(df["LATITUDE"], df["LONGITUDE"], df["county_fips"]))


def haversine(lat1, lon1, lat2, lon2) -> float:
    """Distance in km between two lat/lon points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def latlon_to_fips(lat: float, lon: float, centroids: list) -> str | None:
    """Find the county whose centroid is nearest to (lat, lon)."""
    best_fips = None
    best_dist = float("inf")
    for clat, clon, fips in centroids:
        d = haversine(lat, lon, clat, clon)
        if d < best_dist:
            best_dist = d
            best_fips = fips
    return best_fips


def get_county_fips(row, centroids: list) -> str | None:
    """
    Try zip code first (more precise), fall back to nearest centroid.
    Almost every team has lat/lon, so centroid is the main fallback.
    """
    # Try zip first — it's slightly more accurate for urban boundary cases
    if pd.notna(row.get("postal_code")) and str(row.get("postal_code", "")).strip():
        fips = zip_to_fips(str(row["postal_code"]))
        if fips:
            return fips

    # Fall back to nearest county centroid
    if pd.notna(row.get("lat")) and pd.notna(row.get("lng")):
        return latlon_to_fips(float(row["lat"]), float(row["lng"]), centroids)

    return None


# ─────────────────────────────────────────────
# STEP 1: Load team data
# ─────────────────────────────────────────────

print("Loading team data...")

with open(TEAMS_JSON_PATH) as f:
    raw = json.load(f)

teams = pd.DataFrame(list(raw["teams"].values()))
us_teams = teams[teams["country"] == "USA"].copy()
intl_teams = teams[teams["country"] != "USA"].copy()

print(f"  Total: {len(teams)} | US: {len(us_teams)} | Intl: {len(intl_teams)}")

# ─────────────────────────────────────────────
# STEP 2: Geocode to county FIPS
# ─────────────────────────────────────────────

print("\nGeocoding teams to counties...")
print("  Loading county centroid index...")
centroids = build_centroid_index(COUNTY_CENTROIDS_PATH)
print(f"  Loaded {len(centroids)} county centroids")

us_teams["county_fips"] = us_teams.apply(
    lambda row: get_county_fips(row, centroids), axis=1
)

matched   = us_teams["county_fips"].notna().sum()
unmatched = us_teams["county_fips"].isna().sum()
print(f"  Matched: {matched} | Unmatched: {unmatched}")

# Save unmatched for inspection
us_teams[us_teams["county_fips"].isna()][
    ["team_number", "nickname", "city", "state_prov", "postal_code", "lat", "lng"]
].to_csv("frc_unmatched_teams.csv", index=False)

us_teams = us_teams[us_teams["county_fips"].notna()].copy()

# ─────────────────────────────────────────────
# STEP 3: Pull Census ACS economic data
# ─────────────────────────────────────────────

print("\nFetching Census ACS 5-Year (2023) data...")

CENSUS_VARS = {
    "B19013_001E": "median_household_income",
    "B19301_001E": "per_capita_income",
    "B19083_001E": "gini_index",
    "B23025_003E": "labor_force",
    "B23025_004E": "employed",
    "B17001_002E": "below_poverty",
    "B17001_001E": "poverty_universe",
}

BASE_URL = "https://api.census.gov/data/2023/acs/acs5"

def fetch_census_state(state_fips: str) -> pd.DataFrame:
    var_str = "NAME," + ",".join(CENSUS_VARS.keys())
    url = (
        f"{BASE_URL}?get={var_str}"
        f"&for=county:*&in=state:{state_fips}"
        f"&key={CENSUS_API_KEY}"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return pd.DataFrame(data[1:], columns=data[0])

# Pull only the states we actually have teams in
state_fips_list = us_teams["county_fips"].str[:2].dropna().unique().tolist()
print(f"  Pulling data for {len(state_fips_list)} states...")

census_frames = []
for sfips in state_fips_list:
    try:
        df = fetch_census_state(sfips)
        census_frames.append(df)
        time.sleep(0.1)
    except Exception as e:
        print(f"  Warning: state {sfips} failed — {e}")

counties = pd.concat(census_frames, ignore_index=True)
counties = counties.rename(columns=CENSUS_VARS)
counties["county_fips"] = counties["state"].str.zfill(2) + counties["county"].str.zfill(3)

# Cast numerics, replace Census sentinel (-666666666) with NaN
numeric_cols = list(CENSUS_VARS.values())
counties[numeric_cols] = counties[numeric_cols].apply(pd.to_numeric, errors="coerce")
counties[numeric_cols] = counties[numeric_cols].where(counties[numeric_cols] > -1e8)

# Derived metrics
counties["employment_rate"] = counties["employed"] / counties["labor_force"]
counties["poverty_rate"]    = counties["below_poverty"] / counties["poverty_universe"]

counties = counties[[
    "county_fips", "NAME",
    "median_household_income", "per_capita_income",
    "gini_index", "employment_rate", "poverty_rate",
]].rename(columns={"NAME": "county_name"})

print(f"  Loaded {len(counties)} counties")

# ─────────────────────────────────────────────
# STEP 4: Pull Statbotics EPA
# ─────────────────────────────────────────────

print(f"\nFetching Statbotics EPA for {STATBOTICS_YEARS}...")

SB_BASE = "https://api.statbotics.io/v3"

def fetch_statbotics_year(year: int) -> pd.DataFrame:
    resp = requests.get(f"{SB_BASE}/team_years?year={year}&limit=10000", timeout=60)
    resp.raise_for_status()
    return pd.DataFrame(resp.json())

sb_frames = []
for year in STATBOTICS_YEARS:
    try:
        df = fetch_statbotics_year(year)
        df["season"] = year
        sb_frames.append(df)
        print(f"  {year}: {len(df)} teams")
        time.sleep(0.2)
    except Exception as e:
        print(f"  Warning: {year} failed — {e}")

if sb_frames:
    sb_all = pd.concat(sb_frames, ignore_index=True)
else:
    print("  No Statbotics data was fetched; continuing without EPA data.")
    sb_all = pd.DataFrame()

# Print available EPA columns so you know what you have
epa_cols = [c for c in sb_all.columns if "epa" in c.lower() or c in ("wins", "losses", "count")]
print(f"  Available EPA columns: {epa_cols}")

if not sb_all.empty:
    # Average across years per team
    agg_dict = {c: "mean" for c in epa_cols if c != "season"}
    sb_avg = (
        sb_all.groupby("team")
        .agg({**agg_dict, "season": "count"})
        .rename(columns={"season": "seasons_counted"})
        .reset_index()
        .rename(columns={"team": "team_number"})
    )

    # YoY EPA change (last year - first year in range)
    if "epa_mean" in sb_all.columns and len(STATBOTICS_YEARS) >= 2:
        pivot = sb_all[sb_all["year"].isin([min(STATBOTICS_YEARS), max(STATBOTICS_YEARS)])]
        pivot = pivot.pivot(index="team", columns="year", values="epa_mean")
        first, last = f"epa_{min(STATBOTICS_YEARS)}", f"epa_{max(STATBOTICS_YEARS)}"
        pivot.columns = [f"epa_{c}" for c in pivot.columns]
        if first in pivot.columns and last in pivot.columns:
            pivot["epa_yoy_change"] = pivot[last] - pivot[first]
        pivot = pivot.reset_index().rename(columns={"team": "team_number"})
        sb_avg = sb_avg.merge(pivot[["team_number", "epa_yoy_change"]], on="team_number", how="left")
else:
    sb_avg = pd.DataFrame(columns=["team_number", "seasons_counted"])

sb_avg["team_number"] = pd.to_numeric(sb_avg["team_number"], errors="coerce")
us_teams["team_number"] = pd.to_numeric(us_teams["team_number"], errors="coerce")
print(f"  Averaged EPA for {len(sb_avg)} teams")

# ─────────────────────────────────────────────
# STEP 5: Join and save
# ─────────────────────────────────────────────

print("\nJoining datasets...")

result = us_teams[[
    "team_number", "nickname", "name", "city", "state_prov",
    "postal_code", "lat", "lng", "county_fips", "rookie_year",
]].copy()

result = result.merge(sb_avg, on="team_number", how="left")
result = result.merge(counties, on="county_fips", how="left")

has_econ = result["median_household_income"].notna().sum()
has_epa  = result["epa_mean"].notna().sum() if "epa_mean" in result.columns else 0
print(f"  Teams with economic data: {has_econ} / {len(result)}")
print(f"  Teams with EPA data:      {has_epa} / {len(result)}")

result.to_csv("frc_teams_with_economics.csv", index=False)
print(f"\n✓ Saved {len(result)} teams -> frc_teams_with_economics.csv")

# Quick correlation summary
if "epa_mean" in result.columns:
    print("\nCorrelations with epa_mean:")
    for col in ["median_household_income", "per_capita_income", "gini_index",
                "employment_rate", "poverty_rate"]:
        if col in result.columns:
            r = result[["epa_mean", col]].dropna().corr().iloc[0, 1]
            print(f"  vs {col:<30} r = {r:+.3f}")