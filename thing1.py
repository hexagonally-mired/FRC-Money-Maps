import requests
import pandas as pd

CENSUS_KEY = "8a19b110ab12960b20c0dcaf312e715d6e50966d"
YEAR = "2023"

VARIABLES = {
    "B19013_001E": "median_household_income",
    "B19301_001E": "per_capita_income",
    "B19083_001E": "gini_index",
    "B23025_003E": "labor_force",
    "B23025_004E": "employed",
}

def get_county_data(state_fips: str) -> pd.DataFrame:
    var_str = "NAME," + ",".join(VARIABLES.keys())
    url = (
        f"https://api.census.gov/data/{YEAR}/acs/acs5"
        f"?get={var_str}"
        f"&for=county:*"
        f"&in=state:{state_fips}"
        f"&key={CENSUS_KEY}"
    )
    resp = requests.get(url)
    resp.raise_for_status()
    
    data = resp.json()
    df = pd.DataFrame(data[1:], columns=data[0])  # row 0 is headers
    
    # Rename and cast types
    df = df.rename(columns=VARIABLES)
    df["state_fips"] = df["state"]
    df["county_fips"] = df["state"] + df["county"]  # 5-digit FIPS
    
    numeric_cols = list(VARIABLES.values())
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    
    # Compute employment rate
    df["employment_rate"] = df["employed"] / df["labor_force"]
    
    # Census uses -666666666 for missing — replace with NaN
    df[numeric_cols] = df[numeric_cols].where(df[numeric_cols] > 0)
    
    return df

# Pull PNW states
pnw_states = {"WA": "53", "OR": "41", "AK": "02", "ID": "16"}
frames = [get_county_data(fips) for fips in pnw_states.values()]
counties = pd.concat(frames, ignore_index=True)

print(counties[["NAME", "median_household_income", "gini_index", "employment_rate"]].head(10))