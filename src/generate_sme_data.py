"""Generate the synthetic SME dataset used in this project.

The data is synthetic. Defects (missing values, outliers, mixed formats,
near-duplicates and invalid rows) are planted on purpose so the cleaning
steps in the notebook have something real to fix.

Run from the repo root:  python src/generate_sme_data.py
"""
import random
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_TOTAL = 350
N_NEAR_DUPLICATES = 5
N_BASE = N_TOTAL - N_NEAR_DUPLICATES
REFERENCE_YEAR = 2025  # year used to derive years_in_operation

random.seed(SEED)
np.random.seed(SEED)

PREFIXES = ["Alpha", "Blue", "Crown", "Delta", "Eagle", "First", "Global", "Harbor", "Icon", "Jade",
            "King", "Lotus", "Metro", "Noble", "Orbit", "Peak", "Quest", "Ridge", "Summit", "Terra",
            "Unity", "Vista", "West", "Xcel", "York", "Zenith", "Apex", "Bright", "Core", "Dawn"]
SUFFIXES = ["Solutions", "Enterprises", "Group", "Services", "Industries", "Partners", "Works",
            "Associates", "Consulting", "Trading", "Systems", "Ventures", "Corp", "Co", "Hub"]
STRUCTURES = ["Sole Proprietorship", "Partnership", "Private Limited", "LLC", "Corporation"]
TAGS = {"Sole Proprietorship": "", "Partnership": "& Partners", "Private Limited": "Pvt Ltd",
        "LLC": "LLC", "Corporation": "Inc."}

CITY_STATE = [
    ("New York", "NY"), ("Los Angeles", "CA"), ("Chicago", "IL"), ("Houston", "TX"),
    ("Phoenix", "AZ"), ("Philadelphia", "PA"), ("San Antonio", "TX"), ("San Diego", "CA"),
    ("Dallas", "TX"), ("San Jose", "CA"), ("Austin", "TX"), ("Jacksonville", "FL"),
    ("Fort Worth", "TX"), ("Columbus", "OH"), ("Charlotte", "NC"), ("Indianapolis", "IN"),
    ("San Francisco", "CA"), ("Seattle", "WA"), ("Denver", "CO"), ("Nashville", "TN"),
    ("Oklahoma City", "OK"), ("Portland", "OR"), ("Las Vegas", "NV"), ("Memphis", "TN"),
    ("Louisville", "KY"), ("Baltimore", "MD"), ("Milwaukee", "WI"), ("Albuquerque", "NM"),
    ("Tucson", "AZ"), ("Fresno", "CA"), ("Sacramento", "CA"), ("Mesa", "AZ"),
    ("Kansas City", "MO"), ("Atlanta", "GA"), ("Omaha", "NE"), ("Colorado Springs", "CO"),
    ("Raleigh", "NC"), ("Long Beach", "CA"), ("Virginia Beach", "VA"), ("Minneapolis", "MN"),
]

SECTORS = ["Retail", "Manufacturing", "Technology", "Hospitality", "Construction",
           "Healthcare", "Professional Services", "Logistics", "Agriculture", "Education"]
SECTOR_P = [0.18, 0.15, 0.16, 0.10, 0.12, 0.10, 0.10, 0.05, 0.025, 0.015]

DIGITAL_LEVELS = ["Low", "Medium", "High", "LOW", "med", "HIGH", "low", "Med", "MEDIUM", "medium", "high"]
DIGITAL_P = [0.25, 0.30, 0.20, 0.05, 0.04, 0.03, 0.04, 0.03, 0.03, 0.02, 0.01]

RECESSION_YEARS = {1980, 1981, 1982, 1990, 1991, 2001, 2008, 2009, 2020}

NOTES = [
    "Strong local presence.", "Expanding to new markets.", "Recently rebranded.",
    "Facing supply chain issues.", "Owner plans to retire soon.", "Govt contract pending.",
    "Recvd grant last yr.", "Opened 2nd location.", "Wnr of local biz award 2023.",
    "Online sales growing QoQ.", "Seasonal business - slow in winter.", "New mgmt team since 2022.",
    "Exp staff turnover issues.", "Fmly-run business.", "Seeking investors.",
    "Compliance audit due.", "Recieved grant last yr.", "Expnading to new markts.",
    "Suply chain delays.", "TBD", "tbd", "N/A", "", None,
]

DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%b %Y", "%Y"]


def clean_spaces(text):
    return re.sub(r"\s+", " ", text).strip()


def make_name(structure, used):
    """Build a business name that is unique across the base records.

    The legal tag (LLC, Inc., Pvt Ltd) sits in different places on purpose.
    """
    tag = TAGS[structure]
    while True:
        p = random.choice(PREFIXES)
        s = random.choice(SUFFIXES)
        options = [f"{p} {s} {tag}", f"{p} & {random.choice(PREFIXES)} {s}",
                   f"The {p} {s}", f"{p}{s} {tag}"]
        if tag in ("LLC", "Inc.", "Pvt Ltd"):
            options += [f"{tag} {p} {s}", f"{p} {s}, {tag}"]
        name = clean_spaces(random.choice(options))
        if name not in used:
            used.add(name)
            return name


def make_sector_pool(n):
    """Fix the counts of the two rare sectors so they land in the 5-10 range."""
    fixed = {"Agriculture": 8, "Education": 6}
    others = [s for s in SECTORS if s not in fixed]
    p = np.array([SECTOR_P[SECTORS.index(s)] for s in others])
    pool = [str(s) for s in np.random.choice(others, size=n - sum(fixed.values()), p=p / p.sum())]
    for sector, count in fixed.items():
        pool += [sector] * count
    np.random.shuffle(pool)
    return pool


def build_row(used_names, sector):
    structure = random.choice(STRUCTURES)
    name = make_name(structure, used_names)
    city, state = random.choice(CITY_STATE)

    # founding_year with mixed formats: "2001", "01" or "2001-01-01"
    founding_year_raw = random.randint(1980, 2024)
    r = random.random()
    if r < 0.06:
        founding_year = str(founding_year_raw)[-2:]
    elif r < 0.10:
        founding_year = f"{founding_year_raw}-01-01"
    else:
        founding_year = str(founding_year_raw)

    employees = int(np.random.lognormal(3.5, 1.0))
    employees = max(5, min(employees, 250))
    if random.random() < 0.015:
        employees = random.randint(800, 1500)  # outlier

    years_true = REFERENCE_YEAR - founding_year_raw
    r = random.random()
    if r < 0.06:
        years_in_operation = years_true + random.randint(3, 10)  # too long
    elif r < 0.09:
        years_in_operation = -random.randint(1, 5)  # negative
    else:
        years_in_operation = years_true

    remote_pct = np.nan if random.random() < 0.12 else round(random.uniform(0, 100), 1)

    revenue = max(50000, employees * random.uniform(8000, 60000))
    if random.random() < 0.015:
        revenue = random.uniform(50_000_000, 120_000_000)  # outlier
    if random.random() < 0.02:
        revenue = 20_000_000  # illogical: tiny team, huge revenue
        employees = 5
    revenue = round(revenue, 2)

    margin = random.uniform(-0.25, 0.45)
    profit = round(revenue * margin, 2)
    r = random.random()
    if r < 0.07:
        margin_stored = round(-margin, 4)  # wrong sign
    elif r < 0.19:
        margin_stored = np.nan  # missing
    else:
        margin_stored = round(margin, 4)
    if random.random() < 0.04:  # negative margin with positive profit
        profit = abs(profit)
        margin_stored = np.nan if pd.isna(margin_stored) else -abs(margin_stored)

    cash_flow = round(profit / 12 * random.uniform(0.7, 1.3), 2)

    assets = round(max(10000, revenue * random.uniform(0.5, 3.0) * (1 + years_true * 0.02)), 2)
    liabilities = round(assets * random.uniform(0.1, 0.8), 2)
    de = round(liabilities / (assets - liabilities), 4)
    r = random.random()
    if r < 0.03:
        de = round(random.uniform(8, 15), 4)  # outlier
    elif r < 0.055:
        de = 0.0  # invalid: zero with liabilities > 0

    rating = np.nan if random.random() < 0.13 else round(random.uniform(1.0, 5.0), 1)
    followers = min(int(np.random.exponential(15000)), 500000)

    has_loan = random.choice(["Yes", "No", "Yes"])
    if has_loan == "Yes":
        loan_amount = round(random.uniform(10000, 2_000_000), 2)
    else:
        loan_amount = np.nan if random.random() < 0.7 else 0

    r = random.random()
    sustainability = "Yes" if r < 0.04 else (np.nan if r < 0.08 else "No")

    update = date(random.randint(2020, 2025), random.randint(1, 12), random.randint(1, 28))
    last_update = update.strftime(random.choice(DATE_FORMATS))

    return {
        "sme_id": "",
        "business_name": name,
        "city": city,
        "state": state,
        "founding_year": founding_year,
        "industry_sector": sector,
        "employee_count": employees,
        "business_structure": structure,
        "years_in_operation": years_in_operation,
        "remote_workers_percentage": remote_pct,
        "annual_revenue": revenue,
        "annual_profit": profit,
        "profit_margin": margin_stored,
        "monthly_cash_flow": cash_flow,
        "total_assets": assets,
        "total_liabilities": liabilities,
        "debt_to_equity_ratio": de,
        "num_products_services": random.randint(1, 50),
        "avg_customer_rating": rating,
        "social_media_followers": followers,
        "website_exists": random.choice(["Yes", "No", "Yes", "Yes"]),
        "export_products": random.choice(["Yes", "No", "No", "No"]),
        "digital_adoption_level": str(np.random.choice(DIGITAL_LEVELS, p=DIGITAL_P)),
        "founded_during_recession": "Yes" if founding_year_raw in RECESSION_YEARS else "No",
        "has_loan_outstanding": has_loan,
        "loan_amount": loan_amount,
        "num_competitors_in_region": random.randint(0, 25),
        "sustainability_certified": sustainability,
        "last_financial_update": last_update,
        "notes": random.choice(NOTES),
    }


def near_duplicate(row):
    """Copy a record with a small change to the name and the headcount."""
    dup = dict(row)
    dup["employee_count"] = max(5, int(row["employee_count"]) + random.choice([-3, -2, -1, 1, 2, 3]))
    name = row["business_name"]
    edits = [
        lambda n: n.replace(" & ", " and "),
        lambda n: n.upper(),
        lambda n: n + " Ltd",
        lambda n: n.replace(".", ""),
        lambda n: n.replace(",", ""),
        lambda n: re.sub(r"^The ", "", n),
    ]
    random.shuffle(edits)
    for edit in edits:
        new_name = edit(name)
        if new_name != name:
            dup["business_name"] = new_name
            break
    else:
        dup["business_name"] = name + " Co"
    return dup


def main():
    used_names = set()
    sector_pool = make_sector_pool(N_BASE)
    rows = [build_row(used_names, sector_pool.pop()) for _ in range(N_BASE)]

    candidates = [i for i, r in enumerate(rows) if 6 <= r["employee_count"] <= 250]
    for idx in random.sample(candidates, N_NEAR_DUPLICATES):
        rows.append(near_duplicate(rows[idx]))

    df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    df["sme_id"] = [f"SME-{i:04d}" for i in range(1, len(df) + 1)]

    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "sme_dataset.csv", index=False, encoding="utf-8")
    print(f"Wrote {len(df)} rows x {df.shape[1]} columns to {out_dir / 'sme_dataset.csv'}")


if __name__ == "__main__":
    main()
