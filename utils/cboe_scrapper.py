# cboe_scrapper.py
import os
import re
import requests
import pandas as pd
from time import sleep
import io
# ------------------------------
# Configuration
# ------------------------------
OUTPUT_DIR = "cboe_vix_futures_full"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Expiration dates à partir de 2013
EXPIRATION_DATES = [
    # 2013
    "2013-01-16","2013-02-13","2013-03-20","2013-04-17","2013-05-22","2013-06-19","2013-07-17","2013-08-21","2013-09-18","2013-10-16","2013-11-20","2013-12-18",
    # 2014
    "2014-01-22","2014-02-19","2014-03-18","2014-04-16","2014-05-21","2014-06-18","2014-07-16","2014-08-20","2014-09-17","2014-10-22","2014-11-19","2014-12-17",
    # 2015
    "2015-01-21","2015-02-18","2015-03-18","2015-04-15","2015-05-20","2015-06-17","2015-07-22","2015-08-19","2015-09-16","2015-10-21","2015-11-18","2015-12-16",
    # 2016
    "2016-01-20","2016-02-17","2016-03-16","2016-04-20","2016-05-18","2016-06-15","2016-07-20","2016-08-17","2016-09-21","2016-10-19","2016-11-16","2016-12-21",
    # 2017
    "2017-01-18","2017-02-15","2017-03-22","2017-04-19","2017-05-17","2017-06-21","2017-07-19","2017-08-16","2017-09-20","2017-10-18","2017-11-15","2017-12-20",
    # 2018
    "2018-01-17","2018-02-14","2018-03-21","2018-04-18","2018-05-16","2018-06-20","2018-07-18","2018-08-22","2018-09-19","2018-10-17","2018-11-21","2018-12-19",
    # 2019
    "2019-01-16","2019-02-13","2019-03-19","2019-04-17","2019-05-22","2019-06-19","2019-07-17","2019-08-21","2019-09-18","2019-10-16","2019-11-20","2019-12-18",
    # 2020
    "2020-01-22","2020-02-19","2020-03-18","2020-04-15","2020-05-20","2020-06-17","2020-07-22","2020-08-19","2020-09-16","2020-10-21","2020-11-18","2020-12-16",
    # 2021
    "2021-01-20","2021-02-17","2021-03-17","2021-04-21","2021-05-19","2021-06-16","2021-07-21","2021-08-18","2021-09-15","2021-10-20","2021-11-17","2021-12-22",
    # 2022
    "2022-01-19","2022-02-16","2022-03-15","2022-04-20","2022-05-18","2022-06-15","2022-07-20","2022-08-17","2022-09-21","2022-10-19","2022-11-16","2022-12-21",
    # 2023
    "2023-01-18","2023-02-15","2023-03-22","2023-04-19","2023-05-17","2023-06-21","2023-07-19","2023-08-16","2023-09-20","2023-10-18","2023-11-15","2023-12-20",
    # 2024
    "2024-01-17","2024-02-14","2024-03-20","2024-04-17","2024-05-22","2024-06-18","2024-07-17","2024-08-21","2024-09-18","2024-10-16","2024-11-20","2024-12-18",
    # 2025
    "2025-01-22","2025-02-19","2025-03-18","2025-04-16","2025-05-21","2025-06-18","2025-07-16","2025-08-20","2025-09-17","2025-10-22","2025-11-19","2025-12-17",
    # 2026
    "2026-01-21","2026-02-18","2026-03-18","2026-04-15","2026-05-19","2026-06-17","2026-07-22","2026-08-19","2026-09-16","2026-10-21","2026-11-18","2026-12-16"
]

# ------------------------------
# Fonctions
# ------------------------------
import io  # <-- ajouter en haut du script



def download_csv(date_str):
    """Download the VX future CSV for a given expiration date."""
    url = f"https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/VX/VX_{date_str}.csv"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            return pd.read_csv(io.StringIO(r.text))  # <-- utiliser io.StringIO
        else:
            print(f"Warning: Status {r.status_code} for {url}")
            return None
    except Exception as e:
        print(f"Failed {url}: {e}")
        return None


def process_futures(df):
    """Parse expiration_date from Futures column."""
    df["expiration_date"] = pd.to_datetime(
        df["Futures"].str.extract(r"\((.*?)\)")[0],
        format="%b %Y",
        errors="coerce"
    )
    return df

# ------------------------------
# Main
# ------------------------------
all_dfs = []

for date_str in EXPIRATION_DATES:
    date_fmt = pd.to_datetime(date_str).strftime("%Y-%m-%d")
    print(f"Downloading CSV for {date_fmt}")
    df = download_csv(date_fmt)
    if df is not None:
        df = process_futures(df)
        all_dfs.append(df)
    sleep(0.5)  # éviter le throttling

if not all_dfs:
    print("No data downloaded")
    exit()

df_all = pd.concat(all_dfs, ignore_index=True)

# Sauvegarde du CSV complet
full_path = os.path.join(OUTPUT_DIR, "vix_futures_all.csv")
df_all.to_csv(full_path, index=False)
print(f"Full CSV saved: {full_path}")

# ------------------------------
# Génération Front / 2M / 3M
# ------------------------------
grouped_sorted = df_all.groupby("Trade Date").apply(
    lambda x: x.sort_values("expiration_date")
).reset_index(drop=True)

front_month = grouped_sorted.groupby("Trade Date").nth(0).reset_index()
second_month = grouped_sorted.groupby("Trade Date").nth(1).reset_index()
third_month = grouped_sorted.groupby("Trade Date").nth(2).reset_index()

front_month.to_csv(os.path.join(OUTPUT_DIR, "vix_futures_front_month.csv"), index=False)
second_month.to_csv(os.path.join(OUTPUT_DIR, "vix_futures_2M.csv"), index=False)
third_month.to_csv(os.path.join(OUTPUT_DIR, "vix_futures_3M.csv"), index=False)

print("Front Month / 2M / 3M CSVs generated")
