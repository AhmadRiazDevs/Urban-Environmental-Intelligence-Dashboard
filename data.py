"""
Data Module
Combines:
  1. Synthetic Air Quality Dataset Generator (OpenAQ-style, 100 stations, hourly, 2025)
  2. TLC Trip Data Ingestion with optional web scraping automation.

Key Functions (Air Quality):
- build_dataset: Generate and save the full air quality dataset
- load_dataset: Load dataset from pickle
- STATION_META: DataFrame of station metadata

Key Functions (TLC):
- download_tlc_data: Main function to download parquet files
- scrape_tlc_urls: (BONUS) Automated URL discovery from TLC website
- check_missing_months: Detect missing data files
- impute_december: Handle missing December 2025 data
"""

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import logging
from pathlib import Path
from typing import List, Dict
import re
from datetime import datetime

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Part 1: Synthetic Air Quality Dataset Generator
# ─────────────────────────────────────────────────────────────

SEED = 42
np.random.seed(SEED)

N_STATIONS = 100
YEAR = 2025
HOURS_PER_YEAR = 8760  # 365 * 24

STATIONS = []
for i in range(N_STATIONS):
    zone = "Industrial" if i < 50 else "Residential"
    region = ["North", "South", "East", "West"][i % 4]
    pop_density = np.random.uniform(5000, 15000) if zone == "Industrial" else np.random.uniform(1000, 8000)
    STATIONS.append({
        "station_id": f"STN_{i:03d}",
        "zone": zone,
        "region": region,
        "lat": np.random.uniform(-60, 70),
        "lon": np.random.uniform(-180, 180),
        "population_density": pop_density,
    })

STATION_META = pd.DataFrame(STATIONS)


def generate_station_data(station_row):
    """Generate hourly environmental readings for one station for 2025."""
    sid = station_row["station_id"]
    zone = station_row["zone"]
    region = station_row["region"]

    timestamps = pd.date_range("2025-01-01", periods=HOURS_PER_YEAR, freq="h")
    hours = timestamps.hour.values
    day_of_year = timestamps.day_of_year.values

    # --- Diurnal cycle (traffic peaks at 8am and 6pm)
    diurnal = 1 + 0.5 * (np.exp(-((hours - 8) ** 2) / 8) + np.exp(-((hours - 18) ** 2) / 8))

    # --- Seasonal cycle (winter worse for pollution)
    seasonal = 1 + 0.3 * np.cos(2 * np.pi * day_of_year / 365 + np.pi)

    # --- Base pollution multiplier by zone
    base = 40 if zone == "Industrial" else 18

    # Noise
    noise = lambda s: np.random.normal(0, s, HOURS_PER_YEAR)

    pm25 = np.clip(base * diurnal * seasonal + noise(10), 1, 500)
    pm10 = np.clip(pm25 * np.random.uniform(1.5, 2.5) + noise(8), 2, 600)
    no2  = np.clip(base * 0.8 * diurnal + noise(6), 1, 300)
    ozone = np.clip(60 - 0.4 * no2 + 10 * np.sin(2 * np.pi * hours / 24) + noise(5), 5, 180)
    temperature = 15 + 10 * np.cos(2 * np.pi * day_of_year / 365 + np.pi) + 5 * np.sin(2 * np.pi * hours / 24) + noise(2)
    humidity = np.clip(60 + 10 * np.cos(2 * np.pi * day_of_year / 365) - 5 * np.sin(2 * np.pi * hours / 24) + noise(5), 10, 100)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "station_id": sid,
        "zone": zone,
        "region": region,
        "PM25": pm25,
        "PM10": pm10,
        "NO2": no2,
        "Ozone": ozone,
        "Temperature": temperature,
        "Humidity": humidity,
    })
    return df


def build_dataset(output_path="data/air_quality_2025.pkl"):
    """Build and save the full dataset."""
    Path("data").mkdir(exist_ok=True)
    print("Generating dataset for 100 stations × 8760 hours ...")
    frames = [generate_station_data(row) for _, row in STATION_META.iterrows()]
    df = pd.concat(frames, ignore_index=True)
    df.to_pickle(output_path)
    STATION_META.to_csv("data/station_metadata.csv", index=False)
    print(f"Dataset saved → {output_path}  |  Shape: {df.shape}")
    return df


def load_dataset(path="data/air_quality_2025.pkl"):
    """Load dataset from pickle."""
    return pd.read_pickle(path)


# ─────────────────────────────────────────────────────────────
# Part 2: TLC Trip Data Ingestion
# ─────────────────────────────────────────────────────────────


def scrape_tlc_urls(year: int = 2025, taxi_types: List[str] = ['yellow', 'green']) -> Dict[str, List[str]]:
    """
    BONUS FEATURE: Scrape TLC website to automatically discover parquet file URLs.
    
    Args:
        year: Target year for data download
        taxi_types: List of taxi types ('yellow', 'green', 'fhv', etc.)
    
    Returns:
        Dictionary mapping taxi_type -> list of URLs
        
    Example:
        urls = scrape_tlc_urls(2025, ['yellow', 'green'])
        # Returns: {'yellow': ['https://.../yellow_tripdata_2025-01.parquet', ...],
        #           'green': ['https://.../green_tripdata_2025-01.parquet', ...]}
    """
    base_url = "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
    
    logger.info(f"Scraping TLC website for {year} data URLs...")
    
    try:
        # Fetch the page
        response = requests.get(base_url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all links to parquet files
        urls_by_type = {taxi_type: [] for taxi_type in taxi_types}
        
        # Pattern: {color}_tripdata_{year}-{month}.parquet
        for taxi_type in taxi_types:
            pattern = re.compile(rf'{taxi_type}_tripdata_{year}-\d{{2}}\.parquet')
            
            for link in soup.find_all('a', href=True):
                href = link['href']
                if pattern.search(href):
                    # Construct full URL if relative
                    if not href.startswith('http'):
                        href = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{href}"
                    urls_by_type[taxi_type].append(href)
        
        # Sort URLs by month
        for taxi_type in taxi_types:
            urls_by_type[taxi_type].sort()
        
        logger.info(f"Found {sum(len(urls) for urls in urls_by_type.values())} URLs")
        return urls_by_type
        
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        logger.warning("Falling back to manual URL construction...")
        return construct_manual_urls(year, taxi_types)


def construct_manual_urls(year: int, taxi_types: List[str]) -> Dict[str, List[str]]:
    """
    Fallback: Manually construct URLs when scraping fails.
    TLC uses consistent URL patterns.
    
    Pattern: https://d37ci6vzurychx.cloudfront.net/trip-data/{color}_tripdata_{year}-{month}.parquet
    """
    base_url = "https://d37ci6vzurychx.cloudfront.net/trip-data"
    urls_by_type = {}
    
    for taxi_type in taxi_types:
        urls = []
        for month in range(1, 13):  # All 12 months
            month_str = f"{month:02d}"
            url = f"{base_url}/{taxi_type}_tripdata_{year}-{month_str}.parquet"
            urls.append(url)
        urls_by_type[taxi_type] = urls
    
    return urls_by_type


def download_file(url: str, save_path: Path) -> bool:
    """
    Download a single parquet file with progress logging.
    
    Args:
        url: File URL
        save_path: Local path to save file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading: {url}")
        
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        # Create parent directory if needed
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                
                # Log progress every 10MB
                if downloaded % (10 * 1024 * 1024) < 8192:
                    progress = (downloaded / total_size * 100) if total_size > 0 else 0
                    logger.info(f"  Progress: {progress:.1f}%")
        
        logger.info(f"  ✓ Saved to {save_path}")
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"  ✗ Download failed: {e}")
        return False


def download_tlc_data(
    year: int = 2025,
    taxi_types: List[str] = ['yellow', 'green'],
    output_dir: str = 'data/raw',
    use_scraping: bool = True
) -> Dict[str, List[Path]]:
    """
    Main function to download all TLC trip data for specified year and taxi types.
    
    Args:
        year: Target year
        taxi_types: List of taxi types to download
        output_dir: Directory to save files
        use_scraping: If True, attempt to scrape URLs; if False, use manual construction
        
    Returns:
        Dictionary mapping taxi_type -> list of downloaded file paths
    """
    logger.info(f"Starting TLC data download for {year}...")
    
    # Get URLs
    if use_scraping:
        urls_by_type = scrape_tlc_urls(year, taxi_types)
    else:
        urls_by_type = construct_manual_urls(year, taxi_types)
    
    # Download all files
    downloaded_files = {taxi_type: [] for taxi_type in taxi_types}
    
    for taxi_type, urls in urls_by_type.items():
        logger.info(f"\nDownloading {len(urls)} files for {taxi_type} taxis...")
        
        for url in urls:
            filename = url.split('/')[-1]
            save_path = Path(output_dir) / taxi_type / filename
            
            # Skip if already downloaded
            if save_path.exists():
                logger.info(f"  ⊙ Already exists: {filename}")
                downloaded_files[taxi_type].append(save_path)
                continue
            
            # Download
            success = download_file(url, save_path)
            if success:
                downloaded_files[taxi_type].append(save_path)
    
    return downloaded_files


def check_missing_months(downloaded_files: Dict[str, List[Path]]) -> List[str]:
    """
    Check which months are missing from the downloaded data.
    
    Args:
        downloaded_files: Output from download_tlc_data()
        
    Returns:
        List of missing month names (e.g., ['december'])
    """
    expected_months = [f"{i:02d}" for i in range(1, 13)]
    missing_months = []
    
    for taxi_type, files in downloaded_files.items():
        # Extract months from filenames
        pattern = re.compile(r'_(\d{2})\.parquet$')
        found_months = set()
        
        for file in files:
            match = pattern.search(file.name)
            if match:
                found_months.add(match.group(1))
        
        # Check for missing
        for month in expected_months:
            if month not in found_months:
                month_name = datetime.strptime(month, '%m').strftime('%B').lower()
                if month_name not in missing_months:
                    missing_months.append(month_name)
    
    if missing_months:
        logger.warning(f"Missing months detected: {', '.join(missing_months)}")
    else:
        logger.info("All 12 months present for all taxi types")
    
    return missing_months


def impute_december_2025():
    """
    Handle missing December 2025 data by imputing from historical data.
    Formula: 0.3 * Dec_2023 + 0.7 * Dec_2024
    
    NOTE: This function requires PySpark/Dask to be implemented.
    Placeholder for now - implement with your big data tool.
    """
    logger.info("Imputing December 2025 from historical data...")
    
    # TODO: Implement with Spark/Dask
    # 1. Load december 2023 and 2024 data
    # 2. Calculate weighted average: 0.3 * dec_2023 + 0.7 * dec_2024
    # 3. Adjust timestamps to 2025
    # 4. Save as december_2025_imputed.parquet
    
    logger.warning("Imputation not yet implemented - add Spark/Dask logic here")
    
    return None


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Download with scraping (bonus points!)
    files = download_tlc_data(
        year=2025,
        taxi_types=['yellow', 'green'],
        use_scraping=True
    )
    
    # Check for missing months
    missing = check_missing_months(files)
    
    if 'december' in missing:
        impute_december_2025()
