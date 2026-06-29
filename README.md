# Urban Environmental Intelligence Dashboard

This project is a Streamlit dashboard for visualizing and analyzing synthetic air quality data for 100 sensors in 2025.

## Files
- `Dashboard.py` — main Streamlit app.
- `data.py` — dataset generation and loading utilities.
- `Task1_pca.py`, `Task2_temporal.py`, `Task3_distribution.py`, `Task4_visual_audit.py` — task-specific analysis scripts.
- `data/` — data directory and generated dataset files.
- `outputs/` — output directory for saved figures or exports.

## Requirements
- Python 3.11
- `streamlit`
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `scipy`
- `requests`
- `beautifulsoup4`

## Install dependencies
Open a terminal in the project folder and run:

```powershell
cd "d:\Ahmad Riaz\Semester 6\Data Science\Assignment 2\UEI"
C:/Users/Ahmad/AppData/Local/Microsoft/WindowsApps/python3.11.exe -m pip install streamlit pandas numpy matplotlib scikit-learn scipy requests beautifulsoup4
```

## Run the dashboard
Start the Streamlit app with:

```powershell
C:/Users/Ahmad/AppData/Local/Microsoft/WindowsApps/python3.11.exe -m streamlit run Dashboard.py
```

The first run may generate the dataset and save it as `data/air_quality_2025.pkl`.

## Notes
- The dashboard uses `data.py` to generate or load the dataset.
- If VS Code is warning about missing imports, ensure the selected Python interpreter matches the one used to install packages.
- The app is designed to run from the project root folder so relative paths resolve correctly.
