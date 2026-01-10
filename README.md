# Weight Tracker

A simple offline macOS-friendly tracker for weight, waist, and calorie measurements with
trend charts, moving averages, and maintenance calorie estimates.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Measurements are stored locally at `~/.weight_tracker/measurements.sqlite`.
