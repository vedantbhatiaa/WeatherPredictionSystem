"""
crop_recommender.py — KisanMausam v2
─────────────────────────────────────────────────────────────────────────────
DATA-DRIVEN scoring using Dataset/crop_parameters.json (built from Kaggle
Crop Production India 1997-2015, 246,091 records, 33 states, 124 crops).

What changed vs v1 (hardcoded guesses):
  OLD: state bonus = hand-tuned integers (max +4) with no data foundation.
  NEW: state_crop_affinity 0-100 from actual production volumes.
       Wheat in Punjab → 100 (top crop). Rice in Punjab → 65. Cotton → 11.
       These numbers come from cumulative production 1997-2015, not guesswork.

  OLD: temp/humidity scoring arbitrary (if temp>30: score-=10).
  NEW: temp/humidity scoring from per-crop optimal bands in TEMP_RANGES and
       HUM_RANGES (ICAR agronomic research). Score peaks in optimal band,
       degrades linearly outside it, crashes at critical thresholds.

  OLD: soil scoring made-up.
  NEW: soil_scores per crop from soil-science literature validated against
       state-dominant-soil mapping.

Scoring breakdown (max 100):
  30 pts — State-crop affinity     (from production data)
  20 pts — Temperature match       (from ICAR optimal bands)
  15 pts — Humidity match          (from ICAR optimal bands)
  10 pts — Season match            (from production season dominance)
  10 pts — Soil type match         (from soil-science literature)
   8 pts — Soil pH match
   7 pts — Soil moisture match
─────────────────────────────────────────────────────────────────────────────
"""

import os, json, logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── Load crop_parameters.json once at import ─────────────────────────────────
_BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARAMS_PATH = os.path.join(_BASE_DIR, 'Dataset', 'crop_parameters.json')
_PARAMS: dict = {}


def _load_params() -> dict:
    global _PARAMS
    if _PARAMS:
        return _PARAMS
    try:
        with open(_PARAMS_PATH) as f:
            _PARAMS = json.load(f)
        logger.info(f"crop_parameters.json loaded — "
                    f"{len(_PARAMS.get('state_crop_affinity', {}))} states, "
                    f"{len(_PARAMS.get('soil_scores', {}))} crops with soil data")
    except Exception as e:
        logger.warning(f"crop_parameters.json not found ({e}). "
                       f"Run KisanMausam_Agricultural_Analysis.ipynb to generate it. "
                       f"Using built-in fallback parameters.")
        _PARAMS = _FALLBACK_PARAMS
    return _PARAMS


# ── Fallback parameters (used if JSON not generated yet) ─────────────────────
# These are the same data-driven values pre-embedded so the app works
# even before the notebook has been run.
_FALLBACK_PARAMS = {
    "state_crop_affinity": {
        "Maharashtra":   {"Cotton":8,"Soybean":4,"Sugarcane":100,"Rice":5,"Bajra":2,"Chickpea":2,"Groundnut":2,"Maize":2,"Onion":3,"Tomato":2,"Wheat":2},
        "Punjab":        {"Wheat":100,"Rice":65,"Sugarcane":32,"Cotton":11,"Maize":3,"Groundnut":2,"Chickpea":2},
        "Haryana":       {"Wheat":100,"Rice":60,"Sugarcane":25,"Cotton":15,"Bajra":20,"Maize":8},
        "Uttar Pradesh": {"Sugarcane":100,"Wheat":80,"Rice":40,"Maize":10,"Onion":5,"Chickpea":8},
        "Madhya Pradesh":{"Soybean":100,"Wheat":60,"Cotton":25,"Chickpea":45,"Maize":18},
        "Rajasthan":     {"Wheat":100,"Bajra":47,"Maize":20,"Chickpea":13,"Soybean":11,"Groundnut":8},
        "Gujarat":       {"Cotton":100,"Groundnut":35,"Wheat":18,"Sugarcane":15,"Maize":8,"Soybean":5},
        "Karnataka":     {"Rice":100,"Sugarcane":30,"Maize":28,"Cotton":20,"Groundnut":18,"Tomato":5},
        "Tamil Nadu":    {"Rice":100,"Sugarcane":35,"Cotton":15,"Groundnut":12,"Tomato":8},
        "Andhra Pradesh":{"Rice":100,"Cotton":30,"Sugarcane":20,"Groundnut":18,"Maize":15},
        "Telangana":     {"Rice":100,"Cotton":45,"Maize":30,"Soybean":15,"Groundnut":12},
        "Bihar":         {"Wheat":100,"Rice":45,"Sugarcane":22,"Maize":18,"Chickpea":10},
        "West Bengal":   {"Rice":100,"Maize":8,"Tomato":5},
        "Odisha":        {"Rice":100,"Maize":15,"Groundnut":10},
        "Chhattisgarh":  {"Rice":100,"Maize":20,"Soybean":12},
        "Jharkhand":     {"Rice":100,"Maize":20,"Wheat":15},
        "Assam":         {"Rice":100,"Maize":8,"Sugarcane":12},
        "Kerala":        {"Rice":100,"Sugarcane":36,"Cotton":2,"Groundnut":2},
    },
    "dominant_season": {
        "Maharashtra":   {"Cotton":"Kharif","Soybean":"Kharif","Sugarcane":"Kharif","Wheat":"Rabi"},
        "Punjab":        {"Wheat":"Rabi","Rice":"Kharif","Cotton":"Kharif"},
        "Haryana":       {"Wheat":"Rabi","Rice":"Kharif","Cotton":"Kharif","Bajra":"Kharif"},
        "Uttar Pradesh": {"Wheat":"Rabi","Rice":"Kharif","Sugarcane":"Kharif"},
        "Madhya Pradesh":{"Wheat":"Rabi","Soybean":"Kharif","Cotton":"Kharif","Chickpea":"Rabi"},
        "Rajasthan":     {"Wheat":"Rabi","Bajra":"Kharif","Chickpea":"Rabi","Groundnut":"Kharif"},
        "Gujarat":       {"Cotton":"Kharif","Groundnut":"Kharif","Wheat":"Rabi"},
        "Karnataka":     {"Rice":"Kharif","Maize":"Kharif","Cotton":"Kharif","Sugarcane":"Kharif"},
        "Tamil Nadu":    {"Rice":"Kharif","Sugarcane":"Kharif","Cotton":"Kharif"},
        "Andhra Pradesh":{"Rice":"Kharif","Cotton":"Kharif","Groundnut":"Kharif"},
        "Telangana":     {"Rice":"Kharif","Cotton":"Kharif"},
        "Bihar":         {"Wheat":"Rabi","Rice":"Kharif","Sugarcane":"Kharif"},
        "West Bengal":   {"Rice":"Kharif"},
        "Odisha":        {"Rice":"Kharif"},
    },
    "national_yield_median": {
        "Bajra":1.1,"Chickpea":0.8,"Cotton":1.4,"Groundnut":1.0,
        "Maize":1.8,"Onion":10.3,"Rice":2.0,"Soybean":1.0,
        "Sugarcane":55.4,"Tomato":9.8,"Wheat":2.0,
    },
    "national_affinity": {
        "Sugarcane":100,"Rice":29,"Wheat":24,"Cotton":5,"Maize":5,
        "Groundnut":2,"Bajra":2,"Soybean":3,"Chickpea":2,"Onion":2,"Tomato":2,
    },
    "soil_scores": {
        "Wheat":     {"alluvial":20,"loamy":18,"black":10,"red":5,"sandy":-5,"clay":5},
        "Rice":      {"clay":22,"alluvial":20,"loamy":10,"black":5,"red":2,"sandy":-10},
        "Cotton":    {"black":25,"loamy":12,"alluvial":8,"clay":5,"red":3,"sandy":-8},
        "Soybean":   {"black":20,"loamy":18,"alluvial":12,"red":8,"clay":5,"sandy":3},
        "Chickpea":  {"loamy":18,"alluvial":15,"black":10,"sandy":8,"red":8,"clay":-5},
        "Groundnut": {"sandy":22,"loamy":18,"alluvial":12,"red":10,"black":5,"clay":-8},
        "Maize":     {"loamy":20,"alluvial":18,"black":10,"red":8,"sandy":5,"clay":3},
        "Bajra":     {"sandy":25,"loamy":15,"alluvial":10,"red":10,"black":5,"clay":-5},
        "Sugarcane": {"alluvial":22,"loamy":20,"black":15,"clay":10,"red":5,"sandy":-5},
        "Onion":     {"loamy":22,"alluvial":20,"sandy":12,"red":10,"black":8,"clay":3},
        "Tomato":    {"loamy":22,"alluvial":20,"sandy":15,"red":12,"black":8,"clay":3},
    },
    "temp_ranges": {
        "Wheat":     {"opt_min":12,"opt_max":25,"stress_above":30,"critical_above":35},
        "Rice":      {"opt_min":22,"opt_max":32,"stress_above":36,"critical_above":40},
        "Cotton":    {"opt_min":21,"opt_max":35,"stress_above":38,"critical_above":42},
        "Soybean":   {"opt_min":20,"opt_max":30,"stress_above":35,"critical_above":38},
        "Chickpea":  {"opt_min":15,"opt_max":25,"stress_above":30,"critical_above":35},
        "Groundnut": {"opt_min":22,"opt_max":33,"stress_above":37,"critical_above":40},
        "Maize":     {"opt_min":18,"opt_max":32,"stress_above":36,"critical_above":40},
        "Bajra":     {"opt_min":25,"opt_max":38,"stress_above":42,"critical_above":46},
        "Sugarcane": {"opt_min":24,"opt_max":35,"stress_above":38,"critical_above":42},
        "Onion":     {"opt_min":13,"opt_max":24,"stress_above":28,"critical_above":35},
        "Tomato":    {"opt_min":18,"opt_max":27,"stress_above":32,"critical_above":38},
    },
    "humidity_ranges": {
        "Wheat":     {"opt_min":40,"opt_max":65},
        "Rice":      {"opt_min":70,"opt_max":90},
        "Cotton":    {"opt_min":50,"opt_max":72},
        "Soybean":   {"opt_min":55,"opt_max":75},
        "Chickpea":  {"opt_min":35,"opt_max":60},
        "Groundnut": {"opt_min":50,"opt_max":70},
        "Maize":     {"opt_min":55,"opt_max":80},
        "Bajra":     {"opt_min":30,"opt_max":60},
        "Sugarcane": {"opt_min":65,"opt_max":90},
        "Onion":     {"opt_min":40,"opt_max":65},
        "Tomato":    {"opt_min":45,"opt_max":70},
    },
}


# ── MSP 2024-25 (CACP) ────────────────────────────────────────────────────────
MSP_INR = {
    "Wheat":2275,"Rice":2300,"Cotton":7121,"Soybean":4892,
    "Maize":2090,"Chickpea":5440,"Groundnut":6783,"Bajra":2625,
    "Sugarcane":340,"Onion":1800,"Tomato":1800,
}

SOW_WINDOW = {
    "Wheat":    "Nov–Dec (Rabi)",
    "Rice":     "Jun–Jul (Kharif)",
    "Cotton":   "May–Jun (Kharif)",
    "Soybean":  "Jun–Jul (Kharif)",
    "Maize":    "Jun–Jul (Kharif)",
    "Chickpea": "Oct–Nov (Rabi)",
    "Groundnut":"Jun–Jul (Kharif)",
    "Bajra":    "Jun–Jul (Kharif)",
    "Sugarcane":"Feb–Mar (Annual)",
    "Onion":    "Oct–Nov (Rabi)",
    "Tomato":   "Jun or Nov",
}

WATER_NEED = {
    "Wheat":"Moderate","Rice":"High","Cotton":"Moderate",
    "Soybean":"Moderate","Maize":"Moderate","Chickpea":"Low",
    "Groundnut":"Moderate","Bajra":"Low","Sugarcane":"High",
    "Onion":"Moderate","Tomato":"Moderate",
}

ALL_CROPS = list(MSP_INR.keys())


@dataclass
class CropRecommendation:
    crop:                  str
    score:                 int
    sow_window:            str
    water_need:            str
    avg_yield_t_ha:        float
    msp_inr:               int
    projected_revenue_inr: int
    season_match:          bool
    warnings:              List[str] = field(default_factory=list)


def _score_temperature(temp: float, ranges: dict) -> float:
    """Returns 0-20 based on how well temp fits the crop's optimal band."""
    opt_min = ranges['opt_min']
    opt_max = ranges['opt_max']
    stress  = ranges['stress_above']
    crit    = ranges['critical_above']

    if opt_min <= temp <= opt_max:
        return 20.0

    if temp < opt_min:
        # Cold penalty — linear from opt_min down to opt_min-10
        gap = opt_min - temp
        return max(0.0, 20.0 - gap * 2.0)

    if temp <= stress:
        # Slightly above optimal — small penalty
        gap = temp - opt_max
        return max(0.0, 20.0 - gap * 1.5)

    if temp <= crit:
        # Heat stress zone
        return max(0.0, 10.0 - (temp - stress) * 2.0)

    # Above critical — very low score
    return 0.0


def _score_humidity(hum: float, ranges: dict) -> float:
    """Returns 0-15 based on humidity fit."""
    h_min = ranges['opt_min']
    h_max = ranges['opt_max']

    if h_min <= hum <= h_max:
        return 15.0

    gap = abs(hum - h_min) if hum < h_min else abs(hum - h_max)
    return max(0.0, 15.0 - gap * 0.3)


def _score_soil(soil_type: str, soil_scores: dict) -> float:
    """Returns 0-10 based on soil type compatibility."""
    if not soil_type:
        return 5.0   # neutral if unknown
    raw = soil_scores.get(soil_type.lower(), 0)
    # Shift to 0-10 scale (raw range is -10 to +25)
    return max(0.0, min(10.0, (raw + 10) / 3.5))


def _score_ph(ph: float, crop: str) -> float:
    """Returns 0-8 based on soil pH suitability per crop."""
    pH_ranges = {
        "Wheat":    (6.0, 7.5), "Rice":     (5.5, 6.5),
        "Cotton":   (6.0, 8.0), "Soybean":  (6.0, 7.0),
        "Chickpea": (6.0, 8.0), "Groundnut":(5.5, 7.0),
        "Maize":    (5.8, 7.0), "Bajra":    (5.5, 7.5),
        "Sugarcane":(6.0, 7.5), "Onion":    (6.0, 7.0),
        "Tomato":   (5.8, 7.0),
    }
    lo, hi = pH_ranges.get(crop, (6.0, 7.5))
    if lo <= ph <= hi:
        return 8.0
    gap = abs(ph - lo) if ph < lo else abs(ph - hi)
    return max(0.0, 8.0 - gap * 4.0)


def _score_moisture(moisture: float, crop: str) -> float:
    """Returns 0-7 based on soil moisture suitability."""
    moisture_ranges = {
        "Wheat":    (35, 55), "Rice":     (60, 85), "Cotton":   (45, 65),
        "Soybean":  (45, 65), "Chickpea": (25, 50), "Groundnut":(35, 55),
        "Maize":    (45, 65), "Bajra":    (20, 45), "Sugarcane":(60, 80),
        "Onion":    (35, 55), "Tomato":   (45, 65),
    }
    lo, hi = moisture_ranges.get(crop, (35, 65))
    if lo <= moisture <= hi:
        return 7.0
    gap = abs(moisture - lo) if moisture < lo else abs(moisture - hi)
    return max(0.0, 7.0 - gap * 0.3)


class CropRecommender:
    def __init__(self):
        self.params = _load_params()

    def recommend(
        self,
        avg_forecast_temp:     float,
        avg_forecast_humidity: float,
        soil_pH:               float = 6.83,
        soil_moisture_pct:     float = 45.0,
        current_month:         int   = 4,
        state:                 str   = '',
        soil_type:             str   = '',
        top_n:                 int   = 5,
    ) -> List[CropRecommendation]:

        current_season = 'Kharif' if current_month in range(4, 11) else 'Rabi'
        p = self.params

        state_affinity  = p.get('state_crop_affinity', {}).get(state, {})
        national_aff    = p.get('national_affinity', {})
        dom_seasons     = p.get('dominant_season', {}).get(state, {})
        yield_ref       = p.get('national_yield_median', {})
        soil_scores_all = p.get('soil_scores', {})
        temp_ranges_all = p.get('temp_ranges', {})
        hum_ranges_all  = p.get('humidity_ranges', {})

        results = []
        for crop in ALL_CROPS:
            score    = 0.0
            warnings = []

            # ── 1. State-crop affinity (0-30 pts) ────────────────────────────
            if state_affinity:
                aff = state_affinity.get(crop, national_aff.get(crop, 5))
            else:
                aff = national_aff.get(crop, 5)
            score += (aff / 100.0) * 30.0

            # ── 2. Temperature match (0-20 pts) ──────────────────────────────
            t_ranges = temp_ranges_all.get(crop, {})
            if t_ranges:
                score += _score_temperature(avg_forecast_temp, t_ranges)
                if avg_forecast_temp > t_ranges.get('stress_above', 99):
                    warnings.append(f"Heat stress risk at {avg_forecast_temp:.0f}°C")
                if avg_forecast_temp < t_ranges.get('opt_min', 0) - 5:
                    warnings.append(f"Too cold at {avg_forecast_temp:.0f}°C")
            else:
                score += 10.0  # neutral

            # ── 3. Humidity match (0-15 pts) ──────────────────────────────────
            h_ranges = hum_ranges_all.get(crop, {})
            if h_ranges:
                score += _score_humidity(avg_forecast_humidity, h_ranges)
            else:
                score += 7.5   # neutral

            # ── 4. Season match (0-10 pts) ────────────────────────────────────
            dominant_season = dom_seasons.get(crop, SOW_WINDOW.get(crop, 'Kharif').split()[1] if 'Rabi' in SOW_WINDOW.get(crop,'') else 'Kharif')
            season_match = (dominant_season == current_season)
            if season_match:
                score += 10.0
            else:
                score += 2.0   # small consolation — may still be viable

            # ── 5. Soil type match (0-10 pts) ────────────────────────────────
            soil_crop_scores = soil_scores_all.get(crop, {})
            score += _score_soil(soil_type, soil_crop_scores)

            # ── 6. Soil pH match (0-8 pts) ────────────────────────────────────
            score += _score_ph(soil_pH, crop)

            # ── 7. Soil moisture match (0-7 pts) ─────────────────────────────
            score += _score_moisture(soil_moisture_pct, crop)

            # ── Revenue projection ────────────────────────────────────────────
            msp      = MSP_INR.get(crop, 2000)
            y_ha     = yield_ref.get(crop, 1.5)
            revenue  = int(y_ha * msp * 10)   # per quintal * t/ha * 10 qtl/t

            results.append(CropRecommendation(
                crop                  = crop,
                score                 = max(0, min(100, round(score))),
                sow_window            = SOW_WINDOW.get(crop, ''),
                water_need            = WATER_NEED.get(crop, 'Moderate'),
                avg_yield_t_ha        = y_ha,
                msp_inr               = msp,
                projected_revenue_inr = revenue,
                season_match          = season_match,
                warnings              = warnings,
            ))

        results.sort(key=lambda r: -r.score)
        return results[:top_n]