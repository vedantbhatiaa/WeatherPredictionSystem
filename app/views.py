"""
views.py — KisanMausam
──────────────────────────────────────────────────────────────────────────────
v2 — XGBoost forecast extension (days 8-15):

FORECAST:   Days 1-7  from Weatherbit API (accurate, real data).
            Days 8-15 from XGBoost chain prediction.
              - XGBoost trained on ERA5 dataset (39 cities, 199,407 rows).
              - Best model in the study: lowest RMSE, highest R2 across all cities.
              - City-aware: maps city name to LabelEncoder int so Varanasi
                predicts differently from Kochi.
              - True chaining: each prediction feeds the next day's lag features.
              - Graceful fallback: if xgb pkl not found → climatological blending.

SECURITY:   @login_required on dashboard, get_weather, update_soil, save_profile.

CITY BUG:   STATE_DEFAULT_CITY map. Stale-Mumbai bug fixed via _resolve_city().

ADVISORY:   compute_ag_impact uses real Weatherbit max_temp for heat stress.
            Mumbai April max ~29°C → 0 heat days. Varanasi ~42°C → real count.

CHATBOT:    process_message injects live context so chatbot answers weather
            queries, forecasts, other-city weather, location-specific crops.

PEAK DAY:   get_weather AJAX returns peak_risk_day for advisory tile updates.
"""

import os, json, logging, pickle, math
import requests, numpy as np, pandas as pd, pytz
from datetime import datetime, timedelta

from django.conf import settings
from django.shortcuts import render, redirect
from django.contrib.auth import login, update_session_auth_hash
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.contrib import messages

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .alert_engine import AlertEngine
from .crop_recommender import CropRecommender
from .farm_assistant import FarmAssistant
from .models import FarmerProfile, WeatherCache, AlertLog

logger = logging.getLogger(__name__)

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'Dataset', 'weather.csv')

# ── XGBoost model paths ───────────────────────────────────────────────────────
XGB_MODEL_PATH = os.path.join(BASE_DIR, 'Dataset', 'xgb_weather_model.pkl')
XGB_LE_PATH    = os.path.join(BASE_DIR, 'Dataset', 'city_label_encoder.pkl')

# Feature order MUST exactly match the order used during notebook training
XGB_FEATURES = [
    'temp_lag1', 'temp_lag7', 'temp_lag14', 'hum_lag1', 'wind_lag1',
    'roll7_mean', 'roll7_std', 'roll30_mean', 'precip_lag1',
    'month_sin', 'month_cos', 'doy_sin', 'doy_cos', 'city_enc',
    'temp_min', 'humidity', 'wind_max'
]

# Initialise chatbot with API keys so it can make live calls for other cities
farm_bot = FarmAssistant(
    owm_key  = getattr(settings, 'OWM_KEY',  ''),
    wbit_key = getattr(settings, 'WBIT_KEY', ''),
)

# ── State → default city ──────────────────────────────────────────────────────
STATE_DEFAULT_CITY = {
    'Maharashtra':      'Mumbai',
    'Punjab':           'Ludhiana',
    'Haryana':          'Hisar',
    'Uttar Pradesh':    'Lucknow',
    'Madhya Pradesh':   'Bhopal',
    'Rajasthan':        'Jaipur',
    'Gujarat':          'Ahmedabad',
    'Karnataka':        'Bengaluru',
    'Tamil Nadu':       'Chennai',
    'Andhra Pradesh':   'Vijayawada',
    'Telangana':        'Hyderabad',
    'Bihar':            'Patna',
    'West Bengal':      'Kolkata',
    'Odisha':           'Bhubaneswar',
    'Himachal Pradesh': 'Shimla',
    'Uttarakhand':      'Dehradun',
    'Chhattisgarh':     'Raipur',
    'Jharkhand':        'Ranchi',
    'Assam':            'Guwahati',
    'Kerala':           'Kochi',
    'Other':            'Delhi',
}


# ══════════════════════════════════════════════════════════════════════════════
# MODEL CACHING
# ══════════════════════════════════════════════════════════════════════════════

_model_cache = {}
_xgb_cache   = {}


def get_trained_models():
    """RF rain classifier + simple RF regressors for rain/temp/hum/wind — trained once."""
    if _model_cache:
        return _model_cache
    logger.info("Training RF models (one-time startup)...")
    try:
        df = pd.read_csv(DATASET_PATH).dropna().drop_duplicates()
        le = LabelEncoder()
        df_c = df.copy()
        df_c["WindGustDir"]  = le.fit_transform(df_c["WindGustDir"])
        df_c["RainTomorrow"] = le.fit_transform(df_c["RainTomorrow"])
        X = df_c[["MinTemp", "MaxTemp", "Humidity", "Temp", "WindSpeed"]]
        y = df_c["RainTomorrow"]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        rain_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        rain_clf.fit(Xtr, ytr)

        def make_regressor(col):
            X_ = df[col].iloc[:-1].values.reshape(-1, 1)
            y_ = df[col].iloc[1:].values
            m  = RandomForestRegressor(n_estimators=100, random_state=0)
            m.fit(X_, y_)
            return m

        _model_cache['rain']     = rain_clf
        _model_cache['temp']     = make_regressor('Temp')
        _model_cache['humidity'] = make_regressor('Humidity')
        _model_cache['wind']     = make_regressor('WindSpeed')
        logger.info("RF models ready.")
    except Exception as e:
        logger.error(f"Model training failed: {e}")
    return _model_cache


def _load_xgb_model() -> dict:
    """
    Load XGBoost model + city LabelEncoder from Dataset/ — cached after first load.
    Returns empty dict if pkl files not found (graceful fallback to climatology).
    """
    global _xgb_cache
    if _xgb_cache:
        return _xgb_cache
    try:
        with open(XGB_MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(XGB_LE_PATH, 'rb') as f:
            le = pickle.load(f)
        _xgb_cache = {'model': model, 'le': le}
        logger.info(
            f"XGBoost model loaded from Dataset/ — "
            f"{len(le.classes_)} cities in encoder: {list(le.classes_)}"
        )
    except Exception as e:
        logger.warning(
            f"XGBoost pkl not found ({e}). "
            f"Run notebook Section 11 save cell to generate Dataset/xgb_weather_model.pkl. "
            f"Falling back to climatological blending for days 8-15."
        )
    return _xgb_cache


# ══════════════════════════════════════════════════════════════════════════════
# WEATHER DATA
# ══════════════════════════════════════════════════════════════════════════════

def fetch_current_weather(city: str) -> dict:
    url = (f"https://api.openweathermap.org/data/2.5/weather"
           f"?q={city}&appid={settings.OWM_KEY}&units=metric")
    resp = requests.get(url, timeout=8)
    resp.raise_for_status()
    d = resp.json()
    return {
        "city":         d["name"],
        "country":      d["sys"]["country"],
        "current_temp": round(d["main"]["temp"]),
        "feels_like":   round(d["main"]["feels_like"]),
        "temp_min":     round(d["main"]["temp_min"]),
        "temp_max":     round(d["main"]["temp_max"]),
        "humidity":     round(d["main"]["humidity"]),
        "wind_speed":   round(d["wind"]["speed"], 1),
        "description":  d["weather"][0]["description"],
        "clouds":       d["clouds"]["all"],
    }


def fetch_forecast_weather(city: str) -> dict:
    """
    Weatherbit 16-day daily forecast.
    forecast_max_temps = daily HIGH temp (iPhone/Google convention).
    forecast_avg_temps = daily average (used for crop scoring).
    """
    today = datetime.now().strftime("%Y-%m-%d")
    end   = (datetime.now() + timedelta(days=16)).strftime("%Y-%m-%d")
    url   = (f"https://api.weatherbit.io/v2.0/forecast/daily"
             f"?city={city}&key={settings.WBIT_KEY}"
             f"&start_date={today}&end_date={end}&units=M")
    resp = requests.get(url, timeout=8)
    resp.raise_for_status()
    days = resp.json()["data"]
    d0   = days[0]
    return {
        "avg_temp":           round(d0["temp"]),
        "min_temp":           round(d0["min_temp"]),
        "max_temp":           round(d0["max_temp"]),
        "humidity":           round(d0["rh"]),
        "precipitation":      d0["precip"],
        "wind_speed":         round(d0["wind_spd"], 1),
        "forecast_max_temps": [round(d["max_temp"],  1) for d in days],
        "forecast_avg_temps": [round(d["temp"],      1) for d in days],
        "forecast_min_temps": [round(d["min_temp"],  1) for d in days],
        "forecast_hums":      [round(d["rh"],        1) for d in days],
        "forecast_winds":     [round(d["wind_spd"],  1) for d in days],
        "forecast_precip":    [round(d["precip"],    2) for d in days],
    }


def get_weather_with_fallback(city: str):
    try:
        current = fetch_current_weather(city)
        daily   = fetch_forecast_weather(city)
        try:
            WeatherCache.objects.update_or_create(
                city=city,
                defaults={
                    'current_json':  json.dumps(current),
                    'forecast_json': json.dumps(daily),
                }
            )
        except Exception:
            pass
        return current, daily, False
    except Exception as e:
        logger.warning(f"Live fetch failed for {city}: {e} — using cache")
        try:
            cache   = WeatherCache.objects.get(city=city)
            current = json.loads(cache.current_json)
            daily   = json.loads(cache.forecast_json)
            if 'forecast_max_temps' not in daily:
                daily['forecast_max_temps'] = [daily.get('max_temp', 32)] * 16
                daily['forecast_avg_temps'] = [daily.get('avg_temp', 28)] * 16
                daily['forecast_hums']      = [daily.get('humidity', 65)] * 16
                daily['forecast_winds']     = [daily.get('wind_speed', 5)] * 16
                daily['forecast_precip']    = [0] * 16
            return current, daily, True
        except WeatherCache.DoesNotExist:
            current = {
                "city": city, "country": "IN", "current_temp": "--",
                "feels_like": "--", "temp_min": "--", "temp_max": "--",
                "humidity": 60, "wind_speed": 5.0, "description": "unavailable", "clouds": 0,
            }
            daily = {
                "avg_temp": 28, "min_temp": 22, "max_temp": 34,
                "humidity": 65, "precipitation": 0, "wind_speed": 5.0,
                "forecast_max_temps": [32] * 16, "forecast_avg_temps": [28] * 16,
                "forecast_min_temps": [22] * 16, "forecast_hums":      [65] * 16,
                "forecast_winds":     [5]  * 16, "forecast_precip":    [0]  * 16,
            }
            return current, daily, True


# ══════════════════════════════════════════════════════════════════════════════
# MONTHLY CLIMATOLOGY — fallback when XGBoost pkl not present
# ══════════════════════════════════════════════════════════════════════════════

_monthly_climatology: dict = {}


def _load_monthly_climatology() -> dict:
    """
    Build per-month climatological norms from HistoricalData.csv (Mumbai, 4,227 rows).
    Used as fallback when XGBoost model is not available.
    """
    global _monthly_climatology
    if _monthly_climatology:
        return _monthly_climatology
    try:
        hist_path = os.path.join(BASE_DIR, 'Dataset', 'HistoricalData.csv')
        df = pd.read_csv(hist_path)
        df['date']  = pd.to_datetime(df['observation'], format='%d-%m-%Y', errors='coerce')
        df = df.dropna(subset=['date'])
        df['month'] = df['date'].dt.month
        temp_col = 'tempC_avg\n(0C)'
        hum_col  = 'Relative humidity_avg\n(%)'
        wind_col = 'windspeedKmph_avg\n(Km/h)'
        for m in range(1, 13):
            sub = df[df['month'] == m]
            if len(sub) < 5:
                continue
            _monthly_climatology[m] = {
                'temp_mean': round(float(sub[temp_col].mean()), 1),
                'temp_std':  round(float(sub[temp_col].std()),  1),
                'hum_mean':  round(float(sub[hum_col].mean()),  1),
                'hum_std':   round(float(sub[hum_col].std()),   1),
                'wind_mean': round(float(sub[wind_col].mean()) / 3.6, 1),
            }
        logger.info(f"Climatology loaded for {len(_monthly_climatology)} months.")
    except Exception as e:
        logger.warning(f"Climatology load failed ({e}) — using India general fallback.")
        _monthly_climatology = {
            1:  {'temp_mean': 23.5, 'temp_std': 4.0, 'hum_mean': 55, 'hum_std': 12, 'wind_mean': 2.0},
            2:  {'temp_mean': 26.0, 'temp_std': 4.5, 'hum_mean': 50, 'hum_std': 12, 'wind_mean': 2.2},
            3:  {'temp_mean': 30.0, 'temp_std': 4.0, 'hum_mean': 45, 'hum_std': 12, 'wind_mean': 2.5},
            4:  {'temp_mean': 32.0, 'temp_std': 3.5, 'hum_mean': 45, 'hum_std': 12, 'wind_mean': 3.0},
            5:  {'temp_mean': 33.0, 'temp_std': 3.5, 'hum_mean': 55, 'hum_std': 14, 'wind_mean': 3.5},
            6:  {'temp_mean': 30.0, 'temp_std': 3.0, 'hum_mean': 75, 'hum_std': 12, 'wind_mean': 4.0},
            7:  {'temp_mean': 28.0, 'temp_std': 2.5, 'hum_mean': 85, 'hum_std': 8,  'wind_mean': 4.5},
            8:  {'temp_mean': 27.5, 'temp_std': 2.0, 'hum_mean': 87, 'hum_std': 7,  'wind_mean': 4.0},
            9:  {'temp_mean': 28.0, 'temp_std': 2.5, 'hum_mean': 82, 'hum_std': 9,  'wind_mean': 3.0},
            10: {'temp_mean': 27.0, 'temp_std': 3.0, 'hum_mean': 70, 'hum_std': 12, 'wind_mean': 2.5},
            11: {'temp_mean': 25.0, 'temp_std': 3.5, 'hum_mean': 60, 'hum_std': 12, 'wind_mean': 2.2},
            12: {'temp_mean': 23.0, 'temp_std': 3.5, 'hum_mean': 58, 'hum_std': 12, 'wind_mean': 2.0},
        }
    return _monthly_climatology


# ══════════════════════════════════════════════════════════════════════════════
# XGBoost FORECAST EXTENSION — days 8-15
# ══════════════════════════════════════════════════════════════════════════════

def _build_xgb_feature_row(temp_win, hum_win, wind_win, precip_win, city_enc, pred_date):
    """
    Build one XGBoost feature row from rolling windows + date features.
    All windows should have most recent value last.
    Feature order matches XGB_FEATURES exactly.
    """
    def safe_get(lst, idx_from_end, default=28.0):
        try:
            return float(lst[-idx_from_end])
        except (IndexError, TypeError):
            return float(default)

    temp_lag1   = safe_get(temp_win,   1, 28.0)
    temp_lag7   = safe_get(temp_win,   7, 28.0)
    temp_lag14  = safe_get(temp_win,  14, 28.0)
    hum_lag1    = safe_get(hum_win,    1, 65.0)
    wind_lag1   = safe_get(wind_win,   1,  3.0)
    precip_lag1 = safe_get(precip_win, 1,  0.0)

    window7     = temp_win[-7:]  if len(temp_win) >= 7  else temp_win
    window30    = temp_win[-30:] if len(temp_win) >= 30 else temp_win
    roll7_mean  = float(np.mean(window7))
    roll7_std   = float(np.std(window7))  if len(window7) > 1 else 0.0
    roll30_mean = float(np.mean(window30))

    month     = pred_date.month
    doy       = pred_date.timetuple().tm_yday
    month_sin = math.sin(2 * math.pi * month / 12)
    month_cos = math.cos(2 * math.pi * month / 12)
    doy_sin   = math.sin(2 * math.pi * doy / 365)
    doy_cos   = math.cos(2 * math.pi * doy / 365)

    temp_min = max(0.0, temp_lag1 - 7.0)
    humidity = max(10.0, min(100.0, hum_lag1))
    wind_max = max(0.0, wind_lag1)

    return pd.DataFrame([[
        temp_lag1, temp_lag7, temp_lag14, hum_lag1, wind_lag1,
        roll7_mean, roll7_std, roll30_mean, precip_lag1,
        month_sin, month_cos, doy_sin, doy_cos, float(city_enc),
        temp_min, humidity, wind_max
    ]], columns=XGB_FEATURES)


def fill_forecast_extended(daily: dict, current_month: int,
                            city_name: str = 'Mumbai', needed: int = 15):
    """
    Days 1-N   : Real Weatherbit API data (accurate, used as-is).
    Days N+1-15: XGBoost chain prediction using days 1-N as seed features.

    XGBoost is the most accurate model from the KisanMausam study —
    lowest RMSE and highest R2 across all 39 cities in the ERA5 dataset.

    Chain logic:
      1. Seed rolling windows (temp/hum/wind) from Weatherbit days 1-N.
         Pad older history (days 0 to -13) by repeating day-1 values.
      2. For each extended day d (N+1 to 15):
           - Build feature row from current rolling windows + date features.
           - XGBoost predicts temp_max for that day.
           - Append prediction to windows → feeds next day's lag features.
         Humidity and wind use 7-day trailing mean from the rolling window
         (XGBoost was trained to predict temp_max only).
      3. City-aware: maps city_name to LabelEncoder int. Cities not in
         the 39-city training set fall back to 'Mumbai' encoding.

    Fallback: if XGBoost pkl not found in Dataset/ → climatological blending.

    Returns: (filled_daily_dict, wbit_available_days: int)
    """
    max_temps = list(daily.get('forecast_max_temps', [daily.get('max_temp', 30)]))
    hums      = list(daily.get('forecast_hums',      [daily.get('humidity', 65)]))
    winds     = list(daily.get('forecast_winds',     [daily.get('wind_speed', 5)]))
    avg_temps = list(daily.get('forecast_avg_temps', [daily.get('avg_temp', 28)]))
    min_temps = list(daily.get('forecast_min_temps', [daily.get('min_temp', 22)]))
    precips   = list(daily.get('forecast_precip',    [0.0]))

    available = min(len(max_temps), len(hums), len(winds))

    if available >= needed:
        return daily, needed  # Weatherbit gave enough — no extension needed

    # ── Try XGBoost extension ─────────────────────────────────────────────────
    xgb_cache = _load_xgb_model()
    xgb_model = xgb_cache.get('model')
    le_cities  = xgb_cache.get('le')

    if xgb_model is not None and le_cities is not None:
        try:
            # Resolve city encoding — fallback to Mumbai if not in training set
            city_clean = city_name.strip().title()
            if city_clean in le_cities.classes_:
                city_enc = int(le_cities.transform([city_clean])[0])
            else:
                city_enc = int(le_cities.transform(['Mumbai'])[0])
                logger.info(
                    f"City '{city_clean}' not in XGB encoder "
                    f"— using Mumbai encoding as proxy"
                )

            # Build seed windows: pad 14 older days by repeating day-0 value
            # so temp_lag7 and temp_lag14 are always available for day 8
            pad_temp   = [max_temps[0]] * 14
            pad_hum    = [hums[0]]      * 14
            pad_wind   = [winds[0]]     * 14
            pad_precip = [0.0]          * 14

            temp_win   = pad_temp   + max_temps[:available]
            hum_win    = pad_hum    + hums[:available]
            wind_win   = pad_wind   + winds[:available]
            precip_win = pad_precip + precips[:available]

            tz  = pytz.timezone("Asia/Kolkata")
            now = datetime.now(tz)

            remaining = needed - available
            for i in range(remaining):
                pred_date = now + timedelta(days=available + i + 1)

                X_row  = _build_xgb_feature_row(
                    temp_win, hum_win, wind_win, precip_win,
                    city_enc, pred_date
                )
                pred_t = float(xgb_model.predict(X_row)[0])
                pred_t = round(max(0.0, min(55.0, pred_t)), 1)

                # Humidity and wind: use 7-day trailing mean from rolling window
                # (secondary variables — XGBoost predicts temp_max only)
                pred_h = round(float(np.mean(hum_win[-7:])), 1)
                pred_h = max(10.0, min(100.0, pred_h))
                pred_w = round(float(np.mean(wind_win[-7:])), 1)
                pred_w = max(0.0, min(30.0, pred_w))

                # Update rolling windows with this prediction
                temp_win.append(pred_t)
                hum_win.append(pred_h)
                wind_win.append(pred_w)
                precip_win.append(0.0)

                # Append to daily forecast arrays
                max_temps.append(pred_t)
                avg_temps.append(round(pred_t - 3.5, 1))
                min_temps.append(round(pred_t - 7.0, 1))
                hums.append(pred_h)
                winds.append(pred_w)
                precips.append(0.0)

            daily['forecast_max_temps'] = max_temps[:needed]
            daily['forecast_avg_temps'] = avg_temps[:needed]
            daily['forecast_min_temps'] = min_temps[:needed]
            daily['forecast_hums']      = hums[:needed]
            daily['forecast_winds']     = winds[:needed]
            daily['forecast_precip']    = precips[:needed]

            logger.info(
                f"XGBoost extended forecast: {available} Weatherbit days + "
                f"{remaining} XGB-predicted days for {city_name} "
                f"(city_enc={city_enc})"
            )
            return daily, available

        except Exception as e:
            logger.error(
                f"XGBoost chain prediction failed for {city_name} ({e}) "
                f"— falling back to climatological blending"
            )

    # ── Fallback: climatological blending ────────────────────────────────────
    clim = _load_monthly_climatology().get(current_month, {
        'temp_mean': 30.0, 'temp_std': 4.0,
        'hum_mean':  65.0, 'hum_std':  12.0,
        'wind_mean': 3.0,
    })

    raw_temp_bias  = float(np.mean(max_temps[:available])) - clim['temp_mean'] if max_temps else 0.0
    raw_hum_bias   = float(np.mean(hums[:available]))      - clim['hum_mean']  if hums      else 0.0
    city_temp_bias = max(-1.5 * clim['temp_std'], min(1.5 * clim['temp_std'], raw_temp_bias))
    city_hum_bias  = max(-1.5 * clim['hum_std'],  min(1.5 * clim['hum_std'],  raw_hum_bias))
    target_temp    = round(clim['temp_mean'] + city_temp_bias, 1)
    target_hum     = round(clim['hum_mean']  + city_hum_bias,  1)
    target_wind    = clim['wind_mean']
    anchor_temp    = max_temps[-1] if max_temps else target_temp
    anchor_hum     = hums[-1]      if hums      else target_hum
    anchor_wind    = winds[-1]     if winds      else target_wind

    remaining = needed - available
    for i in range(remaining):
        alpha  = (i + 1) / remaining
        pred_t = round(anchor_temp + alpha * (target_temp - anchor_temp), 1)
        pred_h = round(anchor_hum  + alpha * (target_hum  - anchor_hum),  1)
        pred_w = round(anchor_wind + alpha * (target_wind - anchor_wind), 1)
        max_temps.append(max(0.0,   min(55.0,  pred_t)))
        hums.append(max(10.0,       min(100.0, pred_h)))
        winds.append(max(0.0,       min(30.0,  pred_w)))
        avg_temps.append(round(pred_t - 3.5, 1))
        min_temps.append(round(pred_t - 7.0, 1))
        precips.append(0.0)

    daily['forecast_max_temps'] = max_temps[:needed]
    daily['forecast_avg_temps'] = avg_temps[:needed]
    daily['forecast_min_temps'] = min_temps[:needed]
    daily['forecast_hums']      = hums[:needed]
    daily['forecast_winds']     = winds[:needed]
    daily['forecast_precip']    = precips[:needed]

    logger.info(
        f"Climatological fallback for {city_name}: "
        f"{available} Weatherbit + {remaining} climatological days "
        f"(bias {city_temp_bias:+.1f}C, target {target_temp}C)"
    )
    return daily, available


# ══════════════════════════════════════════════════════════════════════════════
# AGRICULTURE IMPACT
# ══════════════════════════════════════════════════════════════════════════════

def compute_ag_impact(daily, forecast_dates=None, farm_size_acres=1.4):
    """
    Four advisory tiles — all genuinely city-specific:

    Tile 1 — Peak Forecast Temp:  max of 15-day forecast with the date it occurs.
    Tile 2 — Heat Stress Days:    days where daily HIGH > 34C (ICAR threshold).
    Tile 3 — Disease Risk Days:   days where humidity > 82%.
    Tile 4 — Avg Forecast Temp:   15-day seasonal thermal mean (different per city).
    """
    max_temps  = daily.get('forecast_max_temps', [daily.get('max_temp', 32)] * 15)
    humidities = daily.get('forecast_hums',      [daily.get('humidity', 65)] * 15)
    dates      = forecast_dates or []

    temps15 = max_temps[:15]
    hums15  = humidities[:15]

    # Tile 1: Peak forecast temperature
    if temps15:
        peak_idx  = int(np.argmax(temps15))
        peak_temp = round(float(temps15[peak_idx]), 1)
        peak_date = dates[peak_idx] if peak_idx < len(dates) else f"Day {peak_idx + 1}"
    else:
        peak_temp = round(float(daily.get('max_temp', 32)), 1)
        peak_date = dates[0] if dates else "Day 1"

    # Tile 2: Heat stress days
    heat_days = sum(1 for t in temps15 if t > 34.0)

    # Tile 3: Disease risk days (humidity > 82%)
    disease_risk_days = sum(1 for h in hums15 if h > 82.0)
    if disease_risk_days == 0:
        risk_label = "Low risk"
    elif disease_risk_days <= 3:
        risk_label = "Moderate risk"
    elif disease_risk_days <= 7:
        risk_label = "High risk"
    else:
        risk_label = "Very high risk"

    return {
        "avg_temp":           round(float(np.mean(temps15)), 1),
        "avg_humidity":       round(float(np.mean(hums15)), 1),
        "heat_stress_days":   heat_days,
        "high_humidity_days": disease_risk_days,
        "disease_risk_days":  disease_risk_days,
        "disease_risk_label": risk_label,
        "disease_risk_score": min(disease_risk_days * 10, 100),
        "peak_temp":          peak_temp,
        "peak_temp_date":     peak_date,
        "avg_forecast_temp":  round(float(np.mean(temps15)), 1),
        "savings_pct":        round(min(heat_days * 2.5, 25.0), 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SUPABASE SYNC
# ══════════════════════════════════════════════════════════════════════════════

def sync_to_supabase(table: str, data: dict):
    if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_KEY:
        return
    try:
        headers = {
            "apikey":        settings.SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {settings.SUPABASE_SERVICE_KEY}",
            "Content-Type":  "application/json",
            "Prefer":        "resolution=merge-duplicates",
        }
        requests.post(
            f"{settings.SUPABASE_URL}/rest/v1/{table}",
            json=data, headers=headers, timeout=3
        )
    except Exception as e:
        logger.warning(f"Supabase sync failed ({table}): {e}")


# ══════════════════════════════════════════════════════════════════════════════
# CROP RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════

def _get_crop_recs(daily, profile, now):
    avg_temps = daily.get('forecast_avg_temps', [daily['avg_temp']] * 16)
    avg_hums  = daily.get('forecast_hums',      [daily['humidity']] * 16)

    all_recs = CropRecommender().recommend(
        avg_forecast_temp     = float(np.mean(avg_temps)),
        avg_forecast_humidity = float(np.mean(avg_hums)),
        soil_pH               = profile.soil_ph       if profile and profile.soil_ph       else 6.83,
        soil_moisture_pct     = profile.soil_moisture if profile and profile.soil_moisture else 45.0,
        current_month         = now.month,
        state                 = profile.state         if profile and profile.state         else '',
        top_n                 = 11,
    )

    user_crop_names = {c.lower() for c in (profile.crops if profile else [])}
    user_recs  = [r for r in all_recs if r.crop.lower() in user_crop_names]
    other_recs = [r for r in all_recs if r.crop.lower() not in user_crop_names]
    pinned     = min(len(user_recs), 2)
    return (user_recs[:pinned] + other_recs[:max(0, 5 - pinned)])[:5]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_city(profile):
    """
    Resolve the correct city for a farmer.
    Fixes the stale-Mumbai bug: if city is 'Mumbai' but state is not Maharashtra,
    return the state's correct default city instead.
    """
    state = profile.state or ''
    city  = profile.city  or ''
    if not city or (city == 'Mumbai' and state and state != 'Maharashtra'):
        city = STATE_DEFAULT_CITY.get(state, 'Mumbai')
    return city


# ══════════════════════════════════════════════════════════════════════════════
# VIEWS
# ══════════════════════════════════════════════════════════════════════════════

def login_view(request):
    """Login page — no login required."""
    from django.contrib.auth import authenticate as auth_authenticate
    if request.user.is_authenticated:
        return redirect('dashboard')
    error = None
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '').strip()
        if not username or not password:
            error = "Username and password are required."
        else:
            user = auth_authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('dashboard')
            else:
                if User.objects.filter(username=username).exists():
                    error = "Incorrect password. Please try again."
                else:
                    error = "No account found with that username. Please check or create a new account."
    return render(request, 'login.html', {'mode': 'login', 'error': error})


def register_view(request):
    """Registration — no login required."""
    if request.user.is_authenticated:
        return redirect('dashboard')
    error = None
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '').strip()
        # Field name in login.html is confirm_password
        confirm  = request.POST.get('confirm_password', '').strip()
        name     = request.POST.get('name', '').strip()
        role     = request.POST.get('role', 'farmer')
        if not username or not password:
            error = "Username and password are required."
        elif len(password) < 8:
            error = "Password must be at least 8 characters long."
        elif password != confirm:
            error = "Passwords do not match. Please re-enter your password carefully."
        elif User.objects.filter(username=username).exists():
            error = "This username is already taken. Please choose another."
        else:
            user = User.objects.create_user(
                username=username, password=password, first_name=name)
            FarmerProfile.objects.create(user=user, role=role, city='')
            sync_to_supabase('farmers', {
                'django_user_id': user.id, 'username': username,
                'name': name, 'role': role,
                'created_at': datetime.utcnow().isoformat(),
            })
            login(request, user)
            return redirect('onboarding')
    return render(request, 'login.html', {'mode': 'register', 'error': error})


def onboarding_view(request):
    """Onboarding wizard — must be logged in."""
    if not request.user.is_authenticated:
        return redirect('login')
    return render(request, 'onboarding.html')


@login_required
def save_profile(request):
    """AJAX — saves onboarding wizard data. Always updates city from state."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    profile, _ = FarmerProfile.objects.get_or_create(user=request.user)
    state = data.get('state', profile.state) or ''
    profile.state = state

    provided_city = data.get('city', '').strip()
    if provided_city:
        profile.city = provided_city
    else:
        profile.city = STATE_DEFAULT_CITY.get(state, 'Mumbai') if state else 'Mumbai'

    profile.crops      = data.get('crops',      profile.crops)
    profile.crop_stage = data.get('stage',       profile.crop_stage)
    profile.farm_size  = data.get('sizeAcres',   profile.farm_size)
    profile.irrigation = data.get('irrigation',  profile.irrigation)
    profile.soil_type  = data.get('soilType',    profile.soil_type)
    if data.get('soilPH')    is not None: profile.soil_ph       = float(data['soilPH'])
    if data.get('soilMoist') is not None: profile.soil_moisture = float(data['soilMoist'])
    profile.save()

    sync_to_supabase('farmers', {
        'django_user_id': request.user.id, 'username': request.user.username,
        'state': profile.state, 'city': profile.city,
        'crops': json.dumps(profile.crops), 'crop_stage': profile.crop_stage,
        'farm_size_acres': profile.farm_size, 'irrigation': profile.irrigation,
        'soil_type': profile.soil_type, 'soil_ph': profile.soil_ph,
        'soil_moisture': profile.soil_moisture,
        'updated_at': datetime.utcnow().isoformat(),
    })
    return JsonResponse({'status': 'ok'})


@login_required
def update_soil(request):
    """AJAX — chatbot soil pH / moisture update."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)
    data  = json.loads(request.body)
    ph    = data.get('soilPH')
    moist = data.get('soilMoist')
    profile, _ = FarmerProfile.objects.get_or_create(user=request.user)
    if ph    is not None: profile.soil_ph       = float(ph)
    if moist is not None: profile.soil_moisture = float(moist)
    profile.save()

    city = _resolve_city(profile)
    _, daily, _ = get_weather_with_fallback(city)
    now   = datetime.now(pytz.timezone("Asia/Kolkata"))
    crops = _get_crop_recs(daily, profile, now)

    return JsonResponse({
        'status': 'ok', 'soilPH': profile.soil_ph, 'soilMoist': profile.soil_moisture,
        'crops': [{'crop': r.crop, 'score': r.score, 'season': r.sow_window,
                   'water': r.water_need, 'yield': r.avg_yield_t_ha,
                   'msp': r.msp_inr, 'revenue': r.projected_revenue_inr,
                   'warnings': r.warnings} for r in crops],
    })


@login_required
def get_weather(request):
    """AJAX — city search from nav bar."""
    city = request.GET.get('city', '').strip()
    if not city:
        return JsonResponse({'error': 'City required'}, status=400)

    profile = getattr(request.user, 'profile', None)
    models  = get_trained_models()
    if not models:
        return JsonResponse({'error': 'Model unavailable'}, status=503)

    current, daily, from_cache = get_weather_with_fallback(city)
    tz  = pytz.timezone("Asia/Kolkata")
    now = datetime.now(tz)

    # XGBoost extension for days 8-15, city-aware
    daily, wbit_days = fill_forecast_extended(
        daily, current_month=now.month, city_name=city)

    rain_pred = False
    try:
        input_row = pd.DataFrame([{
            "MinTemp": daily["min_temp"], "MaxTemp": daily["max_temp"],
            "Humidity": daily["humidity"], "Temp": daily["avg_temp"],
            "WindSpeed": daily["wind_speed"],
        }])
        rain_pred = bool(models['rain'].predict(input_row)[0])
    except Exception as e:
        logger.error(f"Rain classifier failed: {e}")

    dates    = [(now + timedelta(days=i)).strftime("%d %b") for i in range(15)]
    wbit_max = daily.get('forecast_max_temps', [daily['max_temp']] * 15)
    wbit_hum = daily.get('forecast_hums',      [daily['humidity']] * 15)
    wbit_wnd = daily.get('forecast_winds',     [daily['wind_speed']] * 15)

    farm_size = profile.farm_size if profile else 1.4
    alerts = AlertEngine().generate_alerts(
        future_temps      = wbit_max[:15],
        future_humidities = wbit_hum[:15],
        future_windspeeds = wbit_wnd[:15],
        rain_prediction   = rain_pred,
        future_dates      = dates,
        current_month     = now.month,
        current_temp      = float(current.get("current_temp") or daily.get("avg_temp", 28)),
        current_humidity  = float(current.get("humidity")     or daily.get("humidity", 65)),
    )

    crops = _get_crop_recs(daily, profile, now)
    ag    = compute_ag_impact(daily, forecast_dates=dates, farm_size_acres=farm_size or 1.4)

    return JsonResponse({
        'from_cache':    from_cache,
        'current':       current,
        'daily':         daily,
        'day_min_temp':  daily.get('min_temp', '--'),
        'day_max_temp':  daily.get('max_temp', '--'),
        'wbit_days':     wbit_days,
        'dates':         dates,
        'temps':         [round(t, 1) for t in wbit_max[:15]],
        'humidities':    [round(h, 1) for h in wbit_hum[:15]],
        'winds':         [round(w, 1) for w in wbit_wnd[:15]],
        'rain_pred':     rain_pred,
        'ag':            ag,
        'peak_risk_day': alerts.risk_summary.get('peak_risk_day', 'Stable'),
        'alerts': [{'type': a.alert_type, 'sev': a.severity, 'msg': a.message,
                    'action': a.crop_action, 'day': a.day_label}
                   for a in alerts.all_alerts_sorted],
        'crops': [{'crop': r.crop, 'score': r.score, 'season': r.sow_window,
                   'water': r.water_need, 'yield': r.avg_yield_t_ha,
                   'msp': r.msp_inr, 'revenue': r.projected_revenue_inr,
                   'warn': r.warnings} for r in crops],
    })


@login_required
def dashboard(request):
    profile, created = FarmerProfile.objects.get_or_create(user=request.user)

    if created or not profile.state:
        return redirect('onboarding')

    # City resolution — fixes stale-Mumbai bug
    url_city = request.GET.get('city', '').strip()
    if url_city:
        city = url_city
        profile.city = city
        profile.save(update_fields=['city'])
    else:
        city = _resolve_city(profile)
        if profile.city != city:
            profile.city = city
            profile.save(update_fields=['city'])

    models  = get_trained_models()
    current, daily, from_cache = get_weather_with_fallback(city)

    # XGBoost extension for days 8-15, city-aware
    tz  = pytz.timezone("Asia/Kolkata")
    now = datetime.now(tz)
    daily, wbit_days = fill_forecast_extended(
        daily, current_month=now.month, city_name=city)

    rain_pred = False
    try:
        input_row = pd.DataFrame([{
            "MinTemp": daily["min_temp"], "MaxTemp": daily["max_temp"],
            "Humidity": daily["humidity"], "Temp": daily["avg_temp"],
            "WindSpeed": daily["wind_speed"],
        }])
        rain_pred = bool(models['rain'].predict(input_row)[0])
    except Exception as e:
        logger.error(f"Dashboard rain prediction error: {e}")

    dates    = [(now + timedelta(days=i)).strftime("%d %b") for i in range(15)]
    wbit_max = daily.get('forecast_max_temps', [daily['max_temp']] * 15)
    wbit_hum = daily.get('forecast_hums',      [daily['humidity']] * 15)
    wbit_wnd = daily.get('forecast_winds',     [daily['wind_speed']] * 15)

    alerts = AlertEngine().generate_alerts(
        future_temps      = wbit_max[:15],
        future_humidities = wbit_hum[:15],
        future_windspeeds = wbit_wnd[:15],
        rain_prediction   = rain_pred,
        future_dates      = dates,
        current_month     = now.month,
        current_temp      = float(current.get("current_temp") or daily.get("avg_temp", 28)),
        current_humidity  = float(current.get("humidity")     or daily.get("humidity", 65)),
    )

    crops = _get_crop_recs(daily, profile, now)
    ag    = compute_ag_impact(daily, forecast_dates=dates,
                               farm_size_acres=profile.farm_size or 1.4)

    forecast_days = [{
        'date': dates[i],
        'temp': round(wbit_max[i], 1) if i < len(wbit_max) else '--',
        'hum':  round(wbit_hum[i], 1) if i < len(wbit_hum) else '--',
    } for i in range(15)]

    greeting_hour = now.hour
    greeting = ('Good morning'   if greeting_hour < 12
                else 'Good afternoon' if greeting_hour < 17
                else 'Good evening')

    ctx = {
        'profile':       profile,
        'farmer_name':   request.user.first_name or request.user.username,
        'farmer_crops':  profile.crops,
        'greeting':      greeting,
        'city':          city,
        'current':       current,
        'daily':         daily,
        'rain_pred':     rain_pred,
        'from_cache':    from_cache,
        'forecast_days': forecast_days,
        'forecast_json': json.dumps(forecast_days),
        'alert_report':  alerts,
        'crop_recs':     crops,
        'ag':            ag,
        'soil_ph':       profile.soil_ph       or 6.83,
        'soil_moisture': profile.soil_moisture or 45.0,
        'today':         now.strftime("%B %Y"),
        'season':        'Kharif' if now.month in range(6, 12) else 'Rabi',
        'wbit_days':     wbit_days,
        'day_min_temp':  daily.get('min_temp', current.get('temp_min', '--')),
        'day_max_temp':  daily.get('max_temp', current.get('temp_max', '--')),
    }
    return render(request, 'weather.html', ctx)


def process_message(request):
    """
    Chatbot — builds live context (city, current weather, 7-day forecast,
    profile data) and passes it to FarmAssistant.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request'}, status=400)

    msg = request.POST.get('message', '').strip()
    if not msg:
        return JsonResponse({'response': 'Please type your question.'})

    context = {}
    try:
        profile = (getattr(request.user, 'profile', None)
                   if request.user.is_authenticated else None)
        city    = _resolve_city(profile) if profile else 'Mumbai'

        current, daily, _ = get_weather_with_fallback(city)

        rain_pred = False
        try:
            models = get_trained_models()
            if models:
                input_row = pd.DataFrame([{
                    "MinTemp":   daily["min_temp"],
                    "MaxTemp":   daily["max_temp"],
                    "Humidity":  daily["humidity"],
                    "Temp":      daily["avg_temp"],
                    "WindSpeed": daily["wind_speed"],
                }])
                rain_pred = bool(models['rain'].predict(input_row)[0])
        except Exception:
            pass

        tz    = pytz.timezone("Asia/Kolkata")
        now   = datetime.now(tz)
        dates = [(now + timedelta(days=i)).strftime("%d %b") for i in range(7)]

        context = {
            'city':           city,
            'state':          profile.state          if profile else '',
            'current_temp':   current.get('current_temp', '--'),
            'feels_like':     current.get('feels_like',   '--'),
            'humidity':       current.get('humidity',      '--'),
            'wind_speed':     current.get('wind_speed',    '--'),
            'description':    current.get('description',   ''),
            'temp_min':       daily.get('min_temp',  current.get('temp_min', '--')),
            'temp_max':       daily.get('max_temp',  current.get('temp_max', '--')),
            'rain_pred':      rain_pred,
            'forecast_temps': daily.get('forecast_max_temps', [])[:7],
            'forecast_hums':  daily.get('forecast_hums',      [])[:7],
            'forecast_dates': dates,
            'soil_ph':        profile.soil_ph       if profile else None,
            'soil_moisture':  profile.soil_moisture if profile else None,
            'current_month':  now.month,
        }
    except Exception as e:
        logger.error(f"Chatbot context build failed: {e}")
        context = {}

    resp = farm_bot.process_query(msg, context)
    return JsonResponse({'response': resp})


# ══════════════════════════════════════════════════════════════════════════════
# USER FLOW
# ══════════════════════════════════════════════════════════════════════════════

@login_required
def profile_edit(request):
    profile, _ = FarmerProfile.objects.get_or_create(user=request.user)
    if request.method == 'POST':
        state = request.POST.get('state', profile.state).strip()
        city  = (request.POST.get('city', '').strip()
                 or STATE_DEFAULT_CITY.get(state, profile.city or 'Mumbai'))
        profile.state      = state
        profile.city       = city
        profile.irrigation = request.POST.get('irrigation', profile.irrigation)
        profile.soil_type  = request.POST.get('soil_type',  profile.soil_type)
        profile.crop_stage = request.POST.get('crop_stage', profile.crop_stage)
        farm_sz = request.POST.get('farm_size')
        if farm_sz:
            try:    profile.farm_size = float(farm_sz)
            except: pass
        ph    = request.POST.get('soil_ph')
        moist = request.POST.get('soil_moisture')
        if ph:
            try:    profile.soil_ph = float(ph)
            except: pass
        if moist:
            try:    profile.soil_moisture = float(moist)
            except: pass
        crops_raw = request.POST.get('crops', '')
        if crops_raw:
            profile.crops = [c.strip() for c in crops_raw.split(',') if c.strip()]
        profile.save()
        sync_to_supabase('farmers', {
            'django_user_id': request.user.id, 'username': request.user.username,
            'state': profile.state, 'city': profile.city,
            'crops': json.dumps(profile.crops), 'crop_stage': profile.crop_stage,
            'farm_size_acres': profile.farm_size, 'irrigation': profile.irrigation,
            'soil_type': profile.soil_type, 'soil_ph': profile.soil_ph,
            'soil_moisture': profile.soil_moisture,
            'updated_at': datetime.utcnow().isoformat(),
        })
        messages.success(request, 'Profile updated successfully.')
        return redirect('profile_edit')

    INDIAN_STATES = list(STATE_DEFAULT_CITY.keys())
    ALL_CROPS = ['Wheat','Rice','Cotton','Soybean','Onion','Tomato',
                 'Chickpea','Groundnut','Maize','Bajra','Sugarcane','Mustard','Potato']
    return render(request, 'profile_edit.html', {
        'profile':        profile,
        'farmer_name':    request.user.first_name or request.user.username,
        'indian_states':  INDIAN_STATES,
        'all_crops':      ALL_CROPS,
        'selected_crops': profile.crops,
    })


@login_required
def password_change(request):
    error = success = None
    if request.method == 'POST':
        old_pw  = request.POST.get('old_password', '')
        new_pw  = request.POST.get('new_password', '')
        confirm = request.POST.get('confirm_password', '')
        if not request.user.check_password(old_pw):
            error = 'Current password is incorrect.'
        elif len(new_pw) < 8:
            error = 'New password must be at least 8 characters.'
        elif new_pw != confirm:
            error = 'New passwords do not match.'
        else:
            request.user.set_password(new_pw)
            request.user.save()
            update_session_auth_hash(request, request.user)
            success = True
    return render(request, 'password_change.html', {
        'farmer_name': request.user.first_name or request.user.username,
        'error': error, 'success': success,
    })


def health_check(request):
    from django.db import connection
    try:
        connection.ensure_connection()
        db_ok = True
    except Exception:
        db_ok = False
    return JsonResponse({
        'status':   'ok' if db_ok else 'db_error',
        'database': 'connected' if db_ok else 'unreachable',
        'version':  '1.0.0',
    })