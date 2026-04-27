# forecast/alert_engine.py
# ─────────────────────────────────────────────────────────────────────────────
# Alert Engine — built directly on the RF Regressor and RF Classifier already
# trained in views.py.  Uses the same feature set your models were trained on:
#   Classifier:  MinTemp, MaxTemp, Humidity, Temp, WindSpeed  → RainTomorrow
#   Regressor:   chained single-step predictions for Temp / Humidity / WindSpeed
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict


# ── Alert severity constants ──────────────────────────────────────────────────
SEVERITY_INFO    = "info"
SEVERITY_WARNING = "warning"
SEVERITY_DANGER  = "danger"

# ── Thresholds derived from Mumbai HistoricalData.csv (2010-2021) statistics ──
# Mean temp ≈ 27 °C, max recorded ≈ 37 °C, monsoon humidity often > 85 %
THRESHOLDS = {
    # Temperature
    "heat_wave":        {"temp_max": 37.0,  "severity": SEVERITY_DANGER,  "crop_action": "Irrigate immediately. Avoid daytime field work. Cover nurseries."},
    "heat_stress":      {"temp_max": 34.0,  "severity": SEVERITY_WARNING, "crop_action": "Schedule irrigation in early morning. Monitor crops for wilting."},
    "cold_wave":        {"temp_min": 12.0,  "severity": SEVERITY_DANGER,  "crop_action": "Cover Rabi seedlings overnight. Delay sowing if sustained."},
    "cold_stress":      {"temp_min": 15.0,  "severity": SEVERITY_WARNING, "crop_action": "Check germination beds. Delay irrigation on cold mornings."},

    # Humidity — high humidity + wet soils → disease outbreak risk
    "flood_risk":       {"humidity": 90.0,  "severity": SEVERITY_DANGER,  "crop_action": "Check drainage channels. Harvest mature crops early if rainfall forecast."},
    "disease_risk":     {"humidity": 82.0,  "severity": SEVERITY_WARNING, "crop_action": "Apply preventive fungicide (Mancozeb/Carbendazim). Improve field aeration."},

    # Precipitation probability
    "heavy_rain":       {"precip_prob": 80.0, "severity": SEVERITY_DANGER,  "crop_action": "Postpone fertiliser application. Secure harvested produce."},
    "moderate_rain":    {"precip_prob": 55.0, "severity": SEVERITY_INFO,    "crop_action": "Good time for sowing if soil is prepared. Hold off on pesticide spraying."},

    # Wind speed (m/s) — >8.3 m/s = strong breeze, >13.9 m/s = near gale
    "strong_wind":      {"wind_speed": 13.9, "severity": SEVERITY_DANGER,  "crop_action": "Stake tall crops. Delay spraying operations."},
    "moderate_wind":    {"wind_speed": 8.3,  "severity": SEVERITY_WARNING, "crop_action": "Avoid aerial spraying. Check support structures for tall crops."},

    # Combined compound event: humidity spike + temp drop in same day
    "disease_compound": {"humidity_delta": 15.0, "temp_delta": -3.0, "severity": SEVERITY_DANGER,
                         "crop_action": "High risk of Bacterial Blight / Anthracnose (detected in your dataset). Apply copper-based fungicide within 24 hours."},
}

# ── Maharashtra agro-climate zone crop calendars ─────────────────────────────
# Maps month → Kharif / Rabi sowing windows and harvest windows
CROP_CALENDAR = {
    "rice":       {"sow": [6, 7],    "harvest": [10, 11], "season": "Kharif"},
    "wheat":      {"sow": [11, 12],  "harvest": [3, 4],   "season": "Rabi"},
    "sorghum":    {"sow": [6, 7],    "harvest": [9, 10],  "season": "Kharif"},
    "cotton":     {"sow": [5, 6],    "harvest": [11, 12], "season": "Kharif"},
    "sugarcane":  {"sow": [2, 3],    "harvest": [11, 12], "season": "Annual"},
    "soybean":    {"sow": [6, 7],    "harvest": [10],     "season": "Kharif"},
    "chickpea":   {"sow": [10, 11],  "harvest": [2, 3],   "season": "Rabi"},
    "groundnut":  {"sow": [6, 7],    "harvest": [10, 11], "season": "Kharif"},
    "onion":      {"sow": [10, 11],  "harvest": [2, 3],   "season": "Rabi"},
    "tomato":     {"sow": [6, 7, 11],"harvest": [9, 10, 2],"season": "Both"},
}


@dataclass
class Alert:
    alert_type: str
    severity: str           # "info" | "warning" | "danger"
    message: str
    crop_action: str
    day_index: int          # 0-based index into the 15-day forecast
    day_label: str
    triggered_values: Dict  # the actual values that fired this alert


@dataclass
class AlertReport:
    alerts: List[Alert] = field(default_factory=list)
    compound_events: List[Alert] = field(default_factory=list)
    risk_summary: Dict = field(default_factory=dict)
    active_crops: List[str] = field(default_factory=list)

    @property
    def danger_count(self):
        return sum(1 for a in self.alerts + self.compound_events if a.severity == SEVERITY_DANGER)

    @property
    def warning_count(self):
        return sum(1 for a in self.alerts + self.compound_events if a.severity == SEVERITY_WARNING)

    @property
    def all_alerts_sorted(self):
        all_a = self.alerts + self.compound_events
        order = {SEVERITY_DANGER: 0, SEVERITY_WARNING: 1, SEVERITY_INFO: 2}
        return sorted(all_a, key=lambda a: (a.day_index, order.get(a.severity, 3)))


class AlertEngine:
    """
    Wraps the RF-predicted 15-day forecast arrays from views.py and generates
    agricultural hazard alerts.  Call generate_alerts() with the same arrays
    that weather_view() already produces.
    """

    def generate_alerts(
        self,
        future_temps: List[float],
        future_humidities: List[float],
        future_windspeeds: List[float],
        rain_prediction: bool,          # RF classifier output for day 1
        future_dates: List[str],
        current_month: int,
        current_temp: float,
        current_humidity: float,
    ) -> AlertReport:

        report = AlertReport()
        report.active_crops = self._active_crops(current_month)

        prev_humidity = current_humidity
        prev_temp     = current_temp

        for i, (temp, humidity, wind, date) in enumerate(
            zip(future_temps, future_humidities, future_windspeeds, future_dates)
        ):
            day_label = f"Day {i+1} ({date})"

            # ── Temperature alerts ────────────────────────────────────────────
            if temp >= THRESHOLDS["heat_wave"]["temp_max"]:
                report.alerts.append(Alert(
                    alert_type="Heat Wave",
                    severity=SEVERITY_DANGER,
                    message=f"Extreme heat forecast: {temp:.1f}°C on {date}",
                    crop_action=THRESHOLDS["heat_wave"]["crop_action"],
                    day_index=i, day_label=day_label,
                    triggered_values={"temp": temp}
                ))
            elif temp >= THRESHOLDS["heat_stress"]["temp_max"]:
                report.alerts.append(Alert(
                    alert_type="Heat Stress",
                    severity=SEVERITY_WARNING,
                    message=f"High temperature: {temp:.1f}°C on {date}",
                    crop_action=THRESHOLDS["heat_stress"]["crop_action"],
                    day_index=i, day_label=day_label,
                    triggered_values={"temp": temp}
                ))

            if temp <= THRESHOLDS["cold_wave"]["temp_min"]:
                report.alerts.append(Alert(
                    alert_type="Cold Wave",
                    severity=SEVERITY_DANGER,
                    message=f"Cold wave risk: {temp:.1f}°C forecast on {date}",
                    crop_action=THRESHOLDS["cold_wave"]["crop_action"],
                    day_index=i, day_label=day_label,
                    triggered_values={"temp": temp}
                ))
            elif temp <= THRESHOLDS["cold_stress"]["temp_min"]:
                report.alerts.append(Alert(
                    alert_type="Cold Stress",
                    severity=SEVERITY_WARNING,
                    message=f"Below-normal temperature: {temp:.1f}°C on {date}",
                    crop_action=THRESHOLDS["cold_stress"]["crop_action"],
                    day_index=i, day_label=day_label,
                    triggered_values={"temp": temp}
                ))

            # ── Humidity alerts ───────────────────────────────────────────────
            if humidity >= THRESHOLDS["flood_risk"]["humidity"]:
                report.alerts.append(Alert(
                    alert_type="Flood / Waterlogging Risk",
                    severity=SEVERITY_DANGER,
                    message=f"Critically high humidity: {humidity:.1f}% on {date}. Heavy rainfall likely.",
                    crop_action=THRESHOLDS["flood_risk"]["crop_action"],
                    day_index=i, day_label=day_label,
                    triggered_values={"humidity": humidity}
                ))
            elif humidity >= THRESHOLDS["disease_risk"]["humidity"]:
                report.alerts.append(Alert(
                    alert_type="Crop Disease Risk",
                    severity=SEVERITY_WARNING,
                    message=f"High humidity ({humidity:.1f}%) on {date} — Bacterial Blight / Anthracnose risk elevated",
                    crop_action=THRESHOLDS["disease_risk"]["crop_action"],
                    day_index=i, day_label=day_label,
                    triggered_values={"humidity": humidity}
                ))

            # ── Wind speed alerts ─────────────────────────────────────────────
            if wind >= THRESHOLDS["strong_wind"]["wind_speed"]:
                report.alerts.append(Alert(
                    alert_type="Strong Wind",
                    severity=SEVERITY_DANGER,
                    message=f"Wind speed {wind:.1f} m/s forecast on {date}. Risk of lodging in tall crops.",
                    crop_action=THRESHOLDS["strong_wind"]["crop_action"],
                    day_index=i, day_label=day_label,
                    triggered_values={"wind_speed": wind}
                ))
            elif wind >= THRESHOLDS["moderate_wind"]["wind_speed"]:
                report.alerts.append(Alert(
                    alert_type="Moderate Wind",
                    severity=SEVERITY_WARNING,
                    message=f"Elevated wind speed {wind:.1f} m/s on {date}",
                    crop_action=THRESHOLDS["moderate_wind"]["crop_action"],
                    day_index=i, day_label=day_label,
                    triggered_values={"wind_speed": wind}
                ))

            # ── Compound event: humidity spike + temperature drop ─────────────
            # This mirrors the Bacterial Blight / Anthracnose pattern in your
            # HistoricalData.csv (disease YES rows cluster around humidity spikes
            # following a temperature dip).
            hum_delta  = humidity - prev_humidity
            temp_delta = temp - prev_temp
            if (hum_delta  >= THRESHOLDS["disease_compound"]["humidity_delta"] and
                temp_delta <= THRESHOLDS["disease_compound"]["temp_delta"]):
                report.compound_events.append(Alert(
                    alert_type="Compound Disease Event",
                    severity=SEVERITY_DANGER,
                    message=(f"Humidity jumped +{hum_delta:.1f}% while temp dropped "
                             f"{temp_delta:.1f}°C on {date}. "
                             f"Highest-risk pattern for Bacterial Blight in your dataset."),
                    crop_action=THRESHOLDS["disease_compound"]["crop_action"],
                    day_index=i, day_label=day_label,
                    triggered_values={"hum_delta": hum_delta, "temp_delta": temp_delta}
                ))

            prev_humidity = humidity
            prev_temp     = temp

        # ── Day-1 rain prediction from RF classifier ─────────────────────────
        if rain_prediction:
            report.alerts.insert(0, Alert(
                alert_type="Rain Predicted Tomorrow",
                severity=SEVERITY_WARNING,
                message="RF Classifier predicts rain tomorrow based on current conditions.",
                crop_action="Delay fertiliser/pesticide application. "
                            "Ensure drainage channels are clear.",
                day_index=0, day_label=future_dates[0] if future_dates else "Tomorrow",
                triggered_values={"rf_rain_pred": True}
            ))

        # ── Risk summary dict for template ────────────────────────────────────
        report.risk_summary = {
            "total_alerts":   len(report.alerts) + len(report.compound_events),
            "danger_alerts":  report.danger_count,
            "warning_alerts": report.warning_count,
            "peak_risk_day":  self._peak_risk_day(report),
            "active_crops":   report.active_crops,
        }

        return report

    def _active_crops(self, month: int) -> List[str]:
        """Return crops in sowing or harvest window for the given month."""
        active = []
        for crop, cal in CROP_CALENDAR.items():
            if month in cal["sow"] or month in cal["harvest"]:
                stage = "sowing" if month in cal["sow"] else "harvest"
                active.append(f"{crop.capitalize()} ({stage})")
        return active

    def _peak_risk_day(self, report: AlertReport) -> str:
        danger = [a for a in report.all_alerts_sorted if a.severity == SEVERITY_DANGER]
        if danger:
            return danger[0].day_label
        warning = [a for a in report.all_alerts_sorted if a.severity == SEVERITY_WARNING]
        if warning:
            return warning[0].day_label
        return "None — conditions look stable"
