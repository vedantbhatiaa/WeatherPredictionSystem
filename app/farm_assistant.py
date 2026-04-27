"""
farm_assistant.py — KisanMausam Chatbot
────────────────────────────────────────────────────────────────────────────
v2 enhancements:
  1. process_query(msg, context) — accepts live weather context from views.py
  2. Answers "what's the weather today / this week / forecast" from context
  3. Detects other-city weather queries → live OWM API call
  4. Detects location-specific crop queries → CropRecommender with real weather
  5. All existing rule-based logic fully preserved
────────────────────────────────────────────────────────────────────────────
"""

import re
import os
import requests
from datetime import datetime

# Relative import works fine inside Django app package
try:
    from .crop_recommender import CropRecommender, STATE_RAINFALL_MM
except ImportError:
    CropRecommender = None
    STATE_RAINFALL_MM = {}


# ── City → State mapping (for location-aware crop recs) ──────────────────────
CITY_TO_STATE = {
    'mumbai': 'Maharashtra', 'pune': 'Maharashtra', 'nagpur': 'Maharashtra',
    'nashik': 'Maharashtra', 'aurangabad': 'Maharashtra', 'kolhapur': 'Maharashtra',
    'solapur': 'Maharashtra', 'thane': 'Maharashtra', 'navi mumbai': 'Maharashtra',
    'delhi': 'Other', 'new delhi': 'Other', 'noida': 'Other', 'gurgaon': 'Other',
    'gurugram': 'Other', 'faridabad': 'Other',
    'ludhiana': 'Punjab', 'amritsar': 'Punjab', 'chandigarh': 'Punjab',
    'jalandhar': 'Punjab', 'patiala': 'Punjab', 'bathinda': 'Punjab',
    'hisar': 'Haryana', 'rohtak': 'Haryana', 'karnal': 'Haryana',
    'ambala': 'Haryana', 'panipat': 'Haryana',
    'lucknow': 'Uttar Pradesh', 'agra': 'Uttar Pradesh', 'varanasi': 'Uttar Pradesh',
    'kanpur': 'Uttar Pradesh', 'meerut': 'Uttar Pradesh', 'allahabad': 'Uttar Pradesh',
    'prayagraj': 'Uttar Pradesh', 'bareilly': 'Uttar Pradesh',
    'bhopal': 'Madhya Pradesh', 'indore': 'Madhya Pradesh', 'gwalior': 'Madhya Pradesh',
    'jabalpur': 'Madhya Pradesh', 'ujjain': 'Madhya Pradesh',
    'jaipur': 'Rajasthan', 'jodhpur': 'Rajasthan', 'udaipur': 'Rajasthan',
    'kota': 'Rajasthan', 'ajmer': 'Rajasthan', 'bikaner': 'Rajasthan',
    'ahmedabad': 'Gujarat', 'surat': 'Gujarat', 'vadodara': 'Gujarat',
    'rajkot': 'Gujarat', 'gandhinagar': 'Gujarat', 'bhavnagar': 'Gujarat',
    'bengaluru': 'Karnataka', 'bangalore': 'Karnataka', 'mysuru': 'Karnataka',
    'hubli': 'Karnataka', 'mangaluru': 'Karnataka', 'belgaum': 'Karnataka',
    'chennai': 'Tamil Nadu', 'coimbatore': 'Tamil Nadu', 'madurai': 'Tamil Nadu',
    'trichy': 'Tamil Nadu', 'salem': 'Tamil Nadu', 'tirunelveli': 'Tamil Nadu',
    'hyderabad': 'Telangana', 'warangal': 'Telangana', 'nizamabad': 'Telangana',
    'vijayawada': 'Andhra Pradesh', 'visakhapatnam': 'Andhra Pradesh',
    'guntur': 'Andhra Pradesh', 'tirupati': 'Andhra Pradesh',
    'patna': 'Bihar', 'gaya': 'Bihar', 'muzaffarpur': 'Bihar', 'bhagalpur': 'Bihar',
    'kolkata': 'West Bengal', 'howrah': 'West Bengal', 'durgapur': 'West Bengal',
    'asansol': 'West Bengal', 'siliguri': 'West Bengal',
    'bhubaneswar': 'Odisha', 'cuttack': 'Odisha', 'puri': 'Odisha',
    'shimla': 'Himachal Pradesh', 'manali': 'Himachal Pradesh',
    'dharamshala': 'Himachal Pradesh',
    'dehradun': 'Uttarakhand', 'haridwar': 'Uttarakhand', 'roorkee': 'Uttarakhand',
    'raipur': 'Chhattisgarh', 'bilaspur': 'Chhattisgarh', 'bhilai': 'Chhattisgarh',
    'ranchi': 'Jharkhand', 'jamshedpur': 'Jharkhand', 'dhanbad': 'Jharkhand',
    'guwahati': 'Assam', 'dibrugarh': 'Assam', 'silchar': 'Assam',
    'kochi': 'Kerala', 'thiruvananthapuram': 'Kerala', 'kozhikode': 'Kerala',
    'thrissur': 'Kerala', 'kollam': 'Kerala',
}

STATE_NAMES_LOWER = {
    'maharashtra', 'punjab', 'haryana', 'uttar pradesh', 'madhya pradesh',
    'rajasthan', 'gujarat', 'karnataka', 'tamil nadu', 'andhra pradesh',
    'telangana', 'bihar', 'west bengal', 'odisha', 'himachal pradesh',
    'uttarakhand', 'chhattisgarh', 'jharkhand', 'assam', 'kerala',
}

KNOWN_CITIES = set(CITY_TO_STATE.keys())


class FarmAssistant:
    def __init__(self, owm_key=None, wbit_key=None):
        self.owm_key  = owm_key  or os.getenv('OWM_KEY',  '')
        self.wbit_key = wbit_key or os.getenv('WBIT_KEY', '')

        # ── Crop database ──────────────────────────────────────────────────────
        self.crops = {
            "wheat":     {"soil": ["loamy","clay loam"], "climate": "cool/temperate",
                          "water": "Moderate", "season": "Rabi (sow Nov–Dec)",
                          "ph": "6.0–7.5", "yield": "3.0–4.0 t/ha",
                          "msp": "₹2,275/qtl (2024-25)",
                          "diseases": ["Rust","Smut","Karnal Bunt"],
                          "tips": "Avoid irrigation after 90 days. Terminal heat above 34°C reduces grain weight."},
            "rice":      {"soil": ["clay","clay loam"], "climate": "tropical/humid",
                          "water": "High (needs standing water)", "season": "Kharif (sow Jun–Jul)",
                          "ph": "5.5–6.5", "yield": "2.0–3.0 t/ha",
                          "msp": "₹2,300/qtl (2024-25)",
                          "diseases": ["Blast","Bacterial Blight","Brown Spot","Sheath Blight"],
                          "tips": "Puddle field thoroughly. Transplant 25–30 day seedlings."},
            "cotton":    {"soil": ["black cotton soil","deep loam"], "climate": "tropical/warm",
                          "water": "Moderate", "season": "Kharif (sow May–Jun)",
                          "ph": "6.0–8.0", "yield": "0.4–0.6 t/ha lint",
                          "msp": "₹7,121/qtl (2024-25)",
                          "diseases": ["Bacterial Blight","Fusarium Wilt","Bollworm"],
                          "tips": "Sensitive to waterlogging. Monitor for Bollworm from 45 DAS."},
            "soybean":   {"soil": ["loamy","well-drained"], "climate": "warm/temperate",
                          "water": "Moderate", "season": "Kharif (sow Jun–Jul)",
                          "ph": "6.0–7.0", "yield": "1.2–2.0 t/ha",
                          "msp": "₹4,892/qtl (2024-25)",
                          "diseases": ["Soybean Rust","Root Rot","Yellow Mosaic"],
                          "tips": "Inoculate seeds with Rhizobium. Nitrogen-fixing — reduces fertilizer cost."},
            "tomato":    {"soil": ["loamy","sandy loam"], "climate": "warm",
                          "water": "Moderate", "season": "Sow Jun or Nov",
                          "ph": "5.8–7.0", "yield": "20–30 t/ha",
                          "msp": "₹1,800/qtl (indicative)",
                          "diseases": ["Early Blight","Late Blight","Leaf Curl Virus"],
                          "tips": "Stake plants at 30 cm. Avoid overhead watering — triggers blight."},
            "onion":     {"soil": ["loamy","sandy loam"], "climate": "cool/dry",
                          "water": "Moderate", "season": "Rabi (sow Oct–Nov)",
                          "ph": "6.0–7.0", "yield": "15–25 t/ha",
                          "msp": "₹1,600/qtl (indicative)",
                          "diseases": ["Purple Blotch","Thrips","Downy Mildew"],
                          "tips": "Stop irrigation 2 weeks before harvest. Cure bulbs in shade."},
            "maize":     {"soil": ["loamy","well-drained"], "climate": "warm",
                          "water": "Moderate", "season": "Kharif (sow Jun–Jul)",
                          "ph": "5.8–7.0", "yield": "2.5–4.0 t/ha",
                          "msp": "₹2,090/qtl (2024-25)",
                          "diseases": ["Fall Armyworm","Turcicum Blight","Maize Streak"],
                          "tips": "Critical water periods: knee-high, tasseling, grain fill."},
            "chickpea":  {"soil": ["loamy","sandy loam"], "climate": "cool/dry",
                          "water": "Low (rain-fed mostly)", "season": "Rabi (sow Oct–Nov)",
                          "ph": "6.0–8.0", "yield": "1.0–1.5 t/ha",
                          "msp": "₹5,440/qtl (2024-25)",
                          "diseases": ["Botrytis Gray Mold","Ascochyta Blight","Fusarium Wilt"],
                          "tips": "Drought-tolerant. Excess nitrogen reduces yield. Good Rabi option."},
            "groundnut": {"soil": ["sandy loam","well-drained"], "climate": "warm/tropical",
                          "water": "Moderate", "season": "Kharif (sow Jun–Jul)",
                          "ph": "5.5–7.0", "yield": "1.5–2.5 t/ha",
                          "msp": "₹6,783/qtl (2024-25)",
                          "diseases": ["Tikka Leaf Spot","Stem Rot","Aflatoxin"],
                          "tips": "Calcium important at pegging stage. Avoid waterlogging."},
            "bajra":     {"soil": ["sandy","sandy loam"], "climate": "hot/arid",
                          "water": "Low (drought-tolerant)", "season": "Kharif (sow Jun–Jul)",
                          "ph": "5.5–7.5", "yield": "1.5–2.5 t/ha",
                          "msp": "₹2,625/qtl (2024-25)",
                          "diseases": ["Downy Mildew","Ergot","Blast"],
                          "tips": "Best crop for Rajasthan, Gujarat arid zones. Very drought-tolerant."},
            "sugarcane": {"soil": ["loamy","clay loam"], "climate": "tropical/humid",
                          "water": "High", "season": "Annual (plant Feb–Mar)",
                          "ph": "6.0–7.5", "yield": "60–80 t/ha",
                          "msp": "₹340/qtl FRP (2024-25)",
                          "diseases": ["Red Rot","Smut","Wilt"],
                          "tips": "Ratoon crop saves cost. Trash mulching conserves moisture."},
        }

        # ── Disease database ───────────────────────────────────────────────────
        self.diseases = {
            "bacterial blight": {
                "crops": ["rice","cotton","wheat"],
                "trigger": "Humidity above 82%, warm temperatures, leaf wetness",
                "symptoms": "Water-soaked lesions on leaves that turn yellow then brown. Wilting of young shoots.",
                "prevention": "Use resistant varieties. Avoid overhead irrigation. Maintain field hygiene.",
                "treatment": "Spray Copper Oxychloride (0.3%) or Streptomycin Sulphate (0.015%). Repeat in 10 days.",
            },
            "blast": {
                "crops": ["rice","bajra","wheat"],
                "trigger": "Humidity >85%, temperature 25–28°C, cloudy conditions",
                "symptoms": "Diamond-shaped lesions with grey centre on leaves. Neck blast causes ear to fall over.",
                "prevention": "Seed treatment with Carbendazim. Balanced nitrogen — avoid excess.",
                "treatment": "Spray Tricyclazole (0.06%) or Isoprothiolane (1.5 ml/litre) at first sign.",
            },
            "rust": {
                "crops": ["wheat","barley"],
                "trigger": "Cool, moist weather; temperatures 15–22°C",
                "symptoms": "Reddish-brown or yellow powdery pustules on leaves and stems.",
                "prevention": "Resistant varieties. Early sowing to avoid peak rust season.",
                "treatment": "Spray Propiconazole (0.1%) or Mancozeb (0.2%). 2 sprays, 15 days apart.",
            },
            "anthracnose": {
                "crops": ["soybean","chickpea","cotton","groundnut"],
                "trigger": "Humid conditions, warm weather, post-monsoon period",
                "symptoms": "Dark sunken spots on pods, stems and leaves. Premature defoliation.",
                "prevention": "Crop rotation. Destroy infected debris. Avoid excessive moisture.",
                "treatment": "Carbendazim (1g/litre) or Mancozeb + Carbendazim combination spray.",
            },
            "fusarium wilt": {
                "crops": ["cotton","chickpea","tomato","groundnut"],
                "trigger": "Soil-borne fungus, warm soil, poor drainage",
                "symptoms": "Sudden wilting of plants. Brown discolouration of vascular tissue. Plant death.",
                "prevention": "Resistant varieties. Crop rotation (3+ years). Soil solarisation.",
                "treatment": "No effective chemical cure. Remove and destroy affected plants. Improve drainage.",
            },
            "early blight": {
                "crops": ["tomato","potato"],
                "trigger": "Alternating wet and dry periods, warm temperatures 24–29°C",
                "symptoms": "Dark brown spots with concentric rings (target-board appearance) on lower leaves.",
                "prevention": "Avoid wetting foliage. Remove infected leaves. Proper spacing.",
                "treatment": "Spray Mancozeb (2g/litre) or Chlorothalonil every 7–10 days.",
            },
            "downy mildew": {
                "crops": ["bajra","onion","grapes"],
                "trigger": "Cool, humid weather; humidity >80%, temperatures 10–20°C",
                "symptoms": "Yellow patches on upper leaf surface. Purple/grey fungal growth underneath.",
                "prevention": "Resistant varieties. Avoid dense planting. Good air circulation.",
                "treatment": "Metalaxyl + Mancozeb (0.25%) spray. Apply before symptoms appear in high-risk weather.",
            },
        }

        # ── Pest database ──────────────────────────────────────────────────────
        self.pests = {
            "bollworm": "Cotton bollworm: pink and American bollworm damage bolls. Use pheromone traps for monitoring. Spray Emamectin Benzoate (0.4 ml/litre) or Spinosad when infestation crosses ETL.",
            "fall armyworm": "FAW attacks maize at whorl stage. Check plants early morning. Spray Spinetoram 11.7% SC (0.5 ml/litre) or Chlorantraniliprole into whorl. Pheromone traps for early detection.",
            "thrips": "Sucking pest on onion and chilli. Causes silvering of leaves. Spray Fipronil (1.5 ml/litre) or Spinosad. Avoid waterlogging which worsens thrips damage.",
            "aphids": "Cluster on tender shoots, secrete honeydew leading to sooty mould. Spray Imidacloprid (0.5 ml/litre) or Dimethoate. Yellow sticky traps help monitor population.",
            "stem borer": "Rice and maize stem borer causes dead hearts and white ears. Spray Cartap Hydrochloride (1.5g/litre) or Chlorpyrifos at 1 ml/litre during early infestation.",
            "whitefly": "Transmits viral diseases in cotton and tomato. Yellow sticky traps + Acetamiprid (0.2g/litre) spray. Remove weed hosts. Avoid broad-spectrum insecticides early season.",
        }

        # ── Weather interpretation ─────────────────────────────────────────────
        self.weather_advice = {
            "rain":     "If rain is predicted: delay all fertiliser and pesticide spraying by 24–48 hours. Clear drainage channels. For harvested produce, move to covered storage immediately.",
            "heat":     "Heat stress above 34°C: irrigate early morning (5–7 AM) to cool root zone. Monitor wheat and vegetable crops for wilting. Mulching reduces soil temperature significantly.",
            "wind":     "Strong winds above 35 km/h: stake tall crops (maize, sorghum, vegetables). Avoid aerial spraying. Secure shade nets and poly tunnels.",
            "humidity": "High humidity (above 82%): high risk of fungal diseases. Apply preventive fungicide Mancozeb/Carbendazim. Improve field aeration by thinning dense canopy.",
            "fog":      "Dense fog reduces photosynthesis and promotes disease. Avoid irrigation. Spray Potassium Nitrate (1%) to reduce frost/cold damage in Rabi crops.",
            "drought":  "Extended dry spell: prioritise irrigation for flowering/grain-fill stage crops. Mulch to reduce evaporation. Consider drought-tolerant varieties next season.",
            "flood":    "Post-flood: drain fields immediately. Don't apply urea to waterlogged soil. Spray Mancozeb preventively once water recedes. Check for root rot.",
        }

        # ── Soil advice ────────────────────────────────────────────────────────
        self.soil_advice = {
            "black":   "Black cotton soil (Vertisol) has high clay content and nutrient-rich. It shrinks when dry and swells when wet. Avoid deep tillage when wet. Good for cotton, soybean, wheat.",
            "red":     "Red laterite soil is acidic and low in nutrients. Add lime to raise pH. Apply FYM 10–15 t/ha before sowing. Good for groundnut, millets, pulses.",
            "alluvial":"Alluvial soil is fertile and well-drained. Found in UP, Punjab river plains. Excellent for wheat, rice, sugarcane, vegetables. Maintain organic matter with crop residue incorporation.",
            "sandy":   "Sandy soil drains fast and has low nutrient retention. Add organic compost 10 t/ha. Mulching reduces water loss. More frequent, lighter irrigation needed. Good for groundnut, bajra.",
            "clay":    "Clay soil has poor drainage and compacts easily. Never work when wet. Add coarse sand + organic matter to improve structure. Raised beds help for vegetables.",
            "loamy":   "Loamy soil is the ideal — balanced drainage, fertility and structure. Maintain with regular compost additions and crop rotation. Most crops do well in loamy soil.",
            "acidic":  "Acidic soil (pH <6.0): apply Agricultural Lime (CaCO3) at 2–4 tonnes/ha. Recheck pH after 6 months. Avoid ammonium sulphate fertilizers which further acidify soil.",
            "alkaline":"Alkaline soil (pH >8.0): apply Gypsum (calcium sulphate) at 500 kg/ha. Use acidifying fertilizers like ammonium sulphate. Sulphur application at 20 kg/ha reduces pH over time.",
        }

        # ── General farming FAQs ───────────────────────────────────────────────
        self.faqs = {
            "crop rotation": "Rotate crops every season: deep-rooted → shallow-rooted → legume. E.g. wheat → cotton → chickpea. Breaks pest cycles, improves soil nitrogen, reduces disease.",
            "organic":       "Organic farming: use FYM/compost (10 t/ha), vermicompost, neem-based pesticides, Trichoderma for soil disease, crop rotation. Reduces input cost by 20–30% after establishment.",
            "irrigation":    "Drip irrigation saves 30–50% water vs flood. Schedule based on crop stage and forecast — avoid irrigating before rain. Critical stages: germination, flowering, grain fill.",
            "fertilizer":    "Apply NPK based on soil test. Split urea in 3 doses — basal, tillering, panicle initiation. Never apply before heavy rain. Micronutrients (Zinc, Boron) critical for grain quality.",
            "msp":           "MSP (Minimum Support Price) is set by CACP annually. 2024-25 key MSPs: Wheat ₹2,275, Rice ₹2,300, Cotton ₹7,121, Soybean ₹4,892, Chickpea ₹5,440 per quintal.",
            "sowing":        "Best sowing practices: treat seeds with fungicide + Rhizobium (for legumes). Sow at recommended depth and spacing. Maintain proper plant population for maximum yield.",
            "harvest":       "Harvest at correct moisture: wheat 14–18%, rice 20–22%. Delayed harvest causes shattering loss. Use combine harvester for large areas. Dry grain to 12% before storage.",
            "storage":       "Store grain at <14% moisture. Use hermetic bags or metal silos to prevent pest damage. Apply Aluminium Phosphide tablets (2–3/tonne) for pest control. Check monthly.",
        }

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC ENTRY POINT
    # ══════════════════════════════════════════════════════════════════════════

    def process_query(self, raw_input: str, context: dict = None) -> str:
        """
        context dict (passed from views.py process_message):
          city, state, current_temp, feels_like, humidity, wind_speed,
          description, temp_min, temp_max, rain_pred,
          forecast_temps (list, 7 days), forecast_hums (list, 7 days),
          forecast_dates (list, 7 days), soil_ph, soil_moisture, current_month
        """
        ctx = context or {}
        q   = raw_input.lower().strip()
        if not q:
            return "Please type your question and I'll help you."

        # ── Soil update detection (pH / moisture) ──────────────────────────────
        ph_match = (re.search(r'ph\s*(?:is|=|:)?\s*(\d+\.?\d*)', q)
                    or re.search(r'(\d+\.?\d*)\s*ph', q))
        mo_match = (re.search(r'moisture\s*(?:is|=|:)?\s*(\d+\.?\d*)', q)
                    or re.search(r'(\d+)\s*%?\s*moisture', q))

        if ph_match and not any(w in q for w in ['how', 'what', 'which', 'why']):
            ph = float(ph_match.group(1))
            if 4 <= ph <= 9:
                advice = ("Acidic — consider lime application." if ph < 5.5 else
                          "Ideal for most crops." if ph <= 7 else
                          "Slightly alkaline — wheat and cotton tolerate this well." if ph <= 7.5 else
                          "High pH — apply gypsum. Cotton and sorghum are more tolerant.")
                return f"✅ Soil pH {ph} noted. {advice} Crop recommendations on your dashboard have been updated."
            return "pH should be between 4 and 9. Please check the value."

        if mo_match and not any(w in q for w in ['how', 'what', 'which', 'why']):
            mo = float(mo_match.group(1))
            if 0 <= mo <= 100:
                advice = ("Very low — focus on drought-tolerant crops: bajra, chickpea, mustard." if mo < 25 else
                          "Moderate — good for wheat, maize and most Rabi crops." if mo < 45 else
                          "Good moisture — most crops will thrive." if mo < 70 else
                          "High moisture — rice and sugarcane are well-suited. Check drainage.")
                return f"✅ Soil moisture {mo}% noted. {advice} Recommendations updated."
            return "Moisture should be between 0 and 100%. Please recheck."

        # ── Greetings ──────────────────────────────────────────────────────────
        if re.search(r'\b(hello|hi|hey|namaste|namaskar|good morning|good afternoon|good evening)\b', q):
            city_str = f" for {ctx['city']}" if ctx.get('city') else ""
            return (f"Namaste! 🙏 I'm your Kisan Assistant.\n\n"
                    f"I can help you with:\n"
                    f"• Weather today{city_str} — ask 'what's the weather now'\n"
                    f"• 7-day forecast — ask 'show me the forecast'\n"
                    f"• Weather in other cities — ask 'weather in Delhi'\n"
                    f"• Crop info — ask about wheat, rice, cotton…\n"
                    f"• Top crops by location — ask 'top crops in Gujarat'\n"
                    f"• Disease control — Bacterial Blight, Rust, Blast…\n"
                    f"• Soil improvement — sandy, clay, acidic, alkaline…\n"
                    f"• Update your soil — say: my soil pH is 6.5\n\n"
                    f"What would you like to know?")

        # ── NEW: Crop recs for specific location (check BEFORE other intents) ──
        location_crop = self._location_crop_query(q, ctx)
        if location_crop:
            return location_crop

        # ── NEW: Weather for a different city ──────────────────────────────────
        other_city_response = self._other_city_weather(q, ctx)
        if other_city_response:
            return other_city_response

        # ── NEW: Current weather for own city (from context) ──────────────────
        own_city = ctx.get('city', '')
        if self._is_current_weather_query(q):
            if ctx.get('current_temp') is not None:
                return self._format_current_weather(ctx)
            return f"I don't have live weather data right now. Please check the dashboard weather card for current conditions in {own_city}."

        # ── NEW: Forecast for own city (from context) ─────────────────────────
        if self._is_forecast_query(q):
            if ctx.get('forecast_temps'):
                return self._format_forecast(ctx)
            return "Please check the 15-day forecast strip on your dashboard for upcoming weather."

        # ── Disease queries ────────────────────────────────────────────────────
        for disease, info in self.diseases.items():
            if any(kw in q for kw in disease.split()):
                return (f"🦠 {disease.title()}\n\n"
                        f"Affects: {', '.join(info['crops'])}\n"
                        f"Triggered by: {info['trigger']}\n\n"
                        f"Symptoms: {info['symptoms']}\n\n"
                        f"Prevention: {info['prevention']}\n\n"
                        f"Treatment: {info['treatment']}")

        # ── Pest queries ───────────────────────────────────────────────────────
        for pest, advice in self.pests.items():
            if pest in q or any(w in q for w in pest.split()):
                return f"🐛 {pest.title()}\n\n{advice}"

        # ── Specific crop queries ──────────────────────────────────────────────
        for crop, info in self.crops.items():
            if (f"about {crop}" in q or f"tell me {crop}" in q
                    or q.startswith(crop) or q == crop
                    or (f" {crop}" in q and not any(
                        rec in q for rec in ["which", "best", "recommend", "suggest",
                                             "suitable", "what crop", "good crop"]))):
                diseases_str = ", ".join(info["diseases"]) if info["diseases"] else "Minor disease risk"
                return (f"🌾 {crop.capitalize()}\n\n"
                        f"Season: {info['season']}\n"
                        f"Soil: {', '.join(info['soil'])}\n"
                        f"Climate: {info['climate']}\n"
                        f"Water: {info['water']}\n"
                        f"pH: {info['ph']}\n"
                        f"Yield: {info['yield']}\n"
                        f"MSP: {info['msp']}\n\n"
                        f"Disease risks: {diseases_str}\n\n"
                        f"💡 Tip: {info['tips']}")

        # ── Crop recommendation (generic — without location) ───────────────────
        rec_triggers = ["which crop", "best crop", "what to grow", "what should i grow",
                        "recommend crop", "suggest crop", "what crop", "good crop",
                        "which plant", "what plant", "suitable crop", "what should i plant"]
        if any(t in q for t in rec_triggers):
            return self._recommend_crop(q, ctx)

        # ── Weather action queries ─────────────────────────────────────────────
        weather_triggers = {
            "rain":     ["before rain", "after rain", "rain predicted", "rainfall", "heavy rain", "shower"],
            "heat":     ["heat wave", "heat stress", "high temperature", "too hot", "extreme heat"],
            "wind":     ["strong wind", "high wind", "wind storm", "wind speed", "lodging"],
            "humidity": ["high humidity", "humid conditions", "disease risk", "foggy"],
            "fog":      ["fog", "frost", "cold wave", "cold night"],
            "drought":  ["drought", "dry spell", "no rain", "water scarcity", "water shortage"],
            "flood":    ["flood", "waterlogged", "standing water", "excessive rain", "water logging"],
        }
        for topic, kws in weather_triggers.items():
            if any(kw in q for kw in kws):
                return f"🌤 Weather Advisory\n\n{self.weather_advice[topic]}"

        # ── Soil type queries ──────────────────────────────────────────────────
        soil_triggers = {
            "black":    ["black soil", "cotton soil", "vertisol", "black cotton"],
            "red":      ["red soil", "laterite", "red laterite"],
            "alluvial": ["alluvial", "river soil"],
            "sandy":    ["sandy soil", "sandy land", "sand soil"],
            "clay":     ["clay soil", "clayey", "heavy soil"],
            "loamy":    ["loam soil", "loamy soil"],
            "acidic":   ["acidic soil", "low ph", "acid soil"],
            "alkaline": ["alkaline soil", "high ph", "basic soil", "saline soil"],
        }
        for soil_type, kws in soil_triggers.items():
            if any(kw in q for kw in kws):
                return f"🧪 {soil_type.title()} Soil\n\n{self.soil_advice[soil_type]}"

        # ── Soil improvement ───────────────────────────────────────────────────
        improve_triggers = ["improve soil", "better soil", "soil health", "fix soil",
                            "soil problem", "poor soil", "soil quality", "soil management",
                            "how to improve", "add compost", "organic matter"]
        if any(t in q for t in improve_triggers):
            return self._soil_improvement(q)

        # ── FAQ topics ─────────────────────────────────────────────────────────
        faq_triggers = {
            "crop rotation": ["rotation", "rotate crop"],
            "organic":       ["organic", "natural farming", "zero budget"],
            "irrigation":    ["irrigation", "drip", "watering", "water schedule"],
            "fertilizer":    ["fertilizer", "fertiliser", "npk", "urea", "dap", "nutrients"],
            "msp":           ["msp", "minimum support price", "government price", "procurement price"],
            "sowing":        ["sowing", "seeding", "planting", "seed treatment", "sow"],
            "harvest":       ["harvest", "harvesting", "reaping", "when to cut"],
            "storage":       ["storage", "store grain", "silos", "godown", "warehouse"],
        }
        for topic, kws in faq_triggers.items():
            if any(kw in q for kw in kws):
                return f"📋 {topic.title()}\n\n{self.faqs[topic]}"

        # ── Spray / pesticide queries ──────────────────────────────────────────
        if any(w in q for w in ["spray", "pesticide", "fungicide", "insecticide", "chemical", "dose"]):
            return ("🧴 Spraying Guidelines\n\n"
                    "• Never spray before predicted rain — wash-off wastes chemical\n"
                    "• Spray early morning (6–9 AM) or late evening to avoid evaporation\n"
                    "• Strong winds above 15 km/h — postpone spraying\n"
                    "• Use correct dose: more is not better and causes resistance\n"
                    "• Wear protective equipment: gloves, mask, goggles\n"
                    "• Rotate chemical classes to prevent resistance development\n\n"
                    "Ask me about a specific disease or pest for the exact chemical and dose.")

        # ── Yield / income queries ─────────────────────────────────────────────
        if any(w in q for w in ["yield", "income", "profit", "revenue", "earning", "production"]):
            return ("💰 Yield & Income Estimation\n\n"
                    "Average yields (Maharashtra, good management):\n"
                    "• Wheat: 3.0–4.0 t/ha → ₹68,000–90,000/ha at MSP\n"
                    "• Soybean: 1.5–2.0 t/ha → ₹73,000–98,000/ha at MSP\n"
                    "• Cotton: 0.45 t/ha lint → ₹32,000–40,000/ha at MSP\n"
                    "• Onion: 15–25 t/ha → ₹2,40,000–4,00,000/ha (market price)\n"
                    "• Tomato: 20–30 t/ha → ₹3,60,000–5,40,000/ha (market price)\n\n"
                    "Ask me about a specific crop for detailed cost-benefit.")

        # ── Catch-all: try to find any crop name in question ──────────────────
        for crop in self.crops:
            if crop in q:
                return self._crop_information(crop)

        # ── Final fallback ────────────────────────────────────────────────────
        city_hint = f" or 'weather in {own_city}'" if own_city else ""
        return (f"I didn't quite understand '{raw_input[:60]}'. Try asking:\n\n"
                f"• \"What's the weather today?\"\n"
                f"• \"7-day forecast\"\n"
                f"• \"Weather in Delhi\"{city_hint}\n"
                f"• \"Top crops in Gujarat\"\n"
                f"• \"Tell me about wheat\"\n"
                f"• \"How to prevent Bacterial Blight\"\n"
                f"• \"My soil pH is 6.5\"")

    # ══════════════════════════════════════════════════════════════════════════
    # NEW: CONTEXT-AWARE WEATHER METHODS
    # ══════════════════════════════════════════════════════════════════════════

    def _is_current_weather_query(self, q: str) -> bool:
        """Detect queries about current weather for own city."""
        current_kws = [
            "weather today", "weather now", "current weather", "weather right now",
            "how hot is it", "how cold is it", "what is the temperature",
            "what's the temperature", "temperature today", "temp today",
            "how is the weather", "how's the weather", "what's the weather",
            "weather outside", "weather currently", "humidity now", "humidity today",
            "how humid", "wind speed now", "wind today", "feels like today",
        ]
        return any(kw in q for kw in current_kws)

    def _is_forecast_query(self, q: str) -> bool:
        """Detect queries about upcoming forecast for own city."""
        forecast_kws = [
            "forecast", "next few days", "this week", "upcoming weather",
            "next 7 days", "next week", "next 3 days", "coming days",
            "week ahead", "weather this week", "5 day", "7 day", "15 day",
            "next days", "tomorrow weather", "day after", "future weather",
        ]
        return any(kw in q for kw in forecast_kws)

    def _format_current_weather(self, ctx: dict) -> str:
        """Format a current weather response from context."""
        city  = ctx.get('city', 'your city')
        temp  = ctx.get('current_temp', '--')
        feels = ctx.get('feels_like', '--')
        hum   = ctx.get('humidity', '--')
        wind  = ctx.get('wind_speed', '--')
        desc  = ctx.get('description', 'conditions unknown')
        tmin  = ctx.get('temp_min', '--')
        tmax  = ctx.get('temp_max', '--')
        rain  = ctx.get('rain_pred', False)

        rain_str = "☔ Rain expected tomorrow — delay spraying." if rain else "☀️ No rain predicted tomorrow."
        heat_str = ""
        if isinstance(temp, (int, float)) and temp > 34:
            heat_str = "\n⚠️ Heat stress alert: irrigate early morning, monitor crops for wilting."

        return (f"🌤 Current Weather — {city}\n\n"
                f"Temperature:  {temp}°C (feels like {feels}°C)\n"
                f"Today's range: {tmin}°C – {tmax}°C\n"
                f"Humidity:     {hum}%\n"
                f"Wind speed:   {wind} m/s\n"
                f"Conditions:   {desc.capitalize()}\n\n"
                f"{rain_str}{heat_str}")

    def _format_forecast(self, ctx: dict) -> str:
        """Format a 7-day forecast response from context."""
        city   = ctx.get('city', 'your city')
        temps  = ctx.get('forecast_temps', [])
        hums   = ctx.get('forecast_hums', [])
        dates  = ctx.get('forecast_dates', [])

        if not temps or not dates:
            return "Forecast data not available right now. Check the 15-day strip on your dashboard."

        lines = [f"📅 7-Day Forecast — {city}\n"]
        for i, (d, t, h) in enumerate(zip(dates[:7], temps[:7], hums[:7])):
            icon = ("🌡️" if t > 38 else "☀️" if t > 34 else
                    "🌧️" if h > 80 else "⛅")
            warn = " ⚠️ Heat stress!" if t > 34 else ""
            lines.append(f"{icon} {d}:  {t}°C  |  {h}% RH{warn}")

        rain = ctx.get('rain_pred', False)
        if rain:
            lines.append("\n☔ RF model predicts rain tomorrow — delay fertiliser/pesticide.")
        lines.append("\nFull 15-day forecast is on your dashboard strip.")
        return "\n".join(lines)

    # ══════════════════════════════════════════════════════════════════════════
    # NEW: OTHER-CITY WEATHER
    # ══════════════════════════════════════════════════════════════════════════

    def _other_city_weather(self, q: str, ctx: dict) -> str:
        """Detect and handle weather queries for a city different from own city."""
        weather_kws = ["weather", "temperature", "temp", "forecast", "how hot", "how cold",
                       "humid", "rainfall", "raining", "what's it like", "how is it"]
        if not any(kw in q for kw in weather_kws):
            return ""

        city = self._extract_city_from_query(q)
        if not city:
            return ""

        own_city = (ctx.get('city') or '').lower()
        if city.lower() == own_city:
            return ""  # It's asking about own city — handled by _format_current_weather

        # Fetch live data for the other city
        return self._fetch_other_city_weather(city)

    def _fetch_other_city_weather(self, city: str) -> str:
        """Call OWM API for a city and return formatted weather."""
        if not self.owm_key:
            return f"I can't fetch live weather right now (API key not configured). You can search '{city}' using the city search bar at the top of your dashboard."

        try:
            url = (f"https://api.openweathermap.org/data/2.5/weather"
                   f"?q={city}&appid={self.owm_key}&units=metric")
            resp = requests.get(url, timeout=6)
            if resp.status_code == 404:
                return f"I couldn't find weather data for '{city}'. Please check the spelling or try the city search bar at the top."
            resp.raise_for_status()
            d = resp.json()

            temp  = round(d["main"]["temp"])
            feels = round(d["main"]["feels_like"])
            hum   = d["main"]["humidity"]
            wind  = round(d["wind"]["speed"], 1)
            tmin  = round(d["main"]["temp_min"])
            tmax  = round(d["main"]["temp_max"])
            desc  = d["weather"][0]["description"].capitalize()
            name  = d["name"]

            heat_note = ""
            if temp > 34:
                heat_note = "\n⚠️ Heat stress conditions — crops in this region need early-morning irrigation."

            return (f"🌤 Live Weather — {name}\n\n"
                    f"Temperature:  {temp}°C (feels like {feels}°C)\n"
                    f"Today's range: {tmin}°C – {tmax}°C\n"
                    f"Humidity:     {hum}%\n"
                    f"Wind speed:   {wind} m/s\n"
                    f"Conditions:   {desc}{heat_note}\n\n"
                    f"💡 Tip: To see the full 15-day forecast for {name}, type it in the city search bar at the top.")

        except requests.exceptions.Timeout:
            return f"Weather service timed out for '{city}'. Please try the city search bar on the dashboard."
        except Exception as e:
            return f"Could not fetch weather for '{city}' right now. Try searching via the city bar on the dashboard."

    # ══════════════════════════════════════════════════════════════════════════
    # NEW: LOCATION-SPECIFIC CROP RECOMMENDATIONS
    # ══════════════════════════════════════════════════════════════════════════

    def _location_crop_query(self, q: str, ctx: dict) -> str:
        """Detect and handle crop queries for specific states or cities."""
        crop_kws = ["crop", "grow", "plant", "sow", "cultivate", "farming", "yield",
                    "suitable crop", "best crop", "top crop", "which crop", "what to grow"]
        if not any(kw in q for kw in crop_kws):
            return ""

        location = self._extract_location(q)
        if not location:
            return ""

        # Map to state
        state = CITY_TO_STATE.get(location.lower()) or self._fuzzy_state(location)
        if not state:
            return ""

        # Get weather for that location (city-based if possible)
        avg_temp = 28.0
        avg_hum  = 65.0
        city_for_weather = location.capitalize()

        if self.owm_key:
            try:
                url = (f"https://api.openweathermap.org/data/2.5/weather"
                       f"?q={city_for_weather}&appid={self.owm_key}&units=metric")
                r = requests.get(url, timeout=5)
                if r.status_code == 200:
                    wd = r.json()
                    avg_temp = round(wd["main"]["temp"], 1)
                    avg_hum  = wd["main"]["humidity"]
            except Exception:
                pass

        if CropRecommender is None:
            return f"Crop recommendation engine not available for {location.title()}."

        now_month = ctx.get('current_month', datetime.now().month)
        try:
            recs = CropRecommender().recommend(
                avg_forecast_temp     = avg_temp,
                avg_forecast_humidity = avg_hum,
                soil_pH               = ctx.get('soil_ph') or 6.83,
                soil_moisture_pct     = ctx.get('soil_moisture') or 45.0,
                current_month         = now_month,
                state                 = state,
                top_n                 = 5,
            )
        except Exception:
            return f"Could not generate crop recommendations for {location.title()} right now."

        season = "Kharif" if now_month in range(5, 11) else "Rabi"
        lines = [f"🌾 Top Crops for {location.title()} ({state}) — {season} Season\n"
                 f"Based on current weather ({avg_temp}°C, {avg_hum}% RH)\n"]
        for i, r in enumerate(recs, 1):
            bar = "█" * (r.score // 10) + "░" * (10 - r.score // 10)
            lines.append(f"{i}. {r.crop} [{bar}] {r.score}/100\n"
                         f"   Season: {r.sow_window} | MSP: ₹{r.msp_inr}/qtl | "
                         f"Yield: {r.avg_yield_t_ha} t/ha")
        lines.append("\nAsk 'tell me about [crop]' for detailed info on any of these.")
        return "\n".join(lines)

    # ══════════════════════════════════════════════════════════════════════════
    # NEW: CITY / LOCATION EXTRACTION HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _extract_city_from_query(self, q: str) -> str:
        """Extract a city name from a weather-related query."""
        # Pattern: "weather in/at/for [city]"
        m = re.search(r'(?:weather|temperature|temp|forecast|like|humid|hot|cold)\s+(?:in|at|for|near)\s+([a-z\s]+?)(?:\s*[?!.]|$)', q)
        if m:
            candidate = m.group(1).strip()
            city = self._match_city(candidate)
            if city:
                return city

        # Pattern: "in/at [city] weather"
        m = re.search(r'\b(?:in|at|for)\s+([a-z]+)', q)
        if m:
            candidate = m.group(1).strip()
            if candidate in KNOWN_CITIES:
                return candidate

        # Scan all words for known cities
        for city in sorted(KNOWN_CITIES, key=len, reverse=True):
            if city in q:
                return city
        return ""

    def _extract_location(self, q: str) -> str:
        """Extract city or state from a crop/location query."""
        # Check for state names first
        for state in STATE_NAMES_LOWER:
            if state in q:
                return state.title()

        # Check for city names
        for city in sorted(KNOWN_CITIES, key=len, reverse=True):
            if city in q:
                return city

        # Pattern: "in/for/at [location]"
        m = re.search(r'\b(?:in|for|at|near|around)\s+([a-z\s]+?)(?:\s*[?!.,]|$)', q)
        if m:
            candidate = m.group(1).strip()
            city = self._match_city(candidate)
            if city:
                return city
        return ""

    def _match_city(self, candidate: str) -> str:
        """Match a candidate string to a known city (fuzzy single-word match)."""
        candidate = candidate.strip().lower()
        if candidate in KNOWN_CITIES:
            return candidate
        # Try each word
        for word in candidate.split():
            if word in KNOWN_CITIES:
                return word
        return ""

    def _fuzzy_state(self, location: str) -> str:
        """Try to match location string to a state name."""
        loc_lower = location.lower()
        for state in STATE_NAMES_LOWER:
            if loc_lower in state or state in loc_lower:
                # Return proper-cased state
                return location.title() if loc_lower in STATE_NAMES_LOWER else state.title()
        return ""

    # ══════════════════════════════════════════════════════════════════════════
    # EXISTING HELPERS (preserved unchanged)
    # ══════════════════════════════════════════════════════════════════════════

    def _recommend_crop(self, q: str, ctx: dict = None) -> str:
        soil_map = {
            "sandy":    ["sandy", "sand", "light soil"],
            "clay":     ["clay", "clayey", "heavy"],
            "black":    ["black", "cotton soil", "vertisol"],
            "red":      ["red", "laterite"],
            "alluvial": ["alluvial"],
            "loamy":    ["loam", "loamy"],
        }
        season_map = {
            "Kharif": ["kharif", "summer", "monsoon", "rainy"],
            "Rabi":   ["rabi", "winter", "cold"],
        }
        budget_map = {
            "low":  ["low cost", "cheap", "less money", "small budget", "limited"],
            "high": ["high value", "cash crop", "premium", "export"],
        }
        detected_soil   = next((s for s, kws in soil_map.items() if any(k in q for k in kws)), None)
        detected_season = next((s for s, kws in season_map.items() if any(k in q for k in kws)), None)
        detected_budget = next((b for b, kws in budget_map.items() if any(k in q for k in kws)), None)

        results = []
        for crop, info in self.crops.items():
            score = 0
            if detected_soil:
                if any(detected_soil in s for s in info["soil"]) or detected_soil in " ".join(info["soil"]):
                    score += 2
            if detected_season:
                if detected_season.lower() in info["season"].lower():
                    score += 2
            if detected_budget == "low":
                if crop in ["bajra", "chickpea", "groundnut", "wheat"]:
                    score += 1
            elif detected_budget == "high":
                if crop in ["cotton", "tomato", "onion", "groundnut"]:
                    score += 1
            if score > 0 or (not detected_soil and not detected_season):
                results.append((crop, score))

        results.sort(key=lambda x: -x[1])
        top = results[:4]

        if not top:
            return "Please tell me your soil type and season for crop recommendations."

        lines = ["🌾 Crop Recommendations\n"]
        for crop, _ in top:
            info = self.crops[crop]
            lines.append(f"• {crop.capitalize()} — {info['season']}, yield {info['yield']}, MSP {info['msp']}")
        lines.append("\nTip: Ask 'top crops in [state]' for state-specific recommendations with live weather data.")
        lines.append("Ask 'tell me about [crop name]' for full details.")
        return "\n".join(lines)

    def _soil_improvement(self, q: str) -> str:
        for soil_type, kws in {
            "sandy": ["sandy", "sand"], "clay": ["clay"], "black": ["black"],
            "acidic": ["acidic", "low ph", "acid"], "alkaline": ["alkaline", "high ph", "saline"],
            "red": ["red", "laterite"],
        }.items():
            if any(k in q for k in kws):
                return f"🧪 Improving {soil_type.title()} Soil\n\n{self.soil_advice.get(soil_type, '')}"
        return ("🧪 Soil Improvement Tips\n\n"
                "1. Add organic matter — FYM/compost at 10 t/ha every season\n"
                "2. Practice crop rotation to break pest cycles and fix nitrogen\n"
                "3. Avoid tillage when soil is wet — prevents compaction\n"
                "4. Mulching reduces evaporation and regulates soil temperature\n"
                "5. Get a soil test every 2–3 years to monitor pH and nutrients\n\n"
                "Tell me your soil type (sandy/clay/black/red/acidic/alkaline) for specific advice.")

    def _crop_information(self, crop: str) -> str:
        if crop not in self.crops:
            return (f"I don't have details for {crop} yet. Ask about wheat, rice, cotton, soybean, "
                    f"tomato, onion, maize, chickpea, groundnut, bajra, or sugarcane.")
        info = self.crops[crop]
        return (f"🌾 {crop.capitalize()}\n\n"
                f"Season: {info['season']}\n"
                f"Soil: {', '.join(info['soil'])}\n"
                f"Water: {info['water']}\n"
                f"pH: {info['ph']}\n"
                f"Yield: {info['yield']}\n"
                f"MSP: {info['msp']}\n\n"
                f"Disease risks: {', '.join(info['diseases'])}\n\n"
                f"💡 {info['tips']}")