# KisanMausam — Weather & Farm Advisory

## Project Structure
```
WeatherPredictionSystem/
├── app/
│   ├── views.py            ← all logic (weather, auth, alerts, crops, chatbot)
│   ├── models.py           ← FarmerProfile, WeatherCache, AlertLog
│   ├── admin.py            ← Django admin config
│   ├── alert_engine.py     ← RF-driven hazard detection
│   ├── crop_recommender.py ← crop scoring engine
│   ├── farm_assistant.py   ← chatbot
│   └── urls.py
├── core/
│   ├── settings.py         ← reads from .env
│   ├── urls.py
│   └── wsgi.py
├── templates/
│   ├── login.html          ← login + register
│   ├── onboarding.html     ← 5-step wizard
│   └── weather.html        ← main dashboard
├── static/                 ← css, js, images
├── Dataset/                ← weather.csv, HistoricalData.csv
├── .env.example            ← copy to .env and fill in keys
├── manage.py
└── requirements.txt
```

## How to Run (First Time)

### 1. Setup
```bash
cd WeatherPredictionSystem
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

### 2. Environment
```bash
copy .env.example .env      # Windows
cp .env.example .env        # Mac/Linux
# Edit .env — at minimum set DJANGO_SECRET_KEY
```

### 3. Database
```bash
python manage.py makemigrations app
python manage.py migrate
python manage.py createsuperuser   # creates admin account
```

### 4. Run
```bash
python manage.py runserver
```
Open: http://127.0.0.1:8000

## Supabase Integration

### Step 1 — Create Supabase project
1. Go to supabase.com → New project
2. Settings → API → copy URL and anon key and service role key
3. Settings → Database → copy connection string (URI mode)

### Step 2 — Add to .env
```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_KEY=your-service-role-key
DATABASE_URL=postgresql://postgres:[PASSWORD]@db.[REF].supabase.co:5432/postgres
```

### Step 3 — Create Supabase tables
Run in Supabase SQL Editor:
```sql
-- Farmer profiles (Django also writes to this)
create table farmers (
  id               bigint generated always as identity primary key,
  django_user_id   int unique,
  username         text,
  name             text,
  role             text default 'farmer',
  state            text,
  city             text,
  crops            text,         -- JSON string
  crop_stage       text,
  farm_size_acres  float,
  irrigation       text,
  soil_type        text,
  soil_ph          float,
  soil_moisture    float,
  created_at       timestamptz default now(),
  updated_at       timestamptz default now()
);

-- Alert log
create table alert_log (
  id           bigint generated always as identity primary key,
  city         text,
  alert_type   text,
  severity     text,
  message      text,
  farmer_id    int,
  triggered_at timestamptz default now()
);
```

### Step 4 — Migrate to Supabase DB
```bash
python manage.py migrate   # runs against Supabase PostgreSQL
```

### Step 5 — Admin dashboard
- Django admin: http://127.0.0.1:8000/admin/
- Supabase Table Editor: live farmer data, alerts, crop choices

## URL Map
| URL | View | Description |
|-----|------|-------------|
| `/` | dashboard | Main farm dashboard (login required) |
| `/login/` | login | Farmer sign in |
| `/register/` | register | New farmer account |
| `/onboarding/` | onboarding | 5-step setup wizard |
| `/save-profile/` | save_profile | AJAX — saves onboarding data |
| `/get-weather/?city=X` | get_weather | AJAX — live weather + predictions |
| `/process-message/` | process_message | Chatbot AJAX |
| `/update-soil/` | update_soil | AJAX — soil pH/moisture update |
| `/admin/` | Django admin | View all farmers, alerts, cache |

## API Keys
- OWM_KEY: OpenWeatherMap (current weather)
- WBIT_KEY: Weatherbit (16-day forecast) — dates are now DYNAMIC
  Previously hardcoded 2025 dates — now uses today's date automatically

## Key Fixes in This Version
1. Weatherbit URL dates are DYNAMIC (today → today+16) — was hardcoded 2025
2. RF models train ONCE at startup, cached in memory — was retraining every request
3. Weather cache: API failure → serves last DB cache silently, farmer sees no error
4. Full auth: Login, register, session, logout
5. FarmerProfile persisted in DB + synced to Supabase
6. AlertLog written to DB + Supabase for admin analytics
