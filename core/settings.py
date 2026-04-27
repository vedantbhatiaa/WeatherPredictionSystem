import os
import dj_database_url
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.getenv('DJANGO_SECRET_KEY', 'django-insecure-dev-key-change-in-production')
DEBUG      = os.getenv('DEBUG', 'True') == 'True'

ALLOWED_HOSTS = os.getenv('ALLOWED_HOSTS', '*').split(',')

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'app',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',          # ← static files in prod
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'core.urls'

TEMPLATES = [{
    'BACKEND': 'django.template.backends.django.DjangoTemplates',
    'DIRS': [BASE_DIR / 'templates'],
    'APP_DIRS': True,
    'OPTIONS': {'context_processors': [
        'django.template.context_processors.debug',
        'django.template.context_processors.request',
        'django.contrib.auth.context_processors.auth',
        'django.contrib.messages.context_processors.messages',
        'django.template.context_processors.csrf',
    ]},
}]

WSGI_APPLICATION = 'core.wsgi.application'

# ── Database ──────────────────────────────────────────────────────────────────
# Reads DATABASE_URL from .env → Supabase PostgreSQL in prod, SQLite in dev
DATABASE_URL = os.getenv('DATABASE_URL', '')

if DATABASE_URL:
    DATABASES = {
        'default': dj_database_url.parse(
            DATABASE_URL,
            conn_max_age=600,
            conn_health_checks=True,
        )
    }
else:
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': BASE_DIR / 'db.sqlite3',
        }
    }

# ── Auth ──────────────────────────────────────────────────────────────────────
LOGIN_URL             = '/login/'
LOGIN_REDIRECT_URL    = '/'
LOGOUT_REDIRECT_URL   = '/login/'

# ── Session ───────────────────────────────────────────────────────────────────
SESSION_COOKIE_AGE        = 60 * 60 * 24 * 30   # 30 days
SESSION_SAVE_EVERY_REQUEST = False

# ── Static ────────────────────────────────────────────────────────────────────
STATIC_URL         = '/static/'
STATICFILES_DIRS   = [BASE_DIR / 'static']
STATIC_ROOT        = BASE_DIR / 'staticfiles'
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# ── API Keys (env only — no hardcoded fallbacks) ──────────────────────────────
OWM_KEY  = os.getenv('OWM_KEY',  '')
WBIT_KEY = os.getenv('WBIT_KEY', '')

# ── Supabase ──────────────────────────────────────────────────────────────────
SUPABASE_URL         = os.getenv('SUPABASE_URL', '')
SUPABASE_ANON_KEY    = os.getenv('SUPABASE_ANON_KEY', '')
SUPABASE_SERVICE_KEY = os.getenv('SUPABASE_SERVICE_KEY', '')

# Error handlers
HANDLER404 = 'django.views.defaults.page_not_found'
HANDLER500 = 'django.views.defaults.server_error'