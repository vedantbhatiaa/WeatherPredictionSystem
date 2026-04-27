from django.db import models
from django.contrib.auth.models import User
import json


class FarmerProfile(models.Model):
    """One profile per farmer user — linked to Django auth User."""
    ROLE_CHOICES = [('farmer', 'Farmer'), ('admin', 'Admin')]
    SOIL_CHOICES = [('black', 'Black / Cotton'), ('red', 'Red / Laterite'),
                    ('alluvial', 'Alluvial / Sandy loam'), ('loamy', 'Loamy / Mixed')]
    IRR_CHOICES  = [('rainfed', 'Rain-fed'), ('borewell', 'Borewell'),
                    ('canal', 'Canal'), ('drip', 'Drip / Sprinkler')]
    STAGE_CHOICES = [('Sowing', 'Sowing'), ('Vegetative', 'Vegetative'),
                     ('Flowering', 'Flowering'), ('Harvest', 'Harvest')]

    user         = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    role         = models.CharField(max_length=10, choices=ROLE_CHOICES, default='farmer')
    state        = models.CharField(max_length=60, blank=True)
    city         = models.CharField(max_length=60, blank=True, default='Mumbai')
    farm_size    = models.FloatField(null=True, blank=True, help_text='Acres')
    irrigation   = models.CharField(max_length=20, choices=IRR_CHOICES, blank=True)
    soil_type    = models.CharField(max_length=20, choices=SOIL_CHOICES, blank=True)
    soil_ph      = models.FloatField(null=True, blank=True)
    soil_moisture= models.FloatField(null=True, blank=True)
    crop_stage   = models.CharField(max_length=20, choices=STAGE_CHOICES, blank=True)
    crops_json   = models.TextField(blank=True, default='[]',
                                    help_text='JSON list of crop names e.g. ["Wheat","Rice"]')
    created_at   = models.DateTimeField(auto_now_add=True)
    updated_at   = models.DateTimeField(auto_now=True)

    @property
    def crops(self):
        try:    return json.loads(self.crops_json)
        except: return []

    @crops.setter
    def crops(self, val):
        self.crops_json = json.dumps(val)

    def __str__(self):
        return f"{self.user.username} — {self.state}"

    class Meta:
        verbose_name = 'Farmer Profile'


class WeatherCache(models.Model):
    """Stores the last successful weather fetch per city.
    When live API fails, serve this silently — farmer sees no error."""
    city         = models.CharField(max_length=60, unique=True)
    current_json = models.TextField()     # get_current_weather() result
    forecast_json= models.TextField()     # 15-day arrays
    fetched_at   = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Cache: {self.city} @ {self.fetched_at}"

    class Meta:
        verbose_name = 'Weather Cache'


class AlertLog(models.Model):
    """Every alert fired is logged — admin can see which cities/farmers get what."""
    SEVERITY_CHOICES = [('info','Info'), ('warning','Warning'), ('danger','Danger')]
    farmer       = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    city         = models.CharField(max_length=60)
    alert_type   = models.CharField(max_length=80)
    severity     = models.CharField(max_length=10, choices=SEVERITY_CHOICES)
    message      = models.TextField()
    triggered_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.alert_type} — {self.city}"

    class Meta:
        ordering = ['-triggered_at']
        verbose_name = 'Alert Log'
