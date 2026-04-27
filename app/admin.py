from django.contrib import admin
from .models import FarmerProfile, WeatherCache, AlertLog


@admin.register(FarmerProfile)
class FarmerProfileAdmin(admin.ModelAdmin):
    list_display  = ('user', 'role', 'state', 'city', 'farm_size', 'soil_type', 'crop_stage', 'updated_at')
    list_filter   = ('role', 'state', 'soil_type', 'irrigation')
    search_fields = ('user__username', 'user__email', 'state', 'city')
    readonly_fields = ('created_at', 'updated_at')


@admin.register(WeatherCache)
class WeatherCacheAdmin(admin.ModelAdmin):
    list_display  = ('city', 'fetched_at')
    readonly_fields = ('fetched_at',)


@admin.register(AlertLog)
class AlertLogAdmin(admin.ModelAdmin):
    list_display  = ('alert_type', 'severity', 'city', 'farmer', 'triggered_at')
    list_filter   = ('severity', 'city', 'alert_type')
    search_fields = ('city', 'alert_type', 'farmer__username')
    readonly_fields = ('triggered_at',)
