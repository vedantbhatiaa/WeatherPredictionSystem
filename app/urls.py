from django.urls import path
from . import views

urlpatterns = [
    path('',                  views.dashboard,          name='dashboard'),
    path('register/',         views.register_view,       name='register'),
    path('onboarding/',       views.onboarding_view,     name='onboarding'),
    path('save-profile/',     views.save_profile,        name='save_profile'),
    path('get-weather/',      views.get_weather,         name='get_weather'),
    path('process-message/',  views.process_message,     name='process_message'),
    path('update-soil/',      views.update_soil,         name='update_soil'),
    path('profile/edit/',     views.profile_edit,        name='profile_edit'),
    path('profile/password/', views.password_change,     name='password_change'),
    path('health/',           views.health_check,        name='health_check'),
]