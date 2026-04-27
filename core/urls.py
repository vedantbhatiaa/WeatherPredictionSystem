from django.contrib import admin
from django.urls import path, include
from django.contrib.auth import views as auth_views


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('app.urls')),

    # Login — renders login.html template
    path('login/', auth_views.LoginView.as_view(
        template_name='login.html',
    ), name='login'),

    # Logout — supports both POST (Django 5.1 default) and GET
    # next_page sends user back to login after logout
    path('logout/', auth_views.LogoutView.as_view(
        next_page='/login/',
        http_method_names=['get', 'post'],
    ), name='logout'),
]