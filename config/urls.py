# config/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    # This connects the ingestion app and prefixes it with 'api/'
    path('api/', include('apps.ingestion.urls')),
]