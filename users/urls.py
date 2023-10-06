from .views import login,register
from django.urls import path

urlpatterns = [
    path('login/', login, name='index'),
    path('', register, name='index')

]