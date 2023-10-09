from .views import login,register, home
from django.urls import path

urlpatterns = [
    path('login/', login, name='index'),
    path('', home, name='index'),
    path('register/', register, name='index')

]