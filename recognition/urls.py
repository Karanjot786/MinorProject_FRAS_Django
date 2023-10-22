from .views import register, home, dashboard, train, add_photos, mark_attendance, mark_attendance_out, view_attendance_home, view_attendance_date, view_attendance_student, view_my_attendance_student_login, not_authorised
from django.urls import path
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', home, name='home'),
    path('register/', register, name='register'),
      path('dashboard/', dashboard, name='dashboard'),
      path('train/', train, name='train'),
    path('add_photos/', add_photos, name='add-photos'),
     path('login/',auth_views.LoginView.as_view(template_name='login.html'),name='login'),
    path('logout/',auth_views.LogoutView.as_view(template_name='home.html'),name='logout'),
     path('mark_your_attendance', mark_attendance ,name='mark-your-attendance'),
      path('mark_your_attendance_out', mark_attendance_out ,name='mark-your-attendance-out'),
      path('view_attendance_home',view_attendance_home ,name='view-attendance-home'),
        path('view_attendance_date', view_attendance_date ,name='view-attendance-date'),
        path('view_attendance_student', view_attendance_student ,name='view-attendance-student'),
         path('view_my_attendance', view_my_attendance_student_login ,name='view-my-attendance-student-login'),
       path('not_authorised', not_authorised, name='not-authorised')

]