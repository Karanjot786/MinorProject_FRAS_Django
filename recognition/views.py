from django.shortcuts import render
from django.contrib.auth.models import User
from pandas.plotting import register_matplotlib_converters as rmc
from django_pandas.io import read_frame as rf
from django.contrib.auth.decorators import login_required
import seaborn as sn
from face_recognition.face_recognition_cli import image_files_in_folder as IFF
from sklearn.svm import SVC
from .forms import usernameForm,DateForm,UsernameAndDateForm, DateForm_2
import os
import cv2
import imutils
import dlib
from imutils import face_utils
from imutils.face_utils import FaceAligner
from imutils.video import VideoStream
import face_recognition
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pickle
import time
from django.contrib import messages
from django.shortcuts import redirect
from .models import Present, Time
import datetime
import math
from django.contrib.auth.forms import UserCreationForm


# Create your views here.

def home(request):
        return render(request, 'home.html')

def register(request):
	# if request.user.username!='admin':
	# 	return redirect('not-authorised')
	if request.method=='POST':
		form=UserCreationForm(request.POST)
		if form.is_valid():
			form.save() ###add user to database
			messages.success(request, f'Employee registered successfully!')
			return redirect('dashboard')
	else:
		form=UserCreationForm()
	return render(request,'student_register.html', {'form' : form})

def username_present(username):
	if User.objects.filter(username=username).exists():
		return True
	
	return False

def create_dataset(username):
	id = username
	if(os.path.exists('FR_Data/dataset/{}/'.format(id))==False):
		os.makedirs('FR_Data/dataset/{}/'.format(id))
	directory='FR_Data/dataset/{}/'.format(id)
	print("[INFO] Loading the facial detector")
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')
	fa = FaceAligner(predictor , desiredFaceWidth = 96)
	print("[INFO] Initializing Video stream")
	vs = VideoStream(src=0).start()
	sampleNum = 0
	while(True):
		frame = vs.read()
		frame = imutils.resize(frame ,width = 800)
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = detector(gray_frame,0)	
		for face in faces:
			print("inside for loop")
			(x,y,w,h) = face_utils.rect_to_bb(face)
			face_aligned = fa.align(frame,gray_frame,face)
			sampleNum = sampleNum+1
			if face is None:
				print("face is none")
				continue
			cv2.imwrite(directory+'/'+str(sampleNum)+'.jpg'	, face_aligned)
			face_aligned = imutils.resize(face_aligned ,width = 400)
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
			cv2.waitKey(50)
		cv2.imshow("Add Images",frame)
		cv2.waitKey(1)
		if(sampleNum>300):
			break
	vs.stop()
	cv2.destroyAllWindows()
	

def predict_face(aligned,svc,threshold=0.7):
	face_encodings=np.zeros((1,128))
	try:
		locations=face_recognition.face_locations(aligned)
		faces_encodings=face_recognition.face_encodings(aligned,face_locations=locations)
		if(len(faces_encodings)==0):
			return ([-1],[0])

	except:

		return ([-1],[0])

	prob=svc.predict_proba(faces_encodings)
	result=np.where(prob[0]==np.amax(prob[0]))
	if(prob[0][result[0]]<=threshold):
		return ([-1],prob[0][result[0]])

	return (result[0],prob[0][result[0]])

def v_Data(embedded, targets,):
	X = TSNE(n_components=2).fit_transform(embedded)
	for i, t in enumerate(set(targets)):
		idx = targets == t
		plt.scatter(X[idx, 0], X[idx, 1], label=t)
	plt.legend(bbox_to_anchor=(1, 1));
	rcParams.update({'figure.autolayout': True})
	plt.tight_layout()	
	plt.savefig('static/img/training_visualisation.png')
	plt.close()


			
def DB_attendance_in(present):
    today = datetime.date.today()
    time = datetime.datetime.now()

    for person in present:
        user = User.objects.get(username=person)
        try:
            q = Present.objects.get(user=user, date=today)
        except:
            pass

        if q is None:
            if present[person] == True:
                a = Present(user=user, date=today, present=True)
                a.save()
            else:
                a = Present(user=user, date=today, present=False)
                a.save()
        else:
            if present[person] == True:
                q.present = True
                q.save(update_fields=['present'])

        if present[person] == True:
            a = Time(user=user, date=today, time=time, out=False)
            a.save()

def DB_attendance_out(present):
    today = datetime.date.today()
    time = datetime.datetime.now()

    for person in present:
        user = User.objects.get(username=person)
        if present[person]==True:
            a = Time(user=user, date=today, time=time, out=True)
            a.save()
			

def validity_times(times_all):

	if(len(times_all)>0):
		sign=times_all.first().out
	else:
		sign=True
	times_in=times_all.filter(out=False)
	times_out=times_all.filter(out=True)
	if(len(times_in)!=len(times_out)):
		sign=True
	break_hourss=0
	if(sign==True):
			check=False
			break_hourss=0
			return (check,break_hourss)
	prev=True
	prev_time=times_all.first().time

	for obj in times_all:
		curr=obj.out
		if(curr==prev):
			check=False
			break_hourss=0
			return (check,break_hourss)
		if(curr==False):
			curr_time=obj.time
			to=curr_time
			ti=prev_time
			break_time=((to-ti).total_seconds())/3600
			break_hourss+=break_time


		else:
			prev_time=obj.time

		prev=curr

	return (True,break_hourss)
		
def h_to_h_m(hours):
	
	h=int(hours)
	hours-=h
	m=hours*60
	m=math.ceil(m)
	return str(str(h)+ " hrs " + str(m) + "  mins")

def given_student(present_qs,time_qs,admin=True):
	rmc()
	df_hours=[]
	df_break_hours=[]
	qs=present_qs

	for obj in qs:
		date=obj.date
		times_in=time_qs.filter(date=date).filter(out=False).order_by('time')
		times_out=time_qs.filter(date=date).filter(out=True).order_by('time')
		times_all=time_qs.filter(date=date).order_by('time')
		obj.time_in=None
		obj.time_out=None
		obj.hours=0
		obj.break_hours=0
		if (len(times_in)>0):			
			obj.time_in=times_in.first().time
			
		if (len(times_out)>0):
			obj.time_out=times_out.last().time

		if(obj.time_in is not None and obj.time_out is not None):
			ti=obj.time_in
			to=obj.time_out
			hours=((to-ti).total_seconds())/3600
			obj.hours=hours
		

		else:
			obj.hours=0

		(check,break_hourss)= validity_times(times_all)
		if check:
			obj.break_hours=break_hourss


		else:
			obj.break_hours=0


		
		df_hours.append(obj.hours)
		df_break_hours.append(obj.break_hours)
		obj.hours=h_to_h_m(obj.hours)
		obj.break_hours=h_to_h_m(obj.break_hours)
			
	
	
	
	df = rf(qs)	
	
	
	df["hours"]=df_hours
	df["break_hours"]=df_break_hours

	print(df)
	
	sn.barplot(data=df,x='date',y='hours')
	plt.xticks(rotation='vertical')
	rcParams.update({'figure.autolayout': True})
	plt.tight_layout()
	if(admin):
		plt.savefig('static/img/1.png')
		plt.close()
	else:
		plt.savefig('static/img/2.png')
		plt.close()
	return qs


def total_number_students():
	qs=User.objects.all()
	return (len(qs) -1)
	# -1 to account for admin 

def students_present_today():
	today=datetime.date.today()
	qs=Present.objects.filter(date=today).filter(present=True)
	return len(qs)

@login_required
def dashboard(request):
	if(request.user.username=='admin'):
		print("admin")
		return render(request, 'recognition/admin_dashboard.html')
	else:
		print("not admin")

		return render(request,'recognition/employee_dashboard.html')
	
@login_required
def add_photos(request):
	if request.user.username!='admin':
		return redirect('not-authorised')
	if request.method=='POST':
		form=usernameForm(request.POST)
		data = request.POST.copy()
		username=data.get('username')
		if username_present(username):
			create_dataset(username)
			messages.success(request, f'Dataset Created')
			return redirect('add-photos')
		else:
			messages.warning(request, f'No such username found. Please register employee first.')
			return redirect('dashboard')


	else:
		

			form=usernameForm()
			return render(request,'recognition/add_photos.html', {'form' : form})
	

def mark_attendance(request):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')
    svc_save_path="face_recognition_data/svc.sav"
    with open(svc_save_path, 'rb') as f:
            svc = pickle.load(f)
    fa = FaceAligner(predictor , desiredFaceWidth = 96)
    encoder=LabelEncoder()
    encoder.classes_ = np.load('face_recognition_data/classes.npy')
    faces_encodings = np.zeros((1,128))
    no_of_faces = len(svc.predict_proba(faces_encodings)[0])
    count = {}
    present = dict()
    log_time = dict()
    start = dict()
    for i in range(no_of_faces):
        count[encoder.inverse_transform([i])[0]] = 0
        present[encoder.inverse_transform([i])[0]] = False
    vs = VideoStream(src=0).start() 
    sampleNum = 0
    while(True):
        frame = vs.read()
        frame = imutils.resize(frame ,width = 800)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_frame,0)
        for face in faces:
            print("INFO : inside for loop")
            (x,y,w,h) = face_utils.rect_to_bb(face)
            face_aligned = fa.align(frame,gray_frame,face)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
            (pred,prob)=predict_face(face_aligned,svc)
            if(pred!=[-1]):
                person_name=encoder.inverse_transform(np.ravel([pred]))[0]
                pred=person_name
                if count[pred] == 0:
                    start[pred] = time.time()
                    count[pred] = count.get(pred,0) + 1
                if count[pred] == 4 and (time.time()-start[pred]) > 1.2:
                     count[pred] = 0
                else:
                    present[pred] = True
                    log_time[pred] = datetime.datetime.now()
                    count[pred] = count.get(pred,0) + 1
                    print(pred, present[pred], count[pred])
                cv2.putText(frame, str(person_name)+ str(prob), (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
            else:
                person_name="unknown"
                cv2.putText(frame, str(person_name), (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        cv2.imshow("Mark Attendance - In - Press q to exit",frame)
        key=cv2.waitKey(50) & 0xFF
        if(key==ord("q")):
            break
    
    vs.stop()

    cv2.destroyAllWindows()
    DB_attendance_in(present)
    return redirect('home')




def mark_attendance_out(request):
    D = dlib.get_frontal_face_detector()
    P = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')
    svc_save_path="face_recognition_data/svc.sav"
    with open(svc_save_path, 'rb') as f:
            svc = pickle.load(f)
    fa = FaceAligner(P , desiredFaceWidth = 96)
    encoder=LabelEncoder()
    encoder.classes_ = np.load('face_recognition_data/classes.npy')

    faces_encodings = np.zeros((1,128))
    no_of_faces = len(svc.predict_proba(faces_encodings)[0])
    count = {}
    present = {}
    log_time = {}
    start = {}
    for i in range(no_of_faces):
        count[encoder.inverse_transform([i])[0]] = 0
        present[encoder.inverse_transform([i])[0]] = False

    vs = VideoStream(src=0).start()
    sampleNum = 0

    while(True):
        frame = vs.read()
        frame = imutils.resize(frame ,width = 800)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = D(gray_frame,0)

        for face in faces:
            print("INFO : inside for loop")
            (x,y,w,h) = face_utils.rect_to_bb(face)

            face_aligned = fa.align(frame,gray_frame,face)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)

            (pred,prob)=predict_face(face_aligned,svc)

            if(pred!=[-1]):
                person_name=encoder.inverse_transform(np.ravel([pred]))[0]
                pred=person_name

                if present[pred] == True:
                    present[pred] = False
                    DB_attendance_out(pred)

                    print(pred, present[pred], count[pred])
                cv2.putText(frame, str(person_name)+ str(prob), (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

            else:
                person_name="unknown"
                cv2.putText(frame, str(person_name), (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
        cv2.imshow("Mark Attendance- Out - Press q to exit",frame)
        key=cv2.waitKey(50) & 0xFF
        if(key==ord("q")):
            break

    vs.stop()

    cv2.destroyAllWindows()
    return redirect('home')



def train(request):
	if request.user.username!='admin':
		return redirect('not-authorised')

	training_dir='face_recognition_data/training_dataset'
	count=0
	for person_name in os.listdir(training_dir):
		curr_directory=os.path.join(training_dir,person_name)
		if not os.path.isdir(curr_directory):
			continue
		for imagefile in IFF(curr_directory):
			count+=1

	X=[]
	y=[]
	i=0
	for person_name in os.listdir(training_dir):
		print(str(person_name))
		curr_directory=os.path.join(training_dir,person_name)
		if not os.path.isdir(curr_directory):
			continue
		for imagefile in IFF(curr_directory):
			print(str(imagefile))
			image=cv2.imread(imagefile)
			try:
				X.append((face_recognition.face_encodings(image)[0]).tolist())
				y.append(person_name)
				i+=1
			except:
				print("removed")
				os.remove(imagefile)

			


	targets=np.array(y)
	encoder = LabelEncoder()
	encoder.fit(y)
	y=encoder.transform(y)
	X1=np.array(X)
	print("shape: "+ str(X1.shape))
	np.save('face_recognition_data/classes.npy', encoder.classes_)
	svc = SVC(kernel='linear',probability=True)
	svc.fit(X1,y)
	svc_save_path="face_recognition_data/svc.sav"
	with open(svc_save_path, 'wb') as f:
		pickle.dump(svc,f)

	
	v_Data(X1,targets)
	messages.success(request, f'Training Complete.')
	return render(request,"recognition/train.html")

def student_given_date(present_qs,time_qs):
	rmc()
	df_hours=[]
	df_break_hours=[]
	df_username=[]
	qs=present_qs

	for obj in qs:
		user=obj.user
		times_in=time_qs.filter(user=user).filter(out=False)
		times_out=time_qs.filter(user=user).filter(out=True)
		times_all=time_qs.filter(user=user)
		obj.time_in=None
		obj.time_out=None
		obj.hours=0
		obj.hours=0
		if (len(times_in)>0):			
			obj.time_in=times_in.first().time
		if (len(times_out)>0):
			obj.time_out=times_out.last().time
		if(obj.time_in is not None and obj.time_out is not None):
			ti=obj.time_in
			to=obj.time_out
			hours=((to-ti).total_seconds())/3600
			obj.hours=hours
		else:
			obj.hours=0
		(check,break_hourss)= validity_times(times_all)
		if check:
			obj.break_hours=break_hourss


		else:
			obj.break_hours=0

		
		df_hours.append(obj.hours)
		df_username.append(user.username)
		df_break_hours.append(obj.break_hours)
		obj.hours=h_to_h_m(obj.hours)
		obj.break_hours=h_to_h_m(obj.break_hours)

	



	df = rf(qs)	
	df['hours']=df_hours
	df['username']=df_username
	df["break_hours"]=df_break_hours


	sn.barplot(data=df,x='username',y='hours')
	plt.xticks(rotation='vertical')
	rcParams.update({'figure.autolayout': True})
	plt.tight_layout()
	plt.savefig('./recognition/static/recognition/img/attendance_graphs/hours_vs_employee/1.png')
	plt.close()
	return qs



@login_required
def not_authorised(request):
	return render(request,'recognition/not_authorised.html')

@login_required
def view_attendance_home(request):
	total_num_of_stu=total_number_students()
	stu_present_today=students_present_today()
	return render(request,"recognition/view_attendance_home.html", {'total_num_of_stu' : total_num_of_stu, 'stu_present_today': stu_present_today})


@login_required
def view_attendance_date(request):
	if request.user.username!='admin':
		return redirect('not-authorised')
	qs=None
	time_qs=None
	present_qs=None


	if request.method=='POST':
		form=DateForm(request.POST)
		if form.is_valid():
			date=form.cleaned_data.get('date')
			print("date:"+ str(date))
			time_qs=Time.objects.filter(date=date)
			present_qs=Present.objects.filter(date=date)
			if(len(time_qs)>0 or len(present_qs)>0):
				qs=student_given_date(present_qs,time_qs)


				return render(request,'recognition/view_attendance_date.html', {'form' : form,'qs' : qs })
			else:
				messages.warning(request, f'No records for selected date.')
				return redirect('view-attendance-date')
	else:
		

			form=DateForm()
			return render(request,'recognition/view_attendance_date.html', {'form' : form, 'qs' : qs})


@login_required
def view_attendance_student(request):
	if request.user.username!='admin':
		return redirect('not-authorised')
	time_qs=None
	present_qs=None
	qs=None

	if request.method=='POST':
		form=UsernameAndDateForm(request.POST)
		if form.is_valid():
			username=form.cleaned_data.get('username')
			if username_present(username):
				
				u=User.objects.get(username=username)
				
				time_qs=Time.objects.filter(user=u)
				present_qs=Present.objects.filter(user=u)
				date_from=form.cleaned_data.get('date_from')
				date_to=form.cleaned_data.get('date_to')
				
				if date_to < date_from:
					messages.warning(request, f'Invalid date selection.')
					return redirect('view-attendance-employee')
				else:
					

					time_qs=time_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
					present_qs=present_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
					
					if (len(time_qs)>0 or len(present_qs)>0):
						qs=given_student(present_qs,time_qs,admin=True)
						return render(request,'recognition/view_attendance_employee.html', {'form' : form, 'qs' :qs})
					else:
						messages.warning(request, f'No records for selected duration.')
						return redirect('view-attendance-employee')
			else:
				print("invalid username")
				messages.warning(request, f'No such username found.')
				return redirect('view-attendance-employee')


	else:
		

			form=UsernameAndDateForm()
			return render(request,'recognition/view_attendance_employee.html', {'form' : form, 'qs' :qs})




@login_required
def view_my_attendance_student_login(request):
	if request.user.username=='admin':
		return redirect('not-authorised')
	qs=None
	time_qs=None
	present_qs=None
	if request.method=='POST':
		form=DateForm_2(request.POST)
		if form.is_valid():
			u=request.user
			time_qs=Time.objects.filter(user=u)
			present_qs=Present.objects.filter(user=u)
			date_from=form.cleaned_data.get('date_from')
			date_to=form.cleaned_data.get('date_to')
			if date_to < date_from:
					messages.warning(request, f'Invalid date selection.')
					return redirect('view-my-attendance-employee-login')
			else:
					

					time_qs=time_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
					present_qs=present_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
				
					if (len(time_qs)>0 or len(present_qs)>0):
						qs=given_student(present_qs,time_qs,admin=False)
						return render(request,'recognition/view_my_attendance_employee_login.html', {'form' : form, 'qs' :qs})
					else:
						
						messages.warning(request, f'No records for selected duration.')
						return redirect('view-my-attendance-employee-login')
	else:
		

			form=DateForm_2()
			return render(request,'recognition/view_my_attendance_employee_login.html', {'form' : form, 'qs' :qs})