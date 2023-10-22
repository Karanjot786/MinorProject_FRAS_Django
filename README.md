## SMART ATTENDANCE 💻

This project involves building an attendance system which utilizes facial recognition to mark the presence, time-in, and time-out of students. It covers areas such as facial detection, alignment, and recognition, along with the development of a web application to cater to various use cases of the system such as registration of new students, addition of photos to the training dataset, viewing attendance reports, etc. This project intends to serve as an efficient substitute for traditional manual attendance systems. It can be used in corporate offices, schools, and organizations where security is essential.

This project aims to automate the traditional attendance system where the attendance is marked manually. It also enables an organization to maintain its records like in-time, out time, break time and attendance digitally. Digitalization of the system would also help in better visualization of the data using graphs to display the no. of students present today, total work hours of each student and their break time. Its added features serve as an efficient upgrade and replacement over the traditional attendance system.

## Scope of the project 🚀
Facial recognition is becoming more prominent in our society. It has made major progress in the field of security. It is a very effective tool that can help low enforcers to recognize criminals and software companies are leveraging the technology to help users access the technology. This technology can be further developed to be used in other avenues such as ATMs, accessing confidential files, or other sensitive materials.
This project servers as a foundation for future projects based on facial detection and recognition. This project also convers web development and database management with a user-friendly UI. Using this system any corporate offices, school and organization can replace their traditional way of maintaining attendance of the students and can also generate their availability(presence) report throughout the month.

**The system mainly works around 2 types of users**
1. Student
2. Admin

**Following functionalities can be performed by the admin: <br>**
• Login <br>
• Register new students to the system <br>
• Add Student photos to the training data set <br>
• Train the model <br>
• View attendance report of all students. Attendance can be filtered by date or Student. <br>

**Following functionalities can be performed by the Student: <br>**
• Login <br>
• Mark his/her time-in and time-out by scanning their face <br>
• View attendance report of self <br>

## Face Detection
Dlib's HOG facial detector.

## Facial Landmark Detection
Dlib's 68 point shape predictor

## Extraction of Facial Embeddings
face_recognition by Adam Geitgey

## Classification of Unknown Embedding 
using a Linear SVM (scikit-learn)

 
## How To Run ?
```bash
git clone github.com/karanjot786/minorProject_FRAS_Django
cd minorProject_FRAS_Django
sudo pip3 install virtualenv
virtualenv env
source env/bin/activate
pip3 install cmake 
pip3 install dlib
pip3 install -r requirements.txt
python3 manage.py makemigrations
python3 manage.py migrate
python3 manage.py runserver

```
- Enjoy !

---------

```javascript

if (youEnjoyed) {
    starThisRepository();
}

```

-----------

## Thank You
- Author : [Karanjot Singh](karanjot.vercel.app)
