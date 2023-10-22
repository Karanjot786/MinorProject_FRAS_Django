FROM python:3.8-slim-buster

# RUN apt-get update && apt-get install -y \
#     build-essential cmake \
#     libopenblas-dev liblapack-dev \
#     libx11-dev libgtk-3-dev \
#     python3 python3-dev python3-pip \
#     curl \
#     git\
#     wget

RUN pip3 install setuptools --upgrade
RUN pip3 install cython --upgrade

RUN pip3 install cmake 
RUN pip3 install numpy scipy matplotlib scikit-image scikit-learn ipython
RUN pip install wheel
RUN pip3 install dlib





# If you wanted to use this Dockerfile to run your own app instead, maybe you would do this:
LABEL Practice School 1
ENV PYTHONUNBUFFERED 1
WORKDIR /
COPY . /
RUN pip3 install -r requirements.txt
RUN python3 manage.py makemigrations
RUN python3 manage.py migrate
RUN python3 manage.py runserver