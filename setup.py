"""
The setup.py contains instructions for pip what it should install
"""
from setuptools import setup
import os

def open_file(fname):
   return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
   name="dr_app",            # name of your package
   version="0.0.1",
   description="diabetic retinopathy diagnosis",
   long_description=open_file("README.md"),  # only if you have a README.md
   packages=["model"],      # THIS IS THE FOLDER NAME, PIP WILL COPY EVERYTHIN IN HERE
   url="...",
   license="MIT",
   classifiers=[
      "Programming Language :: Python :: 3.6.13", # earlier python version because of tensorflow
   ]
)
