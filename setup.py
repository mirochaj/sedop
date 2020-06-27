#!/usr/bin/env python

import os

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='sedop',
      version='1.0',
      description='Multi-frequency SED optimization via simulated annealing',
      author='Jordan Mirocha',
      author_email='mirochaj@gmail.com',
      url='github.com/mirochaj/sedop',
      packages=['sedop'],
     )
     
