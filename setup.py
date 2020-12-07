from setuptools import find_packages
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    name='paklib',
    version='0.2.3',
    packages=find_packages(),
    ext_modules=[
        Extension('cyton', ['paklib/cyton.pyx']),
    ],
    cmdclass={'build_ext': build_ext},
    url='https://github.com/ipakeev',
    license='MIT',
    author='Ipakeev',
    author_email='ipakeev93@gmail.com',
    description='Tools for self usage'
)
