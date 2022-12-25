from setuptools import setup
from setuptools import find_packages

setup(name='cancer_ti',
      description='Tool to infer cancer patients development time series',
      author='hengyuan zhang',
      install_requires=['networkx',
                        'numpy',
                        'cupy-cuda*',  # pip install wheel
                        'scikit-learn',
                        'scipy',  # pip install
                        'pandas',
                        'pyHSICLasso',  # pip install
                        'torch',  # conda install pytorch cudatoolkit=11.3 -c pytorch
                        'pyecharts',  # pip install
                        'matplotlib',
                        'seaborn',
                        'requests',
                        'openpyxl',
                        'xlrd'],
      packages=find_packages())