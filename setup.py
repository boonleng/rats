import os
import sys
import textwrap
import pkg_resources

MIN_PYTHON = (3, 4)
if sys.version_info < MIN_PYTHON:
    sys.exit('Python %s or later is required.\n' % '.'.join("%s" % n for n in MIN_PYTHON))

class COLOR:
    reset = "\033[0m"
    red = "\033[38;5;196m"
    orange = "\033[38;5;214m"
    yellow = "\033[38;5;226m"
    lime = "\033[38;5;118m"
    green = "\033[38;5;46m"
    teal = "\033[38;5;49m"
    iceblue = "\033[38;5;51m"
    skyblue = "\033[38;5;45m"
    blue = "\033[38;5;27m"
    purple = "\033[38;5;99m"
    indigo = "\033[38;5;201m"
    hotpink = "\033[38;5;199m"
    pink = "\033[38;5;213m"
    deeppink = "\033[38;5;198m"
    salmon = "\033[38;5;210m"
    python = "\033[38;5;226;48;5;24m"
    radarkit = "\033[38;5;15;48;5;124m"

def colorize(string, color):
    return '{}{}{}'.format(color, string, COLOR.reset)

print('Using version {} ...'.format(colorize(sys.version_info, COLOR.lime)))

def is_installed(requirement):
    try:
        print('Checking for {} ...'.format(requirement))
        pkg_resources.require(requirement)
    except pkg_resources.ResolutionError:
        print('{} not found'.format(requirement))
        return False
    else:
        result = pkg_resources.require(requirement)
        print('{} found {}'.format(requirement, result[0]))
        return True

install_requires = [
    'enum',
    'numpy',
    'scipy',
    'pandas',
    'pandas_datareader',
    'requests_cache',
    'matplotlib'
]

from setuptools import setup

setup(
    name='Regression Analysis of Time Series',
    version='0.5',
    description='A bunch of examples of regression analysis using Deep Learning',
    author='Boonleng Cheong',
    author_email='boonleng@ou.edu',
    url='https://git.arrc.ou.edu/cheo4524/rats.git',
    install_requires=install_requires,
    zip_safe=False,
    license='MIT'
)

