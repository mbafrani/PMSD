"""
Python model "new1StockSFD.py"
Translated using PySD version 0.10.0
"""
from __future__ import division
import numpy as np
from pysd import utils
import xarray as xr

from pysd.py_backend.functions import cache
from pysd.py_backend import functions

_subscript_dict = {}

_namespace = {
    'TIME': 'time',
    'Time': 'time',
    'Arrivalrate1W': 'arrivalrate1w',
    'Finishrate1W': 'finishrate1w',
    'Numofuniqueresource1W': 'numofuniqueresource1w',
    'Processactivetime1W': 'processactivetime1w',
    'Servicetimepercase1W': 'servicetimepercase1w',
    'Timeinprocesspercase1W': 'timeinprocesspercase1w',
    'Waitingtimeinprocesspercase1W': 'waitingtimeinprocesspercase1w',
    'Numinprocesscase1W': 'numinprocesscase1w',
    'FINAL TIME': 'final_time',
    'INITIAL TIME': 'initial_time',
    'SAVEPER': 'saveper',
    'TIME STEP': 'time_step'
}

__pysd_version__ = "0.10.0"

__data = {'scope': None, 'time': lambda: 0}


def _init_outer_references(data):
    for key in data:
        __data[key] = data[key]


def time():
    return __data['time']()


@cache('step')
def arrivalrate1w():
    """
    Real Name: b'Arrivalrate1W'
    Original Eqn: b'A FUNCTION OF(Arrivalrate1W)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return functions.incomplete()


@cache('step')
def finishrate1w():
    """
    Real Name: b'Finishrate1W'
    Original Eqn: b'A FUNCTION OF(Finishrate1W)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return functions.incomplete()


@cache('step')
def numofuniqueresource1w():
    """
    Real Name: b'Numofuniqueresource1W'
    Original Eqn: b'A FUNCTION OF(Numofuniqueresource1W)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return functions.incomplete()


@cache('step')
def processactivetime1w():
    """
    Real Name: b'Processactivetime1W'
    Original Eqn: b'A FUNCTION OF(Processactivetime1W)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return functions.incomplete()


@cache('step')
def servicetimepercase1w():
    """
    Real Name: b'Servicetimepercase1W'
    Original Eqn: b'A FUNCTION OF(Timeinprocesspercase1W)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return functions.incomplete()


@cache('step')
def timeinprocesspercase1w():
    """
    Real Name: b'Timeinprocesspercase1W'
    Original Eqn: b'A FUNCTION OF(Waitingtimeinprocesspercase1W)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return functions.incomplete()


@cache('step')
def waitingtimeinprocesspercase1w():
    """
    Real Name: b'Waitingtimeinprocesspercase1W'
    Original Eqn: b'A FUNCTION OF(Servicetimepercase1W,Numinprocesscase1W)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return functions.incomplete(numinprocesscase1w())


@cache('step')
def numinprocesscase1w():
    """
    Real Name: b'Numinprocesscase1W'
    Original Eqn: b'A FUNCTION OF(Timeinprocesspercase1W)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return functions.incomplete()


@cache('run')
def final_time():
    """
    Real Name: b'FINAL TIME'
    Original Eqn: b'100'
    Units: b'Month'
    Limits: (None, None)
    Type: constant

    b'The final time for the simulation.'
    """
    return 100


@cache('run')
def initial_time():
    """
    Real Name: b'INITIAL TIME'
    Original Eqn: b'0'
    Units: b'Month'
    Limits: (None, None)
    Type: constant

    b'The initial time for the simulation.'
    """
    return 0


@cache('step')
def saveper():
    """
    Real Name: b'SAVEPER'
    Original Eqn: b'TIME STEP'
    Units: b'Month'
    Limits: (0.0, None)
    Type: component

    b'The frequency with which output is stored.'
    """
    return time_step()


@cache('run')
def time_step():
    """
    Real Name: b'TIME STEP'
    Original Eqn: b'1'
    Units: b'Month'
    Limits: (0.0, None)
    Type: constant

    b'The time step for the simulation.'
    """
    return 1
