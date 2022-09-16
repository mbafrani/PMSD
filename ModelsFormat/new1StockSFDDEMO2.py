"""
Python model "new1StockSFDDEMO2.py"
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
    'Timeinprocesspercase1H': 'timeinprocesspercase1h',
    'Servicetimepercase1H': 'servicetimepercase1h',
    'Numofuniqueresource1H': 'numofuniqueresource1h',
    'Arrivalrate1H': 'arrivalrate1h',
    'Finishrate1H': 'finishrate1h',
    'Numinprocesscase1H': 'numinprocesscase1h',
    'Waitingtimeinprocesspercase1H': 'waitingtimeinprocesspercase1h',
    'Processactivetime1H': 'processactivetime1h',
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
def timeinprocesspercase1h():
    """
    Real Name: b'Timeinprocesspercase1H'
    Original Eqn: b'A FUNCTION OF(Timeinprocesspercase1H)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return functions.incomplete()


@cache('step')
def servicetimepercase1h():
    """
    Real Name: b'Servicetimepercase1H'
    Original Eqn: b'A FUNCTION OF(Servicetimepercase1H)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return functions.incomplete()


@cache('step')
def numofuniqueresource1h():
    """
    Real Name: b'Numofuniqueresource1H'
    Original Eqn: b'A FUNCTION OF(Numofuniqueresource1H)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return functions.incomplete()


@cache('step')
def arrivalrate1h():
    """
    Real Name: b'Arrivalrate1H'
    Original Eqn: b'A FUNCTION OF(Arrivalrate1H)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return functions.incomplete()


@cache('step')
def finishrate1h():
    """
    Real Name: b'Finishrate1H'
    Original Eqn: b'Numofuniqueresource1H/Servicetimepercase1H'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return numofuniqueresource1h() / servicetimepercase1h()


@cache('step')
def numinprocesscase1h():
    """
    Real Name: b'Numinprocesscase1H'
    Original Eqn: b'INTEG ( Arrivalrate1H+Finishrate1H, 0)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return _integ_numinprocesscase1h()


@cache('step')
def waitingtimeinprocesspercase1h():
    """
    Real Name: b'Waitingtimeinprocesspercase1H'
    Original Eqn: b'MAX(Timeinprocesspercase1H-Servicetimepercase1H,0)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return np.maximum(timeinprocesspercase1h() - servicetimepercase1h(), 0)


@cache('step')
def processactivetime1h():
    """
    Real Name: b'Processactivetime1H'
    Original Eqn: b'Numinprocesscase1H*Timeinprocesspercase1H*10'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return numinprocesscase1h() * timeinprocesspercase1h() * 10


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


_integ_numinprocesscase1h = functions.Integ(lambda: arrivalrate1h() + finishrate1h(), lambda: 0)
