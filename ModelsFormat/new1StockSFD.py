<<<<<<< HEAD
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
    'Arrivalrate1D': 'arrivalrate1d',
    'Finishrate1D': 'finishrate1d',
    'Numofuniqueresource1D': 'numofuniqueresource1d',
    'Processactivetime1D': 'processactivetime1d',
    'Servicetimepercase1D': 'servicetimepercase1d',
    'Timeinprocesspercase1D': 'timeinprocesspercase1d',
    'Waitingtimeinprocesspercase1D': 'waitingtimeinprocesspercase1d',
    'Numinprocesscase1D': 'numinprocesscase1d',
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
def arrivalrate1d():
    """
    Real Name: b'Arrivalrate1D'
    Original Eqn: b'A FUNCTION OF(Arrivalrate1D)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return functions.incomplete()


@cache('step')
def finishrate1d():
    """
    Real Name: b'Finishrate1D'
    Original Eqn: b'A FUNCTION OF(Finishrate1D)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return functions.incomplete()


@cache('step')
def numofuniqueresource1d():
    """
    Real Name: b'Numofuniqueresource1D'
    Original Eqn: b'A FUNCTION OF(Numofuniqueresource1D)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return functions.incomplete()


@cache('step')
def processactivetime1d():
    """
    Real Name: b'Processactivetime1D'
    Original Eqn: b'A FUNCTION OF(Processactivetime1D)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return functions.incomplete()


@cache('step')
def servicetimepercase1d():
    """
    Real Name: b'Servicetimepercase1D'
    Original Eqn: b'A FUNCTION OF(Servicetimepercase1D)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return functions.incomplete()


@cache('step')
def timeinprocesspercase1d():
    """
    Real Name: b'Timeinprocesspercase1D'
    Original Eqn: b'A FUNCTION OF(Numinprocesscase1D)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return functions.incomplete()


@cache('step')
def waitingtimeinprocesspercase1d():
    """
    Real Name: b'Waitingtimeinprocesspercase1D'
    Original Eqn: b'A FUNCTION OF(Timeinprocesspercase1D,Waitingtimeinprocesspercase1D)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return functions.incomplete(waitingtimeinprocesspercase1d())


@cache('step')
def numinprocesscase1d():
    """
    Real Name: b'Numinprocesscase1D'
    Original Eqn: b'A FUNCTION OF()'
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
=======
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
    'Arrivalrate1D': 'arrivalrate1d',
    'Finishrate1D': 'finishrate1d',
    'Numofuniqueresource1D': 'numofuniqueresource1d',
    'Processactivetime1D': 'processactivetime1d',
    'Servicetimepercase1D': 'servicetimepercase1d',
    'Timeinprocesspercase1D': 'timeinprocesspercase1d',
    'Waitingtimeinprocesspercase1D': 'waitingtimeinprocesspercase1d',
    'Numinprocesscase1D': 'numinprocesscase1d',
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
def arrivalrate1d():
    """
    Real Name: b'Arrivalrate1D'
    Original Eqn: b'A FUNCTION OF(Arrivalrate1D)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return functions.incomplete()


@cache('step')
def finishrate1d():
    """
    Real Name: b'Finishrate1D'
    Original Eqn: b'A FUNCTION OF(Finishrate1D)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return functions.incomplete()


@cache('step')
def numofuniqueresource1d():
    """
    Real Name: b'Numofuniqueresource1D'
    Original Eqn: b'A FUNCTION OF(Numofuniqueresource1D)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return functions.incomplete()


@cache('step')
def processactivetime1d():
    """
    Real Name: b'Processactivetime1D'
    Original Eqn: b'A FUNCTION OF(Processactivetime1D)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return functions.incomplete()


@cache('step')
def servicetimepercase1d():
    """
    Real Name: b'Servicetimepercase1D'
    Original Eqn: b'A FUNCTION OF(Servicetimepercase1D)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return functions.incomplete()


@cache('step')
def timeinprocesspercase1d():
    """
    Real Name: b'Timeinprocesspercase1D'
    Original Eqn: b'A FUNCTION OF(Numinprocesscase1D)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return functions.incomplete()


@cache('step')
def waitingtimeinprocesspercase1d():
    """
    Real Name: b'Waitingtimeinprocesspercase1D'
    Original Eqn: b'A FUNCTION OF(Timeinprocesspercase1D,Waitingtimeinprocesspercase1D)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return functions.incomplete(waitingtimeinprocesspercase1d())


@cache('step')
def numinprocesscase1d():
    """
    Real Name: b'Numinprocesscase1D'
    Original Eqn: b'A FUNCTION OF()'
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
>>>>>>> ba9fe68d53340f50e0572b489d2912038af8c351
