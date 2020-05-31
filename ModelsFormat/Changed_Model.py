<<<<<<< HEAD
"""
Python model "Changed_Model.py"
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
    'done by a': 'done_by_a',
    'act_Register_avg_arrival': 'act_register_avg_arrival',
    'done by b': 'done_by_b',
    'act_Register_avg_duration': 'act_register_avg_duration',
    'speed of a': 'speed_of_a',
    'speed of b': 'speed_of_b',
    'act a avg waiting events': 'act_a_avg_waiting_events',
    'act b avg waiting events': 'act_b_avg_waiting_events',
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
def done_by_a():
    """
    Real Name: b'done by a'
    Original Eqn: b'speed of a'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return speed_of_a()


@cache('run')
def act_register_avg_arrival():
    """
    Real Name: b'act_Register_avg_arrival'
    Original Eqn: b'2'
    Units: b''
    Limits: (None, None)
    Type: constant

    b''
    """
    return 2


@cache('step')
def done_by_b():
    """
    Real Name: b'done by b'
    Original Eqn: b'speed of b'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return speed_of_b()


@cache('step')
def act_register_avg_duration():
    """
    Real Name: b'act_Register_avg_duration'
    Original Eqn: b'INTEG ( done by b, 0)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return _integ_act_register_avg_duration()


@cache('run')
def speed_of_a():
    """
    Real Name: b'speed of a'
    Original Eqn: b'2.25'
    Units: b''
    Limits: (None, None)
    Type: constant

    b''
    """
    return 2.25


@cache('run')
def speed_of_b():
    """
    Real Name: b'speed of b'
    Original Eqn: b'1.88'
    Units: b''
    Limits: (None, None)
    Type: constant

    b''
    """
    return 1.88


@cache('step')
def act_a_avg_waiting_events():
    """
    Real Name: b'act a avg waiting events'
    Original Eqn: b'INTEG ( MAX(act_Register_avg_arrival-done by a,0), 0)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return _integ_act_a_avg_waiting_events()


@cache('step')
def act_b_avg_waiting_events():
    """
    Real Name: b'act b avg waiting events'
    Original Eqn: b'INTEG ( MAX(done by a-done by b,0), 0)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return _integ_act_b_avg_waiting_events()


@cache('run')
def final_time():
    """
    Real Name: b'FINAL TIME'
    Original Eqn: b'5'
    Units: b'Hour'
    Limits: (None, None)
    Type: constant

    b'The final time for the simulation.'
    """
    return 5


@cache('run')
def initial_time():
    """
    Real Name: b'INITIAL TIME'
    Original Eqn: b'0'
    Units: b'Hour'
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
    Units: b'Hour'
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
    Units: b'Hour'
    Limits: (0.0, None)
    Type: constant

    b'The time step for the simulation.'
    """
    return 1


_integ_act_register_avg_duration = functions.Integ(lambda: done_by_b(), lambda: 0)

_integ_act_a_avg_waiting_events = functions.Integ(
    lambda: np.maximum(act_register_avg_arrival() - done_by_a(), 0), lambda: 0)

_integ_act_b_avg_waiting_events = functions.Integ(lambda: np.maximum(done_by_a() - done_by_b(), 0),
                                                  lambda: 0)
=======
"""
Python model "Changed_Model.py"
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
    'done by a': 'done_by_a',
    'act_Register_avg_arrival': 'act_register_avg_arrival',
    'done by b': 'done_by_b',
    'act_Register_avg_duration': 'act_register_avg_duration',
    'speed of a': 'speed_of_a',
    'speed of b': 'speed_of_b',
    'act a avg waiting events': 'act_a_avg_waiting_events',
    'act b avg waiting events': 'act_b_avg_waiting_events',
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
def done_by_a():
    """
    Real Name: b'done by a'
    Original Eqn: b'speed of a'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return speed_of_a()


@cache('run')
def act_register_avg_arrival():
    """
    Real Name: b'act_Register_avg_arrival'
    Original Eqn: b'2'
    Units: b''
    Limits: (None, None)
    Type: constant

    b''
    """
    return 2


@cache('step')
def done_by_b():
    """
    Real Name: b'done by b'
    Original Eqn: b'speed of b'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return speed_of_b()


@cache('step')
def act_register_avg_duration():
    """
    Real Name: b'act_Register_avg_duration'
    Original Eqn: b'INTEG ( done by b, 0)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return _integ_act_register_avg_duration()


@cache('run')
def speed_of_a():
    """
    Real Name: b'speed of a'
    Original Eqn: b'2.25'
    Units: b''
    Limits: (None, None)
    Type: constant

    b''
    """
    return 2.25


@cache('run')
def speed_of_b():
    """
    Real Name: b'speed of b'
    Original Eqn: b'1.88'
    Units: b''
    Limits: (None, None)
    Type: constant

    b''
    """
    return 1.88


@cache('step')
def act_a_avg_waiting_events():
    """
    Real Name: b'act a avg waiting events'
    Original Eqn: b'INTEG ( MAX(act_Register_avg_arrival-done by a,0), 0)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return _integ_act_a_avg_waiting_events()


@cache('step')
def act_b_avg_waiting_events():
    """
    Real Name: b'act b avg waiting events'
    Original Eqn: b'INTEG ( MAX(done by a-done by b,0), 0)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return _integ_act_b_avg_waiting_events()


@cache('run')
def final_time():
    """
    Real Name: b'FINAL TIME'
    Original Eqn: b'5'
    Units: b'Hour'
    Limits: (None, None)
    Type: constant

    b'The final time for the simulation.'
    """
    return 5


@cache('run')
def initial_time():
    """
    Real Name: b'INITIAL TIME'
    Original Eqn: b'0'
    Units: b'Hour'
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
    Units: b'Hour'
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
    Units: b'Hour'
    Limits: (0.0, None)
    Type: constant

    b'The time step for the simulation.'
    """
    return 1


_integ_act_register_avg_duration = functions.Integ(lambda: done_by_b(), lambda: 0)

_integ_act_a_avg_waiting_events = functions.Integ(
    lambda: np.maximum(act_register_avg_arrival() - done_by_a(), 0), lambda: 0)

_integ_act_b_avg_waiting_events = functions.Integ(lambda: np.maximum(done_by_a() - done_by_b(), 0),
                                                  lambda: 0)
>>>>>>> ba9fe68d53340f50e0572b489d2912038af8c351
