"""
Python model "EvalforNonlinearsfdReject.py"
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
    'Added resources': 'added_resources',
    'Arrival rate1D': 'arrival_rate1d',
    'Extera assigned resources': 'extera_assigned_resources',
    'Finish rate1D': 'finish_rate1d',
    'Maximum queue lenght': 'maximum_queue_lenght',
    'Num in process cases1D': 'num_in_process_cases1d',
    'Number of rejected cases': 'number_of_rejected_cases',
    'Number of unique resources1D': 'number_of_unique_resources1d',
    'Process active time1D': 'process_active_time1d',
    'Reject rate': 'reject_rate',
    'Rejected cases per time window': 'rejected_cases_per_time_window',
    'Removed resources': 'removed_resources',
    'Service time per case1D': 'service_time_per_case1d',
    'Time in process per case1D': 'time_in_process_per_case1d',
    'Waiting time in process per case1D': 'waiting_time_in_process_per_case1d',
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
def added_resources():
    """
    Real Name: b'Added resources'
    Original Eqn: b'Extera assigned resources'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return extera_assigned_resources()


@cache('run')
def arrival_rate1d():
    """
    Real Name: b'Arrival rate1D'
    Original Eqn: b'17'
    Units: b''
    Limits: (None, None)
    Type: constant

    b''
    """
    return 17


@cache('run')
def extera_assigned_resources():
    """
    Real Name: b'Extera assigned resources'
    Original Eqn: b'0'
    Units: b''
    Limits: (None, None)
    Type: constant

    b''
    """
    return 0


@cache('step')
def finish_rate1d():
    """
    Real Name: b'Finish rate1D'
    Original Eqn: b'MAX(IF THEN ELSE(Number of unique resources1D*Service time per case1D>Num in process cases1D\\\\ , Num in process cases1D, Number of unique resources1D*Service time per case1D),0)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return np.maximum(
        functions.if_then_else(
            number_of_unique_resources1d() * service_time_per_case1d() > num_in_process_cases1d(),
            num_in_process_cases1d(),
            number_of_unique_resources1d() * service_time_per_case1d()), 0)


@cache('step')
def maximum_queue_lenght():
    """
    Real Name: b'Maximum queue lenght'
    Original Eqn: b'IF THEN ELSE(Num in process cases1D>20,0,20-Num in process cases1D)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return functions.if_then_else(num_in_process_cases1d() > 20, 0, 20 - num_in_process_cases1d())


@cache('step')
def num_in_process_cases1d():
    """
    Real Name: b'Num in process cases1D'
    Original Eqn: b'INTEG ( MAX(Arrival rate1D-Finish rate1D,0), 0)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return _integ_num_in_process_cases1d()


@cache('step')
def number_of_rejected_cases():
    """
    Real Name: b'Number of rejected cases'
    Original Eqn: b'INTEG ( Reject rate, 0)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return _integ_number_of_rejected_cases()


@cache('step')
def number_of_unique_resources1d():
    """
    Real Name: b'Number of unique resources1D'
    Original Eqn: b'INTEG ( Added resources-Removed resources, 6)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return _integ_number_of_unique_resources1d()


@cache('step')
def process_active_time1d():
    """
    Real Name: b'Process active time1D'
    Original Eqn: b'Num in process cases1D*Service time per case1D'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return num_in_process_cases1d() * service_time_per_case1d()


@cache('step')
def reject_rate():
    """
    Real Name: b'Reject rate'
    Original Eqn: b'Rejected cases per time window'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return rejected_cases_per_time_window()


@cache('step')
def rejected_cases_per_time_window():
    """
    Real Name: b'Rejected cases per time window'
    Original Eqn: b'IF THEN ELSE( Arrival rate1D>Finish rate1D+Maximum queue lenght, Arrival rate1D-Finish rate1D\\\\ +Maximum queue lenght, 0 )'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return functions.if_then_else(arrival_rate1d() > finish_rate1d() + maximum_queue_lenght(),
                                  arrival_rate1d() - finish_rate1d() + maximum_queue_lenght(), 0)


@cache('run')
def removed_resources():
    """
    Real Name: b'Removed resources'
    Original Eqn: b'0'
    Units: b''
    Limits: (None, None)
    Type: constant

    b''
    """
    return 0


@cache('run')
def service_time_per_case1d():
    """
    Real Name: b'Service time per case1D'
    Original Eqn: b'2'
    Units: b''
    Limits: (None, None)
    Type: constant

    b''
    """
    return 2


@cache('step')
def time_in_process_per_case1d():
    """
    Real Name: b'Time in process per case1D'
    Original Eqn: b'Service time per case1D+Waiting time in process per case1D'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return service_time_per_case1d() + waiting_time_in_process_per_case1d()


@cache('run')
def waiting_time_in_process_per_case1d():
    """
    Real Name: b'Waiting time in process per case1D'
    Original Eqn: b'0.1'
    Units: b''
    Limits: (None, None)
    Type: constant

    b''
    """
    return 0.1


@cache('run')
def final_time():
    """
    Real Name: b'FINAL TIME'
    Original Eqn: b'30'
    Units: b'Day'
    Limits: (None, None)
    Type: constant

    b'The final time for the simulation.'
    """
    return 30


@cache('run')
def initial_time():
    """
    Real Name: b'INITIAL TIME'
    Original Eqn: b'0'
    Units: b'Day'
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
    Units: b'Day'
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
    Units: b'Day'
    Limits: (0.0, None)
    Type: constant

    b'The time step for the simulation.'
    """
    return 1


_integ_num_in_process_cases1d = functions.Integ(
    lambda: np.maximum(arrival_rate1d() - finish_rate1d(), 0), lambda: 0)

_integ_number_of_rejected_cases = functions.Integ(lambda: reject_rate(), lambda: 0)

_integ_number_of_unique_resources1d = functions.Integ(
    lambda: added_resources() - removed_resources(), lambda: 6)
