"""
Python model "model.py"
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
    'finish_rate8h': 'finish_rate8h',
    'num_of_unique_resource8h': 'num_of_unique_resource8h',
    'process_active_time8h': 'process_active_time8h',
    'service_time_per_case8h': 'service_time_per_case8h',
    'time_in_process_per_case8h': 'time_in_process_per_case8h',
    'num_in_process_case8h': 'num_in_process_case8h',
    'arrival_rate8h': 'arrival_rate8h',
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
def finish_rate8h():
    """
    Real Name: b'finish_rate8h'
    Original Eqn: b'0.5180757402007797 * 21.17982456140351 + -1.122227742423431 * 25.75657894736842 + -0.02254356228253507 * 275.4337752959692 + 0.07131680857538954 * num_in_process_case8h + 22.461295291949927'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return 0.5180757402007797 * 21.17982456140351 + -1.122227742423431 * 25.75657894736842 + -0.02254356228253507 * 275.4337752959692 + 0.07131680857538954 * num_in_process_case8h(
    ) + 22.461295291949927


@cache('step')
def num_of_unique_resource8h():
    """
    Real Name: b'num_of_unique_resource8h'
    Original Eqn: b'0.4581613958648815 * 21.17982456140351 + -0.08515221017517405 * finish_rate8h + 0.012228241637189174 * time_in_process_per_case8h + 0.010335139442016068 * num_in_process_case8h + 11.204948442231531'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return 0.4581613958648815 * 21.17982456140351 + -0.08515221017517405 * finish_rate8h(
    ) + 0.012228241637189174 * time_in_process_per_case8h(
    ) + 0.010335139442016068 * num_in_process_case8h() + 11.204948442231531


@cache('step')
def process_active_time8h():
    """
    Real Name: b'process_active_time8h'
    Original Eqn: b'191.73796695556476 * arrival_rate8h + -5.282837495250178 * finish_rate8h + 70.99598230746354 * num_of_unique_resource8h + 0.009189714880675751 * 1913.5826829972723 + 5.06485954599035 * time_in_process_per_case8h + -1.9169431280272986 * num_in_process_case8h + -876.2069273617217'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return 191.73796695556476 * arrival_rate8h() + -5.282837495250178 * finish_rate8h(
    ) + 70.99598230746354 * num_of_unique_resource8h(
    ) + 0.009189714880675751 * 1913.5826829972723 + 5.06485954599035 * time_in_process_per_case8h(
    ) + -1.9169431280272986 * num_in_process_case8h() + -876.2069273617217


@cache('step')
def service_time_per_case8h():
    """
    Real Name: b'service_time_per_case8h'
    Original Eqn: b'-36.255348706855585 * arrival_rate8h + 13.15096607735153 * finish_rate8h + 57.769799711359035 * num_of_unique_resource8h + 0.008526734613095195 * process_active_time8h + 7.532416087067163 * time_in_process_per_case8h + -0.5680445457464266 * num_in_process_case8h + -992.8423904371032'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return -36.255348706855585 * arrival_rate8h() + 13.15096607735153 * finish_rate8h(
    ) + 57.769799711359035 * num_of_unique_resource8h(
    ) + 0.008526734613095195 * process_active_time8h(
    ) + 7.532416087067163 * time_in_process_per_case8h(
    ) + -0.5680445457464266 * num_in_process_case8h() + -992.8423904371032


@cache('step')
def time_in_process_per_case8h():
    """
    Real Name: b'time_in_process_per_case8h'
    Original Eqn: b'-7.6588655932573335 * 21.17982456140351 + -0.5442915147885695 * finish_rate8h + 3.901769615235846 * 25.75657894736842 + 0.019670614564495394 * 5728.581176900585 + 0.03149377593791478 * 1913.5826829972723 + -0.0639957397342656 * num_in_process_case8h + 195.26641190689844'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return -7.6588655932573335 * 21.17982456140351 + -0.5442915147885695 * finish_rate8h(
    ) + 3.901769615235846 * 25.75657894736842 + 0.019670614564495394 * 5728.581176900585 + 0.03149377593791478 * 1913.5826829972723 + -0.0639957397342656 * num_in_process_case8h(
    ) + 195.26641190689844


@cache('step')
def num_in_process_case8h():
    """
    Real Name: b'num_in_process_case8h'
    Original Eqn: b'INTEG(1 * arrival_rate8h + -1 * finish_rate8h , 1.0)'
    Units: b''
    Limits: (None, None)
    Type: component

    b''
    """
    return _integ_num_in_process_case8h()


@cache('run')
def arrival_rate8h():
    """
    Real Name: b'arrival_rate8h'
    Original Eqn: b'0'
    Units: b''
    Limits: (None, None)
    Type: constant

    b''
    """
    return 0


@cache('run')
def final_time():
    """
    Real Name: b'FINAL TIME'
    Original Eqn: b'456'
    Units: b''
    Limits: (None, None)
    Type: constant

    b'The final time for the simulation.'
    """
    return 456


@cache('run')
def initial_time():
    """
    Real Name: b'INITIAL TIME'
    Original Eqn: b'1'
    Units: b''
    Limits: (None, None)
    Type: constant

    b'The initial time for the simulation.'
    """
    return 1


@cache('step')
def saveper():
    """
    Real Name: b'SAVEPER'
    Original Eqn: b'TIME STEP'
    Units: b''
    Limits: (None, None)
    Type: component

    b'The frequency with which output is stored.'
    """
    return time_step()


@cache('run')
def time_step():
    """
    Real Name: b'TIME STEP'
    Original Eqn: b'1'
    Units: b''
    Limits: (None, None)
    Type: constant

    b'The time step for the simulation.'
    """
    return 1


_integ_num_in_process_case8h = functions.Integ(lambda: 1 * arrival_rate8h() + -1 * finish_rate8h(),
                                               lambda: 1.0)
