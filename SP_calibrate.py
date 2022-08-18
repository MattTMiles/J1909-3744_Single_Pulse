import numpy
import matplotlib.pyplot as plt
import os
import subprocess as sproc 
import glob
import datetime

file_path = '/fred/oz002/users/mmiles/SinglePulse/snr_normal_window/strong_8sec/J1909-3744/2020-02-13-03:05:04/1/1284'

jones_path = '/fred/oz005/users/aparthas/reprocessing_MK/poln_calibration'

def get_calibrator(archive_utc,calib_utcs):
    archive_utc = datetime.datetime.strptime(archive_utc, '%Y-%m-%d-%H:%M:%S')
    time_diff = []
    cals_tocheck=[]
    for calib_utc in calib_utcs:
        utc = os.path.split(calib_utc)[1].split('.jones')[0]
        utc = datetime.datetime.strptime(utc, '%Y-%m-%d-%H:%M:%S')
        if (archive_utc-utc).total_seconds() >0:
            time_diff.append(calib_utc)
    return str(time_diff[-1])

for add_archive in sorted(os.listdir(file_path)):

    calib_utcs = sorted(glob.glob(os.path.join(jones_path,'*jones')))
    archive_utc = os.path.split(add_archive)[1].split('.ar')[0]
    calibrator_archive = get_calibrator(archive_utc,calib_utcs)
    if not os.path_exists(os.path.join(file_path),"{0}.calib")