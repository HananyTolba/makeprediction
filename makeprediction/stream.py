import csv
import time
import datetime
from makeprediction.tools import Time
import numpy as np
import logging

def time_series_stream(function:callable, freq:int =1, filename:str = 'filename', fieldnames:list = ["date", "value"], random_sleep=False, sig_noise = 0):



    filename = f'{filename}.csv'

    with open(filename, 'w') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

    try:
        print('Start data stream ...')
        while True:

            with open(filename, 'a') as csv_file:
                x = datetime.datetime.now()
                x = x.replace(microsecond=0) + datetime.timedelta(seconds=freq)

                dt = Time.date2num(x)
                # line = dt - date2num(x.strftime('%Y-%m-%d'))
                if sig_noise:
                    y = function(dt) + sig_noise*np.random.randn(1)[0]
                else:
                    y = function(dt)

                csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

                info = {
                    fieldnames[0]: x.strftime("%m/%d/%Y %H:%M:%S"),
                    fieldnames[1]: y,
                }
                t = datetime.datetime.now()

                # if t == x + freq:
                csv_writer.writerow(info)
                # x = t

                if random_sleep:
                    time.sleep(np.random.uniform(0, 5, 1)[0])
                else:
                    time.sleep((x - t).total_seconds())
                print(f"{x} ==> {y}")
                # print((x-t).total_seconds())

    except KeyboardInterrupt:
        print('Data generation interrupted by user.')



def timeserie_generator(function:callable,sig_noise = 0):

    while True:
        t = datetime.datetime.now()
        # x = x.replace(microsecond=0) + datetime.timedelta(seconds=freq)
        x = Time.date2num(t)
        if sig_noise:
            y = function(x) + sig_noise*np.random.randn(1)[0]
        else:
            y = function(x)
        yield t, y