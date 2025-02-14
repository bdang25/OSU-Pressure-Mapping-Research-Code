# Emerson Pummill (Modified by Brandon Dang)
# 2/14/2025

# This program is used to collect data from our prototype.
# Ensure that there is a "slopes.csv" file in this folder... this will be verification that the sensors have been calibrated.
# Both the number of sensors and their locations are variable, so these values must be confirmed before running this program:

# Number of sensors
#n =  

# Location of sensors
# Starting at the top left corner of the board and working from left -> right then down
# Each location should have a label, use these values to fill out the array, and add 0x4 before the middle value:
# EX. if we have two sensors at 4xAx0 and 5X8x3, the array should be:
# locations = [
# [4, 0x4A, 0], [5, 0x48, 3]
# ]

locations = [
    [1,0x49,3],[1,0x49,1],[1,0x49,2],[1,0x49,0],[3,0x4A,1],
    [2,0x49,1],[2,0x49,2],[2,0x49,0],[2,0x49,3]#,[3,0x4A,0],
    #[3,0x49,1],[3,0x49,2],[3,0x49,0],[3,0x49,3],[3,0x4A,3],
    #[4,0x49,0],[4,0x49,1],[4,0x49,2],[4,0x49,3],[3,0x4A,2]
#[3,0x49,3],[3,0x4B,3]
#[2,0x49,2]
]

import numpy as np            
import pandas as pd
import time
import csv
import board
import adafruit_tca9548a
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
from scipy import signal

#collect data
def collect_data(chan, add, p) :
    val = AnalogIn(ADS.ADS1115(tca[chan], address=add), eval(f"ADS.P{p}")).value
    #print(val)
    return val


                                                                                                                                                                                                                                                                                                 # Create I2C bus as normal
i2c = board.I2C()

# Create the TCA9548A object and give it the I2C bus
tca = adafruit_tca9548a.TCA9548A(i2c)
    
# Create the lowpass filter
b,a = signal.butter(4,300,fs=860)

#collect data from all sensors
values = []
all_values = []
temp = []
matrix = np.zeros((6,6))
#Take & store the original value of each sensor
for i in locations:
    temp.append(collect_data(i[0], i[1], i[2]))

#Take the inital, store it, use it in collect_data
while True : 
    try:
        values = []
        l = 0
        for i in locations:
            values.append(collect_data(i[0], i[1], i[2]))
            values[l] = (values[l]-temp[l])/(-28.078) #Convert value into a force
            l= l+1
        print(values)
        all_values.append(values)
        #print(all_values)
        time.sleep(.1) #delay between measurements

    except KeyboardInterrupt:
        # If the user terminates the program, create a dataframe from the collected data
        #write data to csv file
        with open('data.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            print(f"all vals: {all_values}")
            for vals in all_values:
                writer.writerow(vals)
        
        #Pass Data through a low-pass filter
        for i in locations:
            all_values = signal.lfilter(b,a,all_values)
        #Write filtered data to csv file
        with open('filtered_data.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            print(f"all vals: {all_values}")
            for vals in all_values:
                writer.writerow(vals)
        #Convert Last Value of Matrix to a 6x6 & write to csv
        for i in range(5):
            for j in range(5):
                matrix[i+1,j+1] = values[3*i+j]
                #matrix[i+1,j+1] = 1
        np.savetxt("6x6_data.csv",matrix,delimiter = ",")
        break                
