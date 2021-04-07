import matplotlib.pyplot as plt
import numpy
import numpy as np 
import csv
import pandas as pd

# initialize matrix with empty values
t = [] # colum for time
data1 = [] # column with data1
signal = []
new_average = []

# open csv file
name = 'sigD.csv' # choose your file among sigA, sigB, sigC, and sigD
A = 0.5
B = 0.5
i = 0
with open(name) as f:
    # open the csv file
    reader = csv.reader(f)
    for row in reader:
        # read the rows 1 one by one
        t.append(float(row[0])) # leftmost column
        data1.append(float(row[1])) # second column
        # data2.append(float(row[2])) # third column

# window_size = 10
# numbers = data1
# numbers_series = pd.Series(numbers)
# windows = numbers_series.rolling(window_size)
# moving_averages = windows.mean()

# moving_averages_list = moving_averages.tolist()
# without_nans = moving_averages_list[window_size - 1:]

# filtered_data  = [0,0,0,0,0,0,0,0,0]
# filtered_data.extend(without_nans)


def moving_avg(x, period):
    smoothing = 2.0 / (period + 1.0)                
    current = numpy.array(x) # take the current value as a numpy array

    previous = numpy.roll(x,1) # create a new array with all the values shifted forward
    previous[0] = x[0] # start with this exact value
    # roll will have moved the end into the beginning not what we want

    return previous + smoothing * (current - previous)


def running_mean(l, N): 
    sum = 0 
    result = list( 0 for x in l) 
 
    for i in range( 0, N ): 
        sum = sum + l[i] 
        result[i] = sum / (i+1) 
 
    for i in range( N, len(l) ): 
        sum = sum - l[i-N] + l[i] 
        result[i] = sum / N 
 
    return result


filtered_data = running_mean(data1,500)

# Here we do FFT
def fft(data,t):
    Fs = len(data)/t[len(t)-1]
    y = data # the data to make the fft from
    n = len(y)
    k = np.arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(int(n/2))] # one side frequency range
    Y = np.fft.fft(y)/n # fft computing and normalization
    Y = Y[range(int(n/2))]
    return Y

Fs = len(data1)/t[len(t)-1]
y = data1 # the data to make the fft from
n = len(y)
k = np.arange(n)
T = n/Fs
frq = k/T # two sides frequency range
frq = frq[range(int(n/2))] # one side frequency range

raw_Y = fft(data1,t)
filtered_Y = fft(filtered_data,t)
plt.loglog(frq,abs(filtered_Y),'red')
plt.loglog(frq,abs(raw_Y),'black')
plt.xlabel('Freq (Hz)')
plt.ylabel('|Y(freq)|')
plt.title('Moving Average Filter with X = 500')
plt.show()