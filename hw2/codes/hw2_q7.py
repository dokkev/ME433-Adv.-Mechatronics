import matplotlib.pyplot as plt
import numpy
import numpy as np 
import csv
import hlist

# initialize matrix with empty values
t = [] # colum for time
data1 = [] # column with data1
signal = []
data2 = []
new_average = []
H = hlist.h
H_size = len(H)

# open csv file
name = 'sigD.csv' # choose your file among sigA, sigB, sigC, and sigD
with open(name) as f:
    # open the csv file
    reader = csv.reader(f)
    for row in reader:
        # read the rows 1 one by one
        t.append(float(row[0])) # leftmost column
        data1.append(float(row[1])) # second column
        # data2.append(float(row[2])) # third column


def FIR_filter(data):
    
    data_size = len(data1)
    temp_data = data.copy()
    for i in range(H_size):
        temp_data.append(data[-1])

    for i in range(data_size):
        values = temp_data[i:i+H_size]
        new_val = np.average(values, weights=H)
        data2.append(new_val)
    return data2


# Here we do FFT
def fft(data,t):
    Fs = len(data)/t[len(t)-1]
    print(Fs)
    y = data # the data to make the fft from
    n = len(y)
    k = np.arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(int(n/2))] # one side frequency range
    Y = np.fft.fft(y)/n # fft computing and normalization
    Y = Y[range(int(n/2))]
    return Y

filtered_data = FIR_filter(data1)
raw_Y = fft(data1,t)
filtered_Y = fft(filtered_data,t)

Fs = len(data1)/t[len(t)-1]
y = data1 
n = len(y)
k = np.arange(n)
T = n/Fs
frq = k/T 
frq = frq[range(int(n/2))]

title = "f = 400, fL = 24, bL = 32 FIR signal"

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(t,data1,'black')
ax1.plot(t,filtered_data,'red')
ax1.set_xlabel('Time')
ax1.set_ylabel('Amplitude')
ax1.title.set_text(title)
ax2.loglog(frq,abs(raw_Y),'black')
ax2.loglog(frq,abs(filtered_Y),'red')
ax2.set_xlabel('Freq (Hz)')
ax2.set_ylabel('|Y(freq)|')
plt.show()