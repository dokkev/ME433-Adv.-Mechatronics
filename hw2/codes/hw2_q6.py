import matplotlib.pyplot as plt
import numpy
import numpy as np 
import csv

# initialize matrix with empty values
t = [] # colum for time
data1 = [] # column with data1
signal = []
new_average = []

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


A = 0.1
B = 0.9

def IIR_filter(data):

    filter = [data[0]]
    for i in range(1,len(data)):
        new_val = A*data[i] + B*filter[-1]
        filter.append(new_val)
    return filter


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


filtered_data1 = running_mean(data1,500)
filtered_data2 = IIR_filter(filtered_data1)
raw_Y = fft(data1,t)
filtered_Y = fft(filtered_data2,t)

Fs = len(data1)/t[len(t)-1]
y = data1 # the data to make the fft from
n = len(y)
k = np.arange(n)
T = n/Fs
frq = k/T # two sides frequency range
frq = frq[range(int(n/2))] # one side frequency range

title = "A:" + str(A) + " B:" + str(B)  + " Filtered IIR Signal with " + name

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(t,data1,'black')
ax1.plot(t,filtered_data2,'red')
ax1.set_xlabel('Time')
ax1.set_ylabel('Amplitude')
ax1.title.set_text(title)
ax2.loglog(frq,abs(raw_Y),'black')
ax2.loglog(frq,abs(filtered_Y),'red')
ax2.set_xlabel('Freq (Hz)')
ax2.set_ylabel('|Y(freq)|')
plt.show()