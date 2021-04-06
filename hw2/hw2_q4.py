import matplotlib.pyplot as plt
import numpy as np 
import csv

# initialize matrix with empty values
t = [] # colum for time
data1 = [] # column with data1

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

# Here we do FFT
Fs = len(data1)/t[len(t)-1]
Ts = 1.0/Fs; # sampling interval
ts = t # time vector
y = data1 # the data to make the fft from
n = len(y) # length of the signal
k = np.arange(n)
T = n/Fs
frq = k/T # two sides frequency range
frq = frq[range(int(n/2))] # one side frequency range
Y = np.fft.fft(y)/n # fft computing and normalization
Y = Y[range(int(n/2))]

# signal vs time and FFT
fig, (ax1, ax2) = plt.subplots(2, 1)

if name =='sigA.csv':
    ax1.title.set_text('SigA Plot for question 4')
elif name =='sigB.csv':
    ax1.title.set_text('SigB Plot for question 4')
elif name =='sigC.csv':
    ax1.title.set_text('SigC Plot for question 4')
elif name =='sigD.csv':
   ax1.title.set_text('SigD Plot for question 4')

ax1.plot(t,data1,'b')
ax1.set_xlabel('Time')
ax1.set_ylabel('Signal')
ax2.loglog(frq,abs(Y),'b') # plotting the fft
ax2.set_xlabel('Freq (Hz)')
ax2.set_ylabel('|Y(freq)|')

plt.show()
