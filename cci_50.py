# Commodity Channel Index Python Code

# Load the necessary packages and modules
from pandas_datareader import data as pdr
import numpy as np
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import fix_yahoo_finance
import pandas as pd

# Commodity Channel Index 
def CCI(data, ndays): 
 TP = (data['High'] + data['Low'] + data['Close']) / 3 
 CCI = pd.Series((TP - TP.rolling(ndays).mean()) / (0.015 * TP.rolling(ndays).std()),
 name = 'CCI') 
 data = data.join(CCI) 
 return data


company_ticker = "SBIN.NS"
# Retrieve the Nifty data from Yahoo finance:
#data = pdr.get_data_yahoo("TITAN.NS", start="2017-09-25", end="2018-09-25") 
data = pdr.get_data_yahoo("SBIN.NS", start="2017-09-27", end="2018-09-25") 
#data = pd.read_csv("/home/jarvis/Documents/Verzeo/stocks/BHARTIARTLALLN.csv")
data = pd.DataFrame(data)

# Compute the Commodity Channel Index(CCI) for NIFTY based on the 20-day Moving average
n = 50
NIFTY_CCI = CCI(data, n)
CCI = NIFTY_CCI['CCI']
print("data")
print(data)
print("----------------------")
print("CCI")
print(CCI)
# Plotting the Price Series chart and the Commodity Channel index below
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(2, 1, 1)
ax.set_xticklabels([])
plt.plot(data['Close'],lw=1)
plt.title(company_ticker + " " + 'NSE Price Chart')
plt.ylabel('Close Price')
plt.grid(True)
bx = fig.add_subplot(2, 1, 2)

li = []
index = []
for i,row in CCI.iteritems():
	index.append(i)
	li.append(row)

count = 0

points1_li = []
index1_li = []

points2_li = []
index2_li = []
	
for i in range(len(li)-1):
	if(((li[i]<-100)and(li[i+1]>-100))or((li[i]>-100)and(li[i+1]<-100))):	
		count+=1
		if(((li[i]<-100)and(li[i+1]>-100))):#increasing
			points1_li.append(li[i+1])
			index1_li.append(index[i+1])
		elif(((li[i]>-100)and(li[i+1]<-100))):#decreasing
			points1_li.append(li[i])
			index1_li.append(index[i])					
	elif(((li[i]<100)and(li[i+1]>100))or((li[i]>100)and(li[i+1]<100))):	
		count+=1
		if((li[i]<100)and(li[i+1]>100)):#increasing
			points2_li.append(int(li[i]))
			index2_li.append(index[i])
		elif((li[i]>100)and(li[i+1]<100)):#decreasing
			points2_li.append(int(li[i+1]))
			index2_li.append(li[i+1])			
						
print("count",count)

plt.ylabel('CCI values')
plt.grid(True)
#plt.setp(plt.gca().get_xticklabels(), rotation=30)
plt.plot(CCI,linestyle='-',lw=0.75,label='CCI')
plt.scatter(index1_li,points1_li,s=20,label = "at -100")
print(index2_li[0])
print(points2_li[0])
plt.scatter(index2_li,points2_li,s=80,label = "at + 100")
plt.legend(loc=2,prop={'size':9.5})
plt.show()


