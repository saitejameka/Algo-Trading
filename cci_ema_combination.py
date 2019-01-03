# Moving Averages Code

# Load the necessary packages and modules
from pandas_datareader import data as pdr
from numpy import *
import matplotlib.pyplot as plt
import fix_yahoo_finance
import pandas as pd
import matplotlib
from matplotlib import pyplot
from matplotlib import dates
import numpy as np
import datetime
from datetime import datetime
import xlrd
import time
import matplotlib.dates as mdates
from datetime import datetime
from pandas_datareader import data as pdr
from numpy import nan
from mpl_finance import candlestick_ohlc



# Exponentially-weighted Moving Average 
def EWMA(data, ndays): 
 EMA = pd.Series(pd.ewma(data['Close'], span = ndays, min_periods = ndays - 1), 
 name = 'EWMA_' + str(ndays)) 
 data = data.join(EMA) 
 return data



# Commodity Channel Index 
def CCI(data, ndays): 
 TP = (data['High'] + data['Low'] + data['Close']) / 3 
 CCI = pd.Series((TP - TP.rolling(ndays).mean()) / (0.013 * TP.rolling(ndays).std()),
 name = 'CCI') 
 data = data.join(CCI) 
 return data



start="2018-05-27"
end="2018-09-27"
company_ticker = "HINDPETRO.NS"
# Retrieve the Nifty data from Yahoo finance:
data = pdr.get_data_yahoo(company_ticker, start="2017-09-27", end="2018-09-27") 
data = pd.DataFrame(data) 
close = data['Close']
print(data)

# Compute the 7-day EWMA for NIFTY
n = 7
EWMA_NIFTY = EWMA(data,n)
EWMA_NIFTY = EWMA_NIFTY.dropna()
EWMA1 = EWMA_NIFTY['EWMA_7']

#print("type,EWMA1 IS",type(EWMA1[0]),"abcdfghj",EWMA1)

# Compute the 30-day EWMA for NIFTY.

ew = 30
EWMA_NIFTY = EWMA(data,ew)
EWMA_NIFTY = EWMA_NIFTY.dropna()
EWMA2 = EWMA_NIFTY['EWMA_30']

#print("type, EWMA2 IS",type(EWMA2[0]),"defrfghjk",EWMA2)

# Plotting the NIFTY Price Series chart and Moving Averages below
plt.figure(figsize=(9,5))
#plt.plot(data['Close'],lw=1, label=temp)


l = len(EWMA1) + len(EWMA2)

ewma1_values_li = []
ewma1_dates_diff_days_li = []
ewma1_dates_diff_li = []
previous_date_value = []
count = 0
sum_untill_now = 0
for i,row in EWMA1.iteritems():
	if(count==0):
		ewma1_dates_diff_days_li.append(0)	
		ewma1_dates_diff_li.append(0)
		previous_date_value.append(i)
		count = 1
		continue
	sum_untill_now += float(str(previous_date_value[0]-i)[1])	
	ewma1_dates_diff_li.append(sum_untill_now)
	#just_day = time_difference.apply(lambda x: pd.tslib.Timedelta(x).days)
	ewma1_dates_diff_days_li.append(i)
	previous_date_value = []
	previous_date_value.append(i)
	ewma1_values_li.append(row)

suma = sum_untill_now
print(ewma1_dates_diff_li)
print(ewma1_values_li)

ewma2_values_li = []
ewma2_dates_diff_days_li = []
ewma2_dates_diff_li = []
previous_date_value = []
count = 0
sum_untill_now = 0

for i,row in EWMA2.iteritems():
	if(count==0):
		ewma2_dates_diff_days_li.append(0)	
		ewma2_dates_diff_li.append(0)
		previous_date_value.append(i)
		count = 1
		continue
	sum_untill_now += float(str(previous_date_value[0]-i)[1])	
	ewma2_dates_diff_li.append(sum_untill_now)
	#just_day = time_difference.apply(lambda x: pd.tslib.Timedelta(x).days)
	ewma2_dates_diff_days_li.append(i)
	previous_date_value = []
	previous_date_value.append(i)
	ewma2_values_li.append(row)
	
	
	
sumb = sum_untill_now
print(len(ewma2_dates_diff_li))
print(len(ewma2_values_li))

ax = plt.subplot()
ax.autoscale_view()
diff = round(sumb-suma,0)
diff = int(diff)
print(sumb-suma)

ewma2_dates_diff_li = [item+abs(sumb-suma) for item in ewma2_dates_diff_li[1:]]	

plt.axis([0,l,min(min(EWMA1),min(EWMA2)),max(max(EWMA1),max(EWMA2))])
plt.axis([0,l,min(min(EWMA1),min(EWMA2)),max(max(EWMA1),max(EWMA2))])

#plt.plot(EWMA1,'r',lw=1, label='7-day EWMA')
#plt.plot(ewma1_dates_diff_li[1:],ewma1_values_li,'r',lw=1, label='7-day EWMA')
plt.plot(ewma1_dates_diff_li[1:],ewma1_values_li)

#plt.plot(EWMA2,'b', lw=1, label='30-day EWMA')
#plt.plot(ewma2_dates_diff_li,ewma2_values_li,'b', lw=1, label='30-day EWMA')
plt.plot(ewma2_dates_diff_li,ewma2_values_li)

plt.legend(loc=1,prop={'size':11})
plt.grid(True)
#plt.gcf().autofmt_xdate()
#plt.setp(plt.gca().get_xticklabels(), rotation=30)

##############################################################################################################################################################
e1x = ewma1_dates_diff_li[1:]
e1x = e1x[abs(diff):]
e1y = ewma1_values_li[abs(diff):]

e2x = ewma2_dates_diff_li
e2y = ewma2_values_li


if(ewma1_dates_diff_li[1:]==ewma2_dates_diff_li):
	print("both emaa 1 and ema2  dates dddiff are equal")
else:
	print("not same --------------=====================-=-=-")
	print(ewma1_dates_diff_li[1:])
	print(ewma2_dates_diff_li)


intr = dict()
for i in range(len(e1x)-1):
    if e1x[i] in e2x and e1x[i+1] in e2x:
        j = e2x.index(e1x[i])
        if (e1y[i] > e2y[j] and e1y[i+1] < e2y[j+1]) or (e1y[i] < e2y[j] and e1y[i+1] > e2y[j+1]): 
            y1 = (e1y[i+1]- e1y[i]) / (e1x[i+1] - e1x[i])
            y2 = (e2y[j+1]- e2y[j]) / (e2x[j+1] - e2x[j])
            b1 = (e1y[i] - (y1*e1x[i]))
            b2 = (e2y[j] - (y2*e2x[j]))
            x = (b1-b2)/ (y2-y1)
            y = y1*x + b1
            intr[x] = y
        else:
            continue
          
         
print("ema intersection keys ")
#print(intr.keys())
print("ema intersection values ")
#print(intr.values())


plt.plot(intr.keys(),intr.values(),'k.')

##############################################################################################################################################################




# Retrieve the Nifty data from Yahoo finance:

company_ticker = "HINDPETRO.NS"
data = pdr.get_data_yahoo(company_ticker, start="2017-09-27", end="2018-09-27") 
#data = pdr.get_data_yahoo(company_ticker, start="2017-09-27", end="2018-09-25") 
#data = pd.read_csv("/home/jarvis/Documents/Verzeo/stocks/BHARTIARTLALLN.csv")


#data = pdr.get_data_yahoo(company_ticker_list[h], start="2017-09-27", end="2018-09-25") 

data = pd.DataFrame(data)

# Compute the Commodity Channel Index(CCI) for NIFTY based on the 20-day Moving average
n = 14
NIFTY_CCI = CCI(data, n)
CCI = NIFTY_CCI['CCI']
print("data")
#print(data)
print("----------------------")
print("CCI")
#print(CCI)
print("close price value, cci values")
print(data['Close'])
#print(CCI)

# Plotting the Price Series chart and the Commodity Channel index below
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(2, 1, 1)
ax.set_xticklabels([])
#plt.plot(data['Close'],lw=1)
plt.title(company_ticker+ " "+'NSE Price Chart')
plt.ylabel('Close Price')
plt.grid(True)
bx = fig.add_subplot(2, 1, 2)

li = []
index = []
for i,row in CCI.iteritems():
	index.append(i)
	li.append(row)
	

cci_dates_diff_li = []
count = 0
cci_values_li = []
initial_date_value = []
for i,row in CCI.iteritems():
	if(count==0):
		date = i
		initial_date_value.append(date)
		count = 1	
	s = str(np.subtract(i,date))
	ind = s.index('d')
	cci_values_li.append(row)
	cci_dates_diff_li.append(int(s[0:ind-1]))
print(type(cci_dates_diff_li[0]))
#print(cci_dates_diff_li)



count = 0
for i in range(len(li)-1):
	if(((li[i]<-100)and(li[i+1]>-100))or((li[i]>-100)and(li[i+1]<-100))):	
		count+=1
	if(((li[i]<100)and(li[i+1]>100))or((li[i]>100)and(li[i+1]<100))):	
		count+=1
#print("count",count)

i = 0
lis = []
previous_li_value = 0
j = 0
markers_on_i = []
dates_values_i = []
dates_at_plus_100 =[]
indexes_li = []
while(i<(len(li)-1)):
	if((li[i]<-100)and(li[i+1]>-100)):#increasing
		j = 0
		while(j<len(li[i:len(li)-1])):
			if((li[j+i]<-100)and(li[i+j+1]>-100)):
				lis.append(li[i+j])
				indexes_li.append(index[i+j])
			if((li[i+j]<100)and(li[i+j+1]>100)):
				if(previous_li_value!=lis[-1]):
					#print("store_value, first value in increasing and their dates  =",lis[-1]," ",li[i+j+1]," ",indexes_li[-1]," 						       ",index[i+j+1])				
					markers_on_i.append(lis[-1])
					markers_on_i.append(li[i+j+1])
					dates_values_i.append(indexes_li[-1])
					dates_at_plus_100.append([index[i+j],index[i+j+1]])
					dates_values_i.append(index[i+j+1])
					previous_li_value = lis[-1]
				#i+=j
			j+=1
	i+=1

i = 0
lists = []
previous_li_value = 0
j = 0
indexes_li = []
markers_on_d = []
dates_values_d = []
while(i<(len(li)-1)):
	if((li[i]>100)and(li[i+1]<100)):#decreasing
		j = 0
		while(j<len(li[i:len(li)-1])):
			if((li[j+i]>100)and(li[i+j+1]<100)):#decreasing		
				lists.append(li[i+j])
				indexes_li.append(index[i+j])
			if((li[i+j]>-100)and(li[i+j+1]<-100)):
				if(previous_li_value!=lists[-1]):
					#print("store_value, first value decreasing and their dates  =",lists[-1]," ",li[i+j+1]," ",indexes_li[-1]," ",index[i+j+1])
					markers_on_d.append(lists[-1])
					markers_on_d.append(li[i+j+1])
					dates_values_d.append(indexes_li[-1])
					dates_values_d.append(index[i+j+1])
					previous_li_value = lists[-1]
				#i+=j
			j+=1
	i+=1
	



plt.plot(ewma1_dates_diff_li[1:],ewma1_values_li)
plt.plot(ewma2_dates_diff_li,ewma2_values_li)


plt.plot(cci_dates_diff_li,cci_values_li,linestyle='-',lw=0.75,label='CCI')
plt.plot(intr.keys(),intr.values(),'k.')
#plt.plot(dates_values_i,markers_on_i,color='green',marker='v',markersize=3,markevery=len(markers_on_i)+1)


scatter_dates_diff_li = []
for i in dates_values_i:
	s = str(np.subtract(i,initial_date_value[0]))
	ind = s.index('d')	
	scatter_dates_diff_li.append(int(s[0:ind-1]))


dates_diff_values_at_plus_100 = []
for dates_100 in dates_at_plus_100:
	d1 = str(np.subtract(dates_100[0],initial_date_value[0]))
	d2 = str(np.subtract(dates_100[1],initial_date_value[0]))
	ind1 = d1.index('d')
	ind2 = d2.index('d')
	if([int(d1[0:ind-1]),int(d2[0:ind-1])] not in dates_diff_values_at_plus_100):
		dates_diff_values_at_plus_100.append([int(d1[0:ind-1]),int(d2[0:ind-1])])
	

#le = len(set(dates_diff_values_at_plus_100))
#print("le is",le)
print("dates_diff_values_at_plus_100",dates_diff_values_at_plus_100)



plt.scatter(scatter_dates_diff_li,markers_on_i,s=200)

#plt.plot(dates_values_d,markers_on_d,color='red',marker='v',markersize=3,markevery=len(markers_on_d)+1)

#plt.scatter(dates_values_d,markers_on_d,s=100)

#plt.gcf().autofmt_xdate()

plt.legend(loc=2,prop={'size':9.5})
plt.ylabel('CCI values')
plt.grid(True)
plt.show()
	
