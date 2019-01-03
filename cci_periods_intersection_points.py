# Commodity Channel Index Python Code

# Load the necessary packages and modules
import xlrd
import math
import time
import matplotlib.dates as mdates
from datetime import datetime
from pandas_datareader import data as pdr
import numpy as np
import matplotlib.pyplot as plt
from numpy import nan
from mpl_finance import candlestick_ohlc
import fix_yahoo_finance
import pandas as pd

# Commodity Channel Index 
def CCI(data, ndays): 
 TP = (data['High'] + data['Low'] + data['Close']) / 3 
 CCI = pd.Series((TP - TP.rolling(ndays).mean()) / (0.013 * TP.rolling(ndays).std()),
 name = 'CCI') 
 data = data.join(CCI)  
 return data


# Retrieve the Nifty data from Yahoo finance:

company_ticker = "HINDPETRO.NS"
data = pdr.get_data_yahoo(company_ticker, start="2017-09-27", end="2018-09-27") 
#data = pdr.get_data_yahoo(company_ticker, start="2017-09-27", end="2018-09-25") 
#data = pd.read_csv("/home/jarvis/Documents/Verzeo/stocks/BHARTIARTLALLN.csv")


#data = pdr.get_data_yahoo(company_ticker_list[h], start="2017-09-27", end="2018-09-25") 

data = pd.DataFrame(data)

# Compute the Commodity Channel Index(CCI) for NIFTY based on the n-days Moving average

print("give the no of periods")
no_of_periods = int(input())
cci_data_list = []
for i in range(no_of_periods):
	print("give the period input")
	n = int(input())
	NIFTY_CCI = CCI(data, n)
	CCI_data = NIFTY_CCI['CCI']
	CCI_data.fillna(0, inplace=True)
	cci_data_list.append(CCI_data)
	# Plotting the Price Series chart and the Commodity Channel index below
	fig = plt.figure(figsize=(7,5))
	#fig.autofmt_xdate()
	ax = fig.add_subplot(2, 1, 1)
	ax.set_xticklabels([])
	plt.plot(data['Close'],lw=1)
	plt.title(company_ticker+ " "+'NSE Price Chart')
	plt.ylabel('Close Price')
	plt.grid(True)
	bx = fig.add_subplot(2, 1, 2)
	plt.plot(CCI_data,linestyle='-',lw=0.75,label='CCI')
	plt.legend(loc=2,prop={'size':9.5})
	plt.ylabel('CCI values')	
	plt.grid(True)

#print("cci da ta  fasdgdhgjhjh list",cci_data_list)
#print("len of cci data list",len(cci_data_list))

#for i in range(len(cci_data_list)):
#plt.plot(cci_data_list[0])
#plt.plot(cci_data_list[1])
	#print("cci_data_list","[",i,"]",cci_data_list[i])

scatter_data_list = []
scatter_date_list = []



count = 0
d_list = []
for d1,cci1 in cci_data_list[0].iteritems():
	for d2,cci2 in cci_data_list[1].iteritems():		
		count += 1
		
		

l = len(cci_data_list[0]) + len(cci_data_list[1])

cci1_values_li = []
cci1_dates_diff_days_li = []
cci1_dates_diff_li = []
previous_date_value = []
count = 0
cci1_dates_li = []
sum_untill_now = 0
for i,row in cci_data_list[0].iteritems():
	if(not np.isnan(row)):
		if(count==0):
			cci1_dates_diff_days_li.append(0)	
			cci1_dates_diff_li.append(0)
			previous_date_value.append(i)
			count = 1
			continue
		sum_untill_now += float(str(previous_date_value[0]-i)[1])	
		cci1_dates_diff_li.append(sum_untill_now)
		cci1_dates_li.append(i)
		#just_day = time_difference.apply(lambda x: pd.tslib.Timedelta(x).days)
		cci1_dates_diff_days_li.append(i)
		previous_date_value = []
		previous_date_value.append(i)
		cci1_values_li.append(row)

suma = sum_untill_now

print("len of cci1 dates and values list are")
print(len(cci1_dates_diff_li))
print(len(cci1_values_li))


for i in range(len(cci1_dates_li)):
	print(cci1_dates_li[i] ," ",cci1_dates_diff_li[i])



print("cci1_dates_diff_li")
print(cci1_dates_diff_li)
print("cci1_values_li")									
print(cci1_values_li)


cci2_values_li = []
cci2_dates_diff_days_li = []
cci2_dates_diff_li = []
previous_date_value = []
cci2_dates_li = []
count = 0
sum_untill_now = 0

for i,row in cci_data_list[1].iteritems():
	if(not np.isnan(row)):
		if(count==0):
			cci2_dates_diff_days_li.append(0)	
			cci2_dates_diff_li.append(0)
			previous_date_value.append(i)
			count = 1
			continue
		sum_untill_now += float(str(previous_date_value[0]-i)[1])	
		cci2_dates_diff_li.append(sum_untill_now)
		cci2_dates_li.append(i)
		#just_day = time_difference.apply(lambda x: pd.tslib.Ti	medelta(x).days)
		cci2_dates_diff_days_li.append(i)
		previous_date_value = []
		previous_date_value.append(i)
		cci2_values_li.append(row)
		
	
	
for i in range(len(cci2_dates_li)):
	print(cci2_dates_li[i] ," ",cci2_dates_diff_li[i])
	
	
sumb = sum_untill_now
print("len of cci2 dates and values list are")
print(len(cci2_dates_diff_li))
print(len(cci2_values_li))


print("cci2_dates_diff_li")
print(cci2_dates_diff_li)
print("cci2_values_li")
print(cci2_values_li)


'''
print("cci_data_list[0] is -=-=")
print(cci_data_list[0])
print("cci_data_list[1] is-=-=")
print(cci_data_list[1])
'''


#plt.axis([0,l,min(min(cci1_values_li),min(cci2_values_li)),max(max(cci1_values_li),max(cci2_values_li))])


plt.plot(cci1_dates_li,cci1_values_li)
plt.plot(cci2_dates_li,cci2_values_li)

		

ax = plt.subplot()
ax.autoscale_view()
diff = round(sumb-suma,0)
diff = int(diff)
print(sumb-suma)

cci2_dates_diff_li = [item+abs(sumb-suma) for item in cci2_dates_diff_li[1:]]	

plt.axis([0,l,min(min(cci_data_list[0]),min(cci_data_list[1])),max(max(cci_data_list[0]),max(cci_data_list[1]))])
plt.axis([0,l,min(min(cci_data_list[0]),min(cci_data_list[1])),max(max(cci_data_list[0]),max(cci_data_list[1]))])

#plt.plot(EWMA1,'r',lw=1, label='7-day EWMA')
#plt.plot(ewma1_dates_diff_li[1:],ewma1_values_li,'r',lw=1, label='7-day EWMA')
plt.plot(cci1_dates_diff_li[1:],cci1_values_li)

#plt.plot(EWMA2,'b', lw=1, label='30-day EWMA')
#plt.plot(ewma2_dates_diff_li,ewma2_values_li,'b', lw=1, label='30-day EWMA')
plt.plot(cci2_dates_diff_li,cci2_values_li)

plt.legend(loc=1,prop={'size':11})
plt.grid(True)
#plt.gcf().autofmt_xdate()
#plt.setp(plt.gca().get_xticklabels(), rotation=30)

##############################################################################################################################################################
e1x = cci1_dates_diff_li[1:]
e1x = e1x[abs(diff):]
e1y = cci1_values_li[abs(diff):]

e2x = cci2_dates_diff_li
e2y = cci2_values_li


print("e1 x values ")
print(e1x)
print(len(e1x))
print("e1 y values")
print(e1y)
print(len(e1y))
print("e2 x values")
print(e2x)
print(len(e2x))
print("e2 y values")
print(e2y)
print(len(e2y))


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

plt.plot(intr.keys(),intr.values(),'k.')

##############################################################################################################################################################
	
print("cci intersection points keys")	
print(intr.keys())
print("cci intersection points values")	
print(intr.values())
		
		

		

#print(d_list)
#print(scatter_data_list)
#print(scatter_date_list)
#plt.scatter(scatter_date_list,scatter_data_list,s=70)
#plt.scatter(cci_date_list[0],cci_data_list[0])
#plt.scatter(cci_date_list[1],cci_data_list[1])

plt.show()

