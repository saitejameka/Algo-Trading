# Commodity Channel Index Python Code

# Load the necessary packages and modules
import xlrd
import time
import pandas as pd
import matplotlib.dates as mdates
from datetime import datetime
from  more_itertools import unique_everseen
from pandas import ExcelWriter
from pandas import ExcelFile
from pandas_datareader import data as pdr
import numpy as np
from numpy import nan
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import fix_yahoo_finance


'''
#linear deviation
def linear_deviation(item):
	total = 0
	count = 0
	for i,v in item.iteritems():
		if(not np.isnan(v)):
			count+=1
			total+=int(v)
	mean = total/count
	numertr = 0
	val_mean_diff = 0
	for i,v in item.iteritems():
		if(not np.isnan(v)):
			val_mean_diff = abs(v-mean)
			numertr+=val_mean_diff
			val_mean_diff = 0	
	l_d = numertr/count
	return l_d
'''





# Commodity Channel Index 
def CCI(data, ndays): 
 TP = (data['High'] + data['Low'] + data['Close']) / 3 					
 CCI = pd.Series((TP - TP.rolling(ndays).mean()) / (constant * TP.rolling(ndays).std()),
 name = 'CCI') 
 data = data.join(CCI)  
 return data

 

'''
# Commodity Channel Index 
def CCI(data, ndays): 
 TP = (data['High'] + data['Low'] + data['Close']) / 3 
 #EMA = pd.Series(pd.ewma(data['Close'], span = ndays, min_periods = ndays - 1),name = 'EWMA_' + str(ndays))
 CCI = pd.Series((TP - TP.rolling(ndays).mean()) / (0.011 *linear_deviation(TP)),
 name = 'CCI') 
 data = data.join(CCI) 
 return data
'''


'''
# Commodity Channel Index
def CCI(data,ndays):
	TP = (data['High'] + data['Low'] + data['Close']) / 3 
	print("len and TP is",len(TP),TP)
	EMA = pd.Series(pd.ewma(data['Close'], span = ndays, min_periods = ndays - 1), 
	name = 'EWMA_' + str(ndays))
	print("len and EMA is",len(EMA),EMA)
	pd_diff_series = pd.Series([])
	count = 0
	date_li = []
	for i,row in EMA.iteritems():
		date_li.append(i)
	for v1,v2 in zip(TP,EMA):
		if((not np.isnan(v1))and(not np.isnan(v2))):
			dif = v1-v2			
			pd_diff_series = pd_diff_series.set_value(date_li[count],dif)	
			count+=1
	
	#print("pd diff series is",pd_diff_series)
	
	l_d = linear_deviation(EMA)
	CCI = pd.Series(pd_diff_series / (constant * l_d)
	name = 'CCI') 	
	#print("CCI values are-=-=",CCI)
	#print("EMA mean is",EMA.mean())
	data = data.join(CCI) 	 
	return data
'''

constant = 0.015

# Retrieve the Nifty data from Yahoo finance:

'''
SBIN.NS
LUPIN.NS
AXISBANK.NS
SUNPHARMA.NS
CIPLA.NS
HINDALCO.NS
UPL.NS
NTPC.NS
INFY.NS
ITC.NS
 - BHARTIARTL.NS
TECHM.NS
ONGC.NS
ICICIBANK.NS
TITAN.NS
ZEEL.NS
TATAMOTORS.NS
IOC.NS
VEDL.NS
M&M.NS
BPCL.NS
TATASTEEL.NS
WIPRO.NS
ADANIPORTS.NS
 - HINDPETRO.NS
COALINDIA.NS
POWERGRID.NS
YESBANK.NS
GAIL.NS
INFRATEL.NS
'''


company_ticker = "BHARTIARTL.NS"
print(company_ticker)
#data = pdr.get_data_yahoo(company_ticker, start="2017-09-25", end="2018-09-25")
data = pdr.get_data_yahoo(company_ticker, start="2017-09-27", end="2018-09-25")
#data = pd.read_csv("/home/jarvis/Documents/Verzeo/stocks/BHARTIARTLALLN.csv")

#data = pdr.get_data_yahoo(company_ticker_list[h], start="2017-09-27", end="2018-09-25")

data = pd.DataFrame(data)

# Compute the Commodity Channel Index(CCI) for NIFTY based on the 20-day Moving average
n = 14
NIFTY_CCI = CCI(data, n)
CCI = NIFTY_CCI['CCI']
#print("data")
#print(data)
#print("----------------------")
#print("CCI")
#print(CCI)
# Plotting the Price Series chart and the Commodity Channel index below
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(2, 1, 1)
ax.set_xticklabels([])
plt.plot(data['Close'],lw=1)
plt.title(str(constant)+ " " + company_ticker+ " "+'NSE Price Chart')
plt.ylabel('close prices')
plt.grid(True)
bx = fig.add_subplot(2, 1, 2)

li = []
index = []
for i,row in CCI.iteritems():
	index.append(i)
	li.append(row)


closeprice_dict = {}
closeprice_dates_li = []
closeprice_dates_li = []
closeprice_values_li = []
for i,row in data['Close'].iteritems():
	closeprice_dict[str(i)] = row
	closeprice_dates_li.append(i)
	closeprice_values_li.append(row)
	
	
	
#print(index)
#print(li)

count = 0
for i in range(len(li)-1):
	if(((li[i]<-100)and(li[i+1]>-100))or((li[i]>-100)and(li[i+1]<-100))):
		count+=1
	if(((li[i]<100)and(li[i+1]>100))or((li[i]>100)and(li[i+1]<100))):
		count+=1
print("count",count)

i = 0
lis = []
previous_li_value = 0
j = 0
markers_on_i = []
dates_values_i = []
indexes_li = []
inc_dec_li = []
inc_dec_index_li = []
condition_count_list = []
while(i<(len(li)-1)):
	if((li[i]<-100)and(li[i+1]>-100)):#increasing
		j = 0
		while(j<len(li[i:len(li)-1])):
			if((li[j+i]<-100)and(li[i+j+1]>-100)):#increasing
				lis.append(li[i+j])
				indexes_li.append(index[i+j])
			if((li[i+j]<100)and(li[i+j+1]>100)):
				k = 0
				flag = 0
				if(previous_li_value!=lis[-1]):
					condition_count_list.append(str(lis[-1])+" "+str(li[i+j+1])+" "+str(indexes_li[-1])+" "+str(index[i+j+1]))
					markers_on_i.append(lis[-1])
					markers_on_i.append(li[i+j+1])
					dates_values_i.append(indexes_li[-1])
					dates_values_i.append(index[i+j+1])
					previous_li_value = lis[-1]
					while(k<len(li[i+j:len(li)-1])):
						if((flag==0)and(li[k+j+i]>100)and(li[k+i+j+1]<100)):
							inc_dec_li.append(li[k+j+i])
							inc_dec_index_li.append(index[i+j+k])
							flag = 1
						k+=1

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
dec_inc_li = []
dec_inc_index_li = []
while(i<(len(li)-1)):
	if((li[i]>100)and(li[i+1]<100)):#decreasing
		j = 0
		while(j<len(li[i:len(li)-1])):
			if((li[j+i]>100)and(li[i+j+1]<100)):#decreasing
				lists.append(li[i+j])
				indexes_li.append(index[i+j])
			if((li[i+j]>-100)and(li[i+j+1]<-100)):
				k = 0
				flag = 0
				if(previous_li_value!=lists[-1]):
					condition_count_list.append(str(lis[-1])+" "+str(li[i+j+1])+" "+str(indexes_li[-1])+" "+str(index[i+j+1]))
					markers_on_d.append(lists[-1])
					markers_on_d.append(li[i+j+1])
					dates_values_d.append(indexes_li[-1])
					dates_values_d.append(index[i+j+1])
					previous_li_value = lists[-1]
					while((flag==0)and(k<len(li[i+j:len(li)-1]))):
						if((li[k+j+i]<-100)and(li[k+i+j+1]>-100)):
							dec_inc_li.append(li[k+j+i])
							dec_inc_index_li.append(index[k+j+i])
							flag = 1
						k+=1
				#i+=j
			j+=1
	i+=1


print("------------------------------------------------------------------------------------------------------------------------------------------------------------------")
plt.plot(CCI,linestyle='-',lw=0.75,label='CCI')
#print(dates_values_i,markers_on_i)
#print(dates_values_d,markers_on_d)


#plt.scatter(dates_values_i,markers_on_i,color='green',marker='v')
#plt.scatter(dates_values_d,markers_on_d,color='red',marker='v')
	
plt.scatter(dates_values_i,markers_on_i,s=40,label="increasing")
plt.scatter(dates_values_d,markers_on_d,s=60,label="decreasing")
dates_corresponding_rows_indexes = []
indexes_corresponding_rows_dates_inc = []
indexes_corresponding_rows_dates_dec = []
cci_li = []
closeprice_condition_li = []
closeprice_condition_indexes_li = []

print("------------------------------------------------------------------------------------------------------------------------------------------------------------------")
u = 0

for i in dates_values_i:
	dates_corresponding_rows_indexes.append(index.index(i))	
	closeprice_condition_indexes_li.append(closeprice_dates_li.index(i))
	closeprice_condition_li.append(closeprice_values_li[closeprice_dates_li.index(i)])
	cci_li.append(li[index.index(i)])

print("closeprice_condition_indexes_li",closeprice_condition_indexes_li)
print("closeprice_condition_li",closeprice_condition_li)

l = int(len(set(cci_li)))
u = int(len(set(dates_corresponding_rows_indexes)))
cp_condition_li_len = int(len(set(closeprice_condition_indexes_li)))


for i in dates_corresponding_rows_indexes[0:u]:
	indexes_corresponding_rows_dates_inc.append(index[int(i)])

closeprice_condition_duplicates_removed_li_inc = []	
for i in closeprice_condition_li[0:u]:
	closeprice_condition_duplicates_removed_li_inc.append(i)

print("closeprice_condition_duplicates_removed_li for increasing is",closeprice_condition_duplicates_removed_li_inc)

print("indices corresponding rows dates for increasing",indexes_corresponding_rows_dates_inc)
print("cci values for increasing ",cci_li[0:l])
print("",)


cci_li = []
closeprice_condition_li = []
closeprice_condition_indexes_li = []
dates_corresponding_rows_indexes = []

print("------------------------------------------------------------------------------------------------------------------------------------------------------------------")	

indexes_corresponding_rows_dates_dec = []

for i in dates_values_d:
	dates_corresponding_rows_indexes.append(index.index(i))
	closeprice_condition_indexes_li.append(closeprice_dates_li.index(i))
	closeprice_condition_li.append(closeprice_values_li[closeprice_dates_li.index(i)])
	cci_li.append(li[index.index(i)])
	
l = int(len(set(cci_li)))
u = int(len(set(dates_corresponding_rows_indexes)))

for i in dates_corresponding_rows_indexes[0:u]:
	indexes_corresponding_rows_dates_dec.append(index[int(i)])

closeprice_condition_duplicates_removed_li_dec = []	
for i in closeprice_condition_li[0:u]:
	closeprice_condition_duplicates_removed_li_dec.append(i)

print("closeprice_condition_duplicates_removed_li for decreasing is",closeprice_condition_duplicates_removed_li_dec)

print("dates corresponding rows indexes for decreasing",indexes_corresponding_rows_dates_dec)
print("cci values for decreasing ",cci_li[0:l])
print("------------------------------------------------------------------------------------------------------------------------------------------------------------------")

#l = int((len(set(cci_li))-3)/2)
l = int(len(set(cci_li)))
print("cci values",cci_li[0:l])
#plt.gcf().autofmt_xdate()

plt.legend(loc=2,prop={'size':6.5})
plt.ylabel('CCI values')
plt.grid(True)
plt.show()
