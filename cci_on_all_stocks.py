# Commodity Channel Index Python Code

# Load the necessary packages and modules
import xlrd
import time
import matplotlib.dates as mdates
from datetime import datetime
from pandas_datareader import data as pdr
import numpy as np
import matplotlib.pyplot as plt
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
#data = pdr.get_data_yahoo("^BHARTIARTL", start="2017-09-25", end="2018-09-25") 
#data = pdr.get_data_yahoo(company_ticker, start="2017-09-27", end="2018-09-25") 
#data = pd.read_csv("/home/jarvis/Documents/Verzeo/stocks/BHARTIARTLALLN.csv")

company_ticker_list = []
loc = ("/home/jarvis/Documents/Verzeo/stocks/NIFTY_50_TEST.xlsx")
  
wb = xlrd.open_workbook(loc) 
sheet = wb.sheet_by_index(0) 
sheet.cell_value(0, 0) 
  
for i in range(1,sheet.nrows):
	company_ticker_list.append(sheet.cell_value(i, 0)) 

h = 0
d = 0
print(len(company_ticker_list))
while(h<len(company_ticker_list)):
	print("d",d)
	d+=1
	data = pdr.get_data_yahoo(company_ticker_list[h], start="2017-09-25", end="2018-09-25") 

	data = pd.DataFrame(data)

	# Compute the Commodity Channel Index(CCI) for NIFTY based on the 20-day Moving average
	n = 14
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
	plt.title(company_ticker_list[h]+'NSE Price Chart')
	plt.ylabel('Close Price')
	plt.grid(True)
	bx = fig.add_subplot(2, 1, 2)

	li = []
	index = []
	for i,row in CCI.iteritems():
		index.append(i)
		li.append(row)

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
	while(i<(len(li)-1)):
		if((li[i]<-100)and(li[i+1]>-100)):#increasing
			j = 0
			while(j<len(li[i:len(li)-1])):
				if((li[j+i]<-100)and(li[i+j+1]>-100)):
					lis.append(li[i+j])
					indexes_li.append(index[i+j])
				if((li[i+j]<100)and(li[i+j+1]>100)):
					if(previous_li_value!=lis[-1]):
						print("store_value, first value in increasing and their dates  =",lis[-1]," ",li[i+j+1]," ",indexes_li[-1]," 						       ",index[i+j+1])				
						markers_on_i.append(lis[-1])
						markers_on_i.append(li[i+j+1])
						dates_values_i.append(indexes_li[-1])
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
						print("store_value, first value decreasing and their dates  =",lists[-1]," ",li[i+j+1]," ",indexes_li[-1]," ",index[i+j+1])
						markers_on_d.append(lists[-1])
						markers_on_d.append(li[i+j+1])
						dates_values_d.append(indexes_li[-1])
						dates_values_d.append(index[i+j+1])
						previous_li_value = lists[-1]
					#i+=j
				j+=1
		i+=1


	plt.plot(CCI,linestyle='-',lw=0.75,label='CCI')
	#plt.plot(dates_values_i,markers_on_i,color='green',marker='v',markersize=3,markevery=len(markers_on_i)+1)
	plt.scatter(dates_values_i,markers_on_i,s=200)
	#plt.plot(dates_values_d,markers_on_d,color='red',marker='v',markersize=3,markevery=len(markers_on_d)+1)
	plt.scatter(dates_values_d,markers_on_d,s=100)
	#plt.gcf().autofmt_xdate()
	
	plt.legend(loc=2,prop={'size':9.5})
	plt.ylabel('CCI values')
	plt.grid(True)
	#plt.setp(plt.gca().get_xticklabels(), rotation=30)
	fi = company_ticker_list[h]+".png"
	fig.savefig(fi)
	h+=1
	print("h is",h)
	#plt.show()
	print("after show fun")
	print("after close fig fin")
print(len(company_ticker_list))
