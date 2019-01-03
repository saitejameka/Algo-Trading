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
import datetime
from datetime import datetime


# Exponentially-weighted Moving Average 
def EWMA(data, ndays): 
 EMA = pd.Series(pd.ewma(data['Close'], span = ndays, min_periods = ndays - 1), 
 name = 'EWMA_' + str(ndays)) 
 data = data.join(EMA) 
 return data


start="2018-05-27"
end="2018-09-27"
temp = "HINDPETRO.NS"
# Retrieve the Nifty data from Yahoo finance:
data = pdr.get_data_yahoo(temp, start="2017-09-27", end="2018-09-27") 
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

plt.show()
