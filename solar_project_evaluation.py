# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 22:00:04 2020

@author: ABHIRUP
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr
import statsmodels.api as sm

def Solar_Position_Specific_Location_Daily(lat, lon, year, month, day, tz):
    
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    def correct_range(d):
    
        if d < 0:
            if d < -360:
                d += (d // 360) * 360
            else:
                d += 360
        
        elif d > 360:
            d -= (d // 360) * 360

        return d
    
    time = []
    elev = np.zeros(1440)
    azim = np.zeros(1440)
    decli = np.zeros(1440)
    rascn = np.zeros(1440)
   
    if tz > 0:
           
        h = 24 - tz
        if h % 1 != 0:
            hour = int(h)
            minute = int((h % 1) * 60)
        else:
            hour = h
            minute = 0
            
        d = day
        if d == 1:
            if month == 1:
                year -= 1
                month = 12
                d = 31
            else:
                month -= 1
                if month == 4 or month == 6 or month == 9 or month == 11:
                    d = 30
                elif month == 2:
                    if year % 4 == 0:
                        d = 29
                    else:
                        d = 28
                else:
                    d = 31
        else:
            d -= 1
    
    elif tz < 0:
        
        h = np.abs(tz)
        if h % 1 != 0:
            hour = int(h)
            minute = int((h % 1) * 60)
        else:
            hour = h
            minute = 0
            
        d = day
        if d == 31:
            if month == 12:
                year += 1
                month = 1
                d = 1
            else:
                month += 1
                d = 1
        elif d == 30:
            if month == 4 or month == 6 or month == 9 or month == 11:
                month += 1
                d = 1
            else:
                d += 1
        elif d == 29:
            if month == 2:
                month += 1
                d = 1
            else:
                d += 1
        elif d == 28:
            if month == 2:
                if year % 4 != 0:
                    month += 1
                    d = 1
                else:
                    d += 1
            else:
                d += 1
        else:
            d += 1

    else:
        hour = 0
        minute = 0

    HH = 00
    MM = 00
    
    i = 1
    while i <= 1440:
                
        time.append(str(HH)+':'+str(MM))

        if tz > 0:
            if i <= tz*2:
                day = d
        elif tz < 0:
            if i > 48 - tz*2:
                day = d
            
            
        ts = pd.Timestamp(year = year,  month = month, day = day, hour = hour, minute = minute, second = 0)    
        JD = ts.to_julian_date()
        n = JD - 2451545.0
        T = n / 36525
        
        L = correct_range(280.46646 + (36000.76983 * T) + (0.0003032 * T**2))
        g = correct_range(357.52911 + (35999.05029 * T) - (0.0001537 * T**2))
        #e = 0.016708634 - (0.000042037 * T) - (0.0000001267 * T**2)
        C = (1.914602 - (0.004817 * T) - (0.000014 * T**2)) * np.sin(np.radians(g)) + (0.019993 - (0.000101 * T)) * np.sin(np.radians(g) * 2) + (0.000289 * np.sin(np.radians(g) * 3))
        ec_lon = correct_range(L + C)
        #g_true = correct_range(g + C)
        #R = 1.00014 - 0.01671 * np.cos(np.radians(g)) - 0.00014 * np.cos(2 * np.radians(g))
        O = 125.04 - (1934.136 * T);
        ec_lon_app = ec_lon - 0.00569 - (0.00478 * np.sin(np.radians(O)))
        U = T / 100
        e0 = 23.439291 - 1.300258 * U - 1.55 * U**2 + 1999.25 * U**3 - 51.38 * U**4 - 249.67 * U**5 - 39.05 * U**6 + 7.12 * U**7 + 27.87 * U**8 + 5.79 * U**9 + 2.45 * U**10
        eCorrected = e0 + 0.00256 * np.cos(np.radians(O));
        RA = np.arctan2(np.cos(np.radians(eCorrected)) * np.sin(np.radians(ec_lon_app)), np.cos(np.radians(ec_lon_app)))
        declination = np.arcsin(np.sin(np.radians(eCorrected)) * np.sin(np.radians(ec_lon_app)))
        
        RA_deg = np.degrees(RA)
        declination_deg = np.degrees(declination)
        
        GMST = correct_range(280.46061837 + 360.98564736629 * n + 0.000387933 * T**2 - T**3 / 38710000)
        LST = GMST + lon
        LHA = correct_range(LST - RA_deg)
        elevation = np.arcsin(np.sin(lat_rad)*np.sin(declination) + np.cos(lat_rad)*np.cos(declination)*np.cos(np.radians(LHA)))
        #azimuth = np.arcsin(np.cos(declination)*np.sin(np.radians(LHA)) / np.cos(elevation))
        azimuth = np.arccos((np.sin(declination) - np.sin(elevation)*np.sin(lat_rad)) / (np.cos(elevation)*np.cos(lat_rad)))

        elevation_deg = np.degrees(elevation)
        azimuth_deg = np.degrees(azimuth)
        
        if i > 720:
            azimuth_deg = 360 - np.degrees(azimuth)
        
        elev[i-1] = elevation_deg
        azim[i-1] = azimuth_deg
        decli[i-1] = declination_deg
        rascn[i-1] = RA_deg
              
        minute += 1       
        if minute == 60:
            hour += 1
            minute = 0
            if hour == 24:
                hour = 0

        MM += 1
        if MM == 60:
            HH += 1
            MM = 0
                    
        i += 1
        
    for j in range(710,760,1):
        if lat > np.mean(decli):
            if azim[j] < azim[j-1]:
                azim[j-1] = 360 - azim[j-1]
    
    angle_df = pd.DataFrame({
            'Time': time,
            'Elevation': elev,
            'Azimuth': azim,
            'Declination': decli,
            'Right Ascesion': rascn})
        
    return angle_df


observed_df = pd.read_csv('C:/Work/Solar Project/observed_data/Almeria_Spain_basic_19940101_20181231.csv', names=['Date','Time','GHI','DIF','GTI','flagR','SE','SA','TEMP','WS','WD'], delimiter=';', skiprows=53)

Date = observed_df.Date
T = observed_df.Time
SE = observed_df.SE
SA = observed_df.SA
GHI = observed_df.GHI


S0 = 1367
year = 2017
if year % 4 == 0:
    Dn = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
else:
    Dn = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


City_Name = 'Almeria (Spain)'    
lat = 37.094416                   
lon = -2.359850
tz = 0   

###############################################################################    
###############################################################################
###############################################################################

fig1 = plt.figure(figsize = (44, 44))

plt.subplot(4,4,1)

day = 20
month = 3
year = 2017

angle_df = Solar_Position_Specific_Location_Daily(lat, lon, year, month, day, tz)

Elevation = angle_df.Elevation
Azimuth = angle_df.Azimuth
Time = angle_df.Time
Declination = angle_df.Declination    


idxx1 = np.arange(10, 1436, 15)
Elev_model = np.zeros(96)
Azim_model = np.zeros(96)
decli_model = np.zeros(96)
for i in range(96):
    Elev_model[i] = Elevation[idxx1[i]]
    Azim_model[i] = Azimuth[idxx1[i]]
    decli_model[i] = Declination[idxx1[i]]


Zenith_model = 90 - Elev_model

for i in range(len(Zenith_model)):
    if (Zenith_model[i] > 90) or (Zenith_model[i] < 0):
        Zenith_model[i] = 90
 
GHI_model = S0 * np.cos(np.radians(Zenith_model))
       

idx = np.argwhere(Elev_model > 0)
spots = idx.shape[0]
idx = idx.reshape(spots,) 
t = np.arange(0,96,1)


if day < 10:
    d = '0'+ str(day)
else:
    d = str(day)

if month < 10:
    m = '0'+ str(month)
else:
    m = str(month)
    
date = d + '.' + m + '.' + str(year)
init = list(Date).index(date)
idx_obs = np.arange(init, init+96, 1)

Time_obs = T[idx_obs].to_list()
SE_obs = SE[idx_obs].to_numpy()
SA_obs = SA[idx_obs].to_numpy()
SA_obs += 180
GHI_obs = GHI[idx_obs].to_numpy()

idx1 = np.argwhere(SE_obs > 0)
spots1 = idx1.shape[0]
idx1 = idx1.reshape(spots1,)


if len(idx > 0):
    plt.plot(t[0:idx[1]], Elev_model[0:idx[1]], color='skyblue', lw=4)
    plt.plot(t[idx[0]-1:idx[-1]+2], Elev_model[idx[0]-1:idx[-1]+2], color='orange', lw=4)
    plt.plot(t[idx[-1]+1:], Elev_model[idx[-1]+1:], color='skyblue', lw=4)
    plt.fill_between(t, Elev_model, where=Elev_model>0, color='C1', alpha=0.4, label=f'Day, Sunrise: {Time_obs[idx[0]]}')
    plt.fill_between(t, Elev_model, where=Elev_model<0, color='C0', alpha=0.4, label=f'Night, Sunset: {Time_obs[idx[-1]]}')
    plt.plot(SE_obs, color='k', label=f'Observed\nSunrise: {Time_obs[idx1[0]]}\nSunset: {Time_obs[idx1[-1]]}')
    plt.legend(loc='best', facecolor='white')
    plt.xticks(np.arange(0, 104, 8),('00:00', '02:00', '04:00', '06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00', '22:00', '24:00'))
    plt.ylabel('Elevation (Degree)', fontweight = 'bold', fontsize = 15)
    plt.title(f'{City_Name}   {day}-{month}-{year}', fontweight = 'bold', fontsize = 20)
    plt.grid()
        
else:
    plt.plot(Elev_model, color='skyblue', lw=4)
    plt.fill_between(t, Elev_model, where=Elev_model>0, color='C1', alpha=0.4, label='Day')        
    plt.fill_between(t, Elev_model, where=Elev_model<0, color='C0', alpha=0.4, label='Night')
    plt.legend(loc='best', facecolor='white')
    plt.xlabel('Local time (hrs)', fontweight = 'bold', fontsize = 15)
    plt.ylabel('Elevation (Degree)', fontweight = 'bold', fontsize = 15)
    plt.title(f'{City_Name}   {day}-{month}-{year}', fontweight = 'bold', fontsize = 20)
    plt.grid()


plt.subplot(4,4,2)

est1 = sm.OLS(Elev_model, SE_obs)
est1 = est1.fit()
coef = np.sqrt(est1.rsquared)

plt.plot(SE_obs, Elev_model, label=f'Corr_coef = {coef:.5f}')
plt.grid()
plt.legend(loc='best')
plt.xlabel('Observed Elevation (Degree)', fontweight = 'bold', fontsize = 15)
plt.ylabel('Model Elevation (Degree)', fontweight = 'bold', fontsize = 15)
plt.title('Model vs Observed (Daily Solar Elevation)', fontweight = 'bold', fontsize = 20)


plt.subplot(4,4,3)

est2 = sm.OLS(Azim_model, SA_obs)
est2 = est2.fit()
coef2 = np.sqrt(est2.rsquared)

plt.plot(SA_obs[1:], Azim_model[1:], label=f'Corr_coef = {coef2:.5f}')
plt.grid()
plt.legend(loc='best')
plt.xlabel('Observed Azimuthn (Degree)', fontweight = 'bold', fontsize = 15)
plt.ylabel('Model Azimuth (Degree)', fontweight = 'bold', fontsize = 15)
plt.title('Model vs Observed (Daily Solar Azimuth)', fontweight='bold', fontsize=20)


plt.subplot(4,4,4)

X = GHI_obs
X = sm.add_constant(X)
y = GHI_model

est = sm.OLS(y,X)
est = est.fit()
est.summary()
est.params
coef1 = np.sqrt(est.rsquared)
conf = est.conf_int(0.0001)

const, slope = est.params

y_hat = est.predict(X)
y1 = X * conf[1,0] + conf[0,0]
y2 = X * conf[1,1] + conf[0,1]

plt.scatter(GHI_obs, GHI_model)
plt.plot(X[:,1],y_hat,'k',label=f'OLS fit\nCorr_coef = {coef1:.5f}')
plt.fill_between(X[:,1], y1[:,1], y2[:,1], alpha=0.4)
plt.legend(loc='best')
plt.grid()
plt.ylabel('Model GHI (W/m2)', fontweight = 'bold', fontsize = 15)
plt.xlabel('Observed GHI (W/m2)', fontweight = 'bold', fontsize = 15)
plt.title('Model vs Observed (GHI)', fontweight = 'bold', fontsize = 20)


###############################################################################

plt.subplot(4,4,5)

day = 21
month = 6
year = 2017


angle_df = Solar_Position_Specific_Location_Daily(lat, lon, year, month, day, tz)

Elevation = angle_df.Elevation
Azimuth = angle_df.Azimuth
Time = angle_df.Time


idxx1 = np.arange(10, 1436, 15)
Elev_model = np.zeros(96)
for i in range(96):
    Elev_model[i] = Elevation[idxx1[i]]

idx = np.argwhere(Elev_model > 0)
spots = idx.shape[0]
idx = idx.reshape(spots,) 
t = np.arange(0,96,1)


if day < 10:
    d = '0'+ str(day)
else:
    d = str(day)

if month < 10:
    m = '0'+ str(month)
else:
    m = str(month)
    
date = d + '.' + m + '.' + str(year)
init = list(Date).index(date)
idx_obs = np.arange(init, init+96, 1)

Time_obs = T[idx_obs].to_list()
SE_obs = SE[idx_obs].to_numpy()
SA_obs = SA[idx_obs].to_numpy()
SA_obs += 180
GHI_obs = GHI[idx_obs].to_numpy()

idx1 = np.argwhere(SE_obs > 0)
spots1 = idx1.shape[0]
idx1 = idx1.reshape(spots1,)

Zenith_model = 90 - Elev_model

for i in range(len(Zenith_model)):
    if (Zenith_model[i] > 90) or (Zenith_model[i] < 0):
        Zenith_model[i] = 90
 
GHI_model = S0 * np.cos(np.radians(Zenith_model))


if len(idx > 0):
    plt.plot(t[0:idx[1]], Elev_model[0:idx[1]], color='skyblue', lw=4)
    plt.plot(t[idx[0]-1:idx[-1]+2], Elev_model[idx[0]-1:idx[-1]+2], color='orange', lw=4)
    plt.plot(t[idx[-1]+1:], Elev_model[idx[-1]+1:], color='skyblue', lw=4)
    plt.fill_between(t, Elev_model, where=Elev_model>0, color='C1', alpha=0.4, label=f'Day, Sunrise: {Time_obs[idx[0]]}')
    plt.fill_between(t, Elev_model, where=Elev_model<0, color='C0', alpha=0.4, label=f'Night, Sunset: {Time_obs[idx[-1]]}')
    plt.plot(SE_obs, color='k', label=f'Observed\nSunrise: {Time_obs[idx1[0]]}\nSunset: {Time_obs[idx1[-1]]}')
    plt.legend(loc='best', facecolor='white')
    plt.xticks(np.arange(0, 104, 8),('00:00', '02:00', '04:00', '06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00', '22:00', '24:00'))
    plt.ylabel('Elevation (Degree)', fontweight = 'bold', fontsize = 15)
    plt.title(f'{City_Name}   {day}-{month}-{year}', fontweight = 'bold', fontsize = 20)
    plt.grid()
        
else:
    plt.plot(Elev_model, color='skyblue', lw=4)
    plt.fill_between(t, Elev_model, where=Elev_model>0, color='C1', alpha=0.4, label='Day')        
    plt.fill_between(t, Elev_model, where=Elev_model<0, color='C0', alpha=0.4, label='Night')
    plt.legend(loc='best', facecolor='white')
    plt.xlabel('Local time (hrs)', fontweight = 'bold', fontsize = 15)
    plt.ylabel('Elevation (Degree)', fontweight = 'bold', fontsize = 15)
    plt.title(f'{City_Name}   {day}-{month}-{year}', fontweight = 'bold', fontsize = 20)
    plt.grid()


plt.subplot(4,4,6)

est1 = sm.OLS(Elev_model, SE_obs)
est1 = est1.fit()
coef = np.sqrt(est1.rsquared)

plt.plot(SE_obs, Elev_model, label=f'Corr_coef = {coef:.5f}')
plt.grid()
plt.legend(loc='best')
plt.xlabel('Observed Elevation (Degree)', fontweight = 'bold', fontsize = 15)
plt.ylabel('Model Elevation (Degree)', fontweight = 'bold', fontsize = 15)
plt.title('Model vs Observed (Daily Solar Elevation)', fontweight = 'bold', fontsize = 20)


plt.subplot(4,4,7)

est2 = sm.OLS(Azim_model, SA_obs)
est2 = est2.fit()
coef2 = np.sqrt(est2.rsquared)

plt.plot(SA_obs[1:], Azim_model[1:], label=f'Corr_coef = {coef2:.5f}')
plt.grid()
plt.legend(loc='best')
plt.xlabel('Observed Azimuthn (Degree)', fontweight = 'bold', fontsize = 15)
plt.ylabel('Model Azimuth (Degree)', fontweight = 'bold', fontsize = 15)
plt.title('Model vs Observed (Daily Solar Azimuth)', fontweight='bold', fontsize=20)


plt.subplot(4,4,8)

X = GHI_obs
X = sm.add_constant(X)
y = GHI_model

est=sm.OLS(y,X)
est=est.fit()
est.summary()
est.params
coef1 = np.sqrt(est.rsquared)
conf = est.conf_int(0.0001)

const, slope = est.params

y_hat = est.predict(X)
y1 = X * conf[1,0] + conf[0,0]
y2 = X * conf[1,1] + conf[0,1]


plt.scatter(GHI_obs, GHI_model)
plt.plot(X[:,1],y_hat,'k',label=f'OLS fit\nCorr_coef = {coef1:.5f}')
plt.fill_between(X[:,1], y1[:,1], y2[:,1], alpha=0.4)
plt.legend(loc='best')
plt.grid()
plt.ylabel('Model GHI (W/m2)', fontweight = 'bold', fontsize = 15)
plt.xlabel('Observed GHI (W/m2)', fontweight = 'bold', fontsize = 15)
plt.title('Model vs Observed (GHI)', fontweight = 'bold', fontsize = 20)


###############################################################################

plt.subplot(4,4,9)

day = 22
month = 9
year = 2017


angle_df = Solar_Position_Specific_Location_Daily(lat, lon, year, month, day, tz)

Elevation = angle_df.Elevation
Azimuth = angle_df.Azimuth
Time = angle_df.Time


idxx1 = np.arange(10, 1436, 15)
Elev_model = np.zeros(96)
for i in range(96):
    Elev_model[i] = Elevation[idxx1[i]]

idx = np.argwhere(Elev_model > 0)
spots = idx.shape[0]
idx = idx.reshape(spots,) 
t = np.arange(0,96,1)


if day < 10:
    d = '0'+ str(day)
else:
    d = str(day)

if month < 10:
    m = '0'+ str(month)
else:
    m = str(month)
    
date = d + '.' + m + '.' + str(year)
init = list(Date).index(date)
idx_obs = np.arange(init, init+96, 1)

Time_obs = T[idx_obs].to_list()
SE_obs = SE[idx_obs].to_numpy()
SA_obs = SA[idx_obs].to_numpy()
SA_obs += 180
GHI_obs = GHI[idx_obs].to_numpy()

idx1 = np.argwhere(SE_obs > 0)
spots1 = idx1.shape[0]
idx1 = idx1.reshape(spots1,)

Zenith_model = 90 - Elev_model

for i in range(len(Zenith_model)):
    if (Zenith_model[i] > 90) or (Zenith_model[i] < 0):
        Zenith_model[i] = 90
 
GHI_model = S0 * np.cos(np.radians(Zenith_model))

if len(idx > 0):
    plt.plot(t[0:idx[1]], Elev_model[0:idx[1]], color='skyblue', lw=4)
    plt.plot(t[idx[0]-1:idx[-1]+2], Elev_model[idx[0]-1:idx[-1]+2], color='orange', lw=4)
    plt.plot(t[idx[-1]+1:], Elev_model[idx[-1]+1:], color='skyblue', lw=4)
    plt.fill_between(t, Elev_model, where=Elev_model>0, color='C1', alpha=0.4, label=f'Day, Sunrise: {Time_obs[idx[0]]}')
    plt.fill_between(t, Elev_model, where=Elev_model<0, color='C0', alpha=0.4, label=f'Night, Sunset: {Time_obs[idx[-1]]}')
    plt.plot(SE_obs, color='k', label=f'Observed\nSunrise: {Time_obs[idx1[0]]}\nSunset: {Time_obs[idx1[-1]]}')
    plt.legend(loc='best', facecolor='white')
    plt.xticks(np.arange(0, 104, 8),('00:00', '02:00', '04:00', '06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00', '22:00', '24:00'))
    plt.ylabel('Elevation (Degree)', fontweight = 'bold', fontsize = 15)
    plt.title(f'{City_Name}   {day}-{month}-{year}', fontweight = 'bold', fontsize = 20)
    plt.grid()
        
else:
    plt.plot(Elev_model, color='skyblue', lw=4)
    plt.fill_between(t, Elev_model, where=Elev_model>0, color='C1', alpha=0.4, label='Day')        
    plt.fill_between(t, Elev_model, where=Elev_model<0, color='C0', alpha=0.4, label='Night')
    plt.legend(loc='best', facecolor='white')
    plt.xlabel('Local time (hrs)', fontweight = 'bold', fontsize = 15)
    plt.ylabel('Elevation (Degree)', fontweight = 'bold', fontsize = 15)
    plt.title(f'{City_Name}   {day}-{month}-{year}', fontweight = 'bold', fontsize = 20)
    plt.grid()


plt.subplot(4,4,10)

est1 = sm.OLS(Elev_model, SE_obs)
est1 = est1.fit()
coef = np.sqrt(est1.rsquared)

plt.plot(SE_obs, Elev_model, label=f'Corr_coef = {coef:.5f}')
plt.grid()
plt.legend(loc='best')
plt.xlabel('Observed Elevation (Degree)', fontweight = 'bold', fontsize = 15)
plt.ylabel('Model Elevation (Degree)', fontweight = 'bold', fontsize = 15)
plt.title('Model vs Observed (Daily Solar Elevation)', fontweight = 'bold', fontsize = 20)


plt.subplot(4,4,11)

est2 = sm.OLS(Azim_model, SA_obs)
est2 = est2.fit()
coef2 = np.sqrt(est2.rsquared)

plt.plot(SA_obs[1:], Azim_model[1:], label=f'Corr_coef = {coef2:.5f}')
plt.grid()
plt.legend(loc='best')
plt.xlabel('Observed Azimuthn (Degree)', fontweight = 'bold', fontsize = 15)
plt.ylabel('Model Azimuth (Degree)', fontweight = 'bold', fontsize = 15)
plt.title('Model vs Observed (Daily Solar Azimuth)', fontweight='bold', fontsize=20)


plt.subplot(4,4,12)

X = GHI_obs
X = sm.add_constant(X)
y = GHI_model

est=sm.OLS(y,X)
est=est.fit()
est.summary()
est.params
coef1 = np.sqrt(est.rsquared)
conf = est.conf_int(0.0001)

const, slope = est.params

y_hat = est.predict(X)
y1 = X * conf[1,0] + conf[0,0]
y2 = X * conf[1,1] + conf[0,1]


plt.scatter(GHI_obs, GHI_model)
plt.plot(X[:,1],y_hat,'k',label=f'OLS fit\nCorr_coef = {coef1:.5f}')
plt.fill_between(X[:,1], y1[:,1], y2[:,1], alpha=0.4)
plt.legend(loc='best')
plt.grid()
plt.ylabel('Model GHI (W/m2)', fontweight = 'bold', fontsize = 15)
plt.xlabel('Observed GHI (W/m2)', fontweight = 'bold', fontsize = 15)
plt.title('Model vs Observed (GHI)', fontweight = 'bold', fontsize = 20)


###############################################################################

plt.subplot(4,4,13)

day = 21
month = 12
year = 2017


angle_df = Solar_Position_Specific_Location_Daily(lat, lon, year, month, day, tz)

Elevation = angle_df.Elevation
Azimuth = angle_df.Azimuth
Time = angle_df.Time


idxx1 = np.arange(10, 1436, 15)
Elev_model = np.zeros(96)
for i in range(96):
    Elev_model[i] = Elevation[idxx1[i]]

idx = np.argwhere(Elev_model > 0)
spots = idx.shape[0]
idx = idx.reshape(spots,) 
t = np.arange(0,96,1)


if day < 10:
    d = '0'+ str(day)
else:
    d = str(day)

if month < 10:
    m = '0'+ str(month)
else:
    m = str(month)
    
date = d + '.' + m + '.' + str(year)
init = list(Date).index(date)
idx_obs = np.arange(init, init+96, 1)

Time_obs = T[idx_obs].to_list()
SE_obs = SE[idx_obs].to_numpy()
SA_obs = SA[idx_obs].to_numpy()
SA_obs += 180
GHI_obs = GHI[idx_obs].to_numpy()

idx1 = np.argwhere(SE_obs > 0)
spots1 = idx1.shape[0]
idx1 = idx1.reshape(spots1,)

Zenith_model = 90 - Elev_model

for i in range(len(Zenith_model)):
    if (Zenith_model[i] > 90) or (Zenith_model[i] < 0):
        Zenith_model[i] = 90
 
GHI_model = S0 * np.cos(np.radians(Zenith_model))

if len(idx > 0):
    plt.plot(t[0:idx[1]], Elev_model[0:idx[1]], color='skyblue', lw=4)
    plt.plot(t[idx[0]-1:idx[-1]+2], Elev_model[idx[0]-1:idx[-1]+2], color='orange', lw=4)
    plt.plot(t[idx[-1]+1:], Elev_model[idx[-1]+1:], color='skyblue', lw=4)
    plt.fill_between(t, Elev_model, where=Elev_model>0, color='C1', alpha=0.4, label=f'Day, Sunrise: {Time_obs[idx[0]]}')
    plt.fill_between(t, Elev_model, where=Elev_model<0, color='C0', alpha=0.4, label=f'Night, Sunset: {Time_obs[idx[-1]]}')
    plt.plot(SE_obs, color='k', label=f'Observed\nSunrise: {Time_obs[idx1[0]]}\nSunset: {Time_obs[idx1[-1]]}')
    plt.legend(loc='best', facecolor='white')
    plt.xlabel('Local time (hrs)', fontweight = 'bold', fontsize = 18)
    plt.xticks(np.arange(0, 104, 8),('00:00', '02:00', '04:00', '06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00', '22:00', '24:00'))

    plt.ylabel('Elevation (Degree)', fontweight = 'bold', fontsize = 18)
    plt.title(f'{City_Name}   {day}-{month}-{year}', fontweight = 'bold', fontsize = 20)
    plt.grid()
        
else:
    plt.plot(Elev_model, color='skyblue', lw=4)
    plt.fill_between(t, Elev_model, where=Elev_model>0, color='C1', alpha=0.4, label='Day')        
    plt.fill_between(t, Elev_model, where=Elev_model<0, color='C0', alpha=0.4, label='Night')
    plt.legend(loc='best', facecolor='white')
    plt.xlabel('Local time (hrs)', fontweight = 'bold', fontsize = 18)
    plt.ylabel('Elevation (Degree)', fontweight = 'bold', fontsize = 18)
    plt.title(f'{City_Name}   {day}-{month}-{year}', fontweight = 'bold', fontsize = 20)
    plt.grid()


plt.subplot(4,4,14)

est1 = sm.OLS(Elev_model, SE_obs)
est1 = est1.fit()
coef = np.sqrt(est1.rsquared)

plt.plot(SE_obs, Elev_model, label=f'Corr_coef = {coef:.5f}')
plt.grid()
plt.legend(loc='best')
plt.xlabel('Observed Elevation (Degree)', fontweight = 'bold', fontsize = 15)
plt.ylabel('Model Elevation (Degree)', fontweight = 'bold', fontsize = 15)
plt.title('Model vs Observed (Daily Solar Elevation)', fontweight = 'bold', fontsize = 20)
    

plt.subplot(4,4,15)

est2 = sm.OLS(Azim_model, SA_obs)
est2 = est2.fit()
coef2 = np.sqrt(est2.rsquared)

plt.plot(SA_obs[1:], Azim_model[1:], label=f'Corr_coef = {coef2:.5f}')
plt.grid()
plt.legend(loc='best')
plt.xlabel('Observed Azimuthn (Degree)', fontweight = 'bold', fontsize = 15)
plt.ylabel('Model Azimuth (Degree)', fontweight = 'bold', fontsize = 15)
plt.title('Model vs Observed (Daily Solar Azimuth)', fontweight='bold', fontsize=20)

    
plt.subplot(4,4,16)

X = GHI_obs
X = sm.add_constant(X)
y = GHI_model

est=sm.OLS(y,X)
est=est.fit()
est.summary()
est.params
conf = est.conf_int(0.0001)
coef1 = np.sqrt(est.rsquared)
const, slope = est.params

y_hat = est.predict(X)
y1 = X * conf[1,0] + conf[0,0]
y2 = X * conf[1,1] + conf[0,1]

plt.scatter(GHI_obs, GHI_model)
plt.plot(X[:,1],y_hat,'k',label=f'OLS fit\nCorr_coef = {coef1:.5f}')
plt.fill_between(X[:,1], y1[:,1], y2[:,1], alpha=0.4)
plt.legend(loc='best')
plt.grid()
plt.ylabel('Model GHI (W/m2)', fontweight = 'bold', fontsize = 15)
plt.xlabel('Observed GHI (W/m2)', fontweight = 'bold', fontsize = 15)
plt.title('Model vs Observed (GHI)', fontweight = 'bold', fontsize = 20)    
    
    
    

