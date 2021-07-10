# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 21:08:55 2020

@author: ABHIRUP
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr
import statsmodels.api as sm
import xarray as xr


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


###############################################################################
    
def daily_model_GHI_elev(lat, lon, year, month, day, tz):
    
    angle_df = Solar_Position_Specific_Location_Daily(lat, lon, year, month, day, tz)

    Elevation = angle_df.Elevation
    Azimuth = angle_df.Azimuth
    Time = angle_df.Time
    
    idx1 = np.arange(59, 1499, 60)

    Elev_model = np.zeros(24)
    Zenith_model = np.zeros(24)

    for i in range(24):
        Elev_model[i] = Elevation[idx1[i]]
        Zenith_model[i] = 90 - Elev_model[i]
        if (Zenith_model[i] > 90) or (Zenith_model[i] < 0):
            Zenith_model[i] = 90
        if Elev_model[i] <= 0:
            Elev_model[i] = 0
            
    S0 = 1367

    GHI_model = S0 * np.cos(np.radians(Zenith_model))
    
    return GHI_model, Elev_model

###############################################################################
 
from netCDF4 import Dataset

observed_df_1 = Dataset('C:/Work/Solar Project/observed_data/CERES_SYN1deg-1H_Terra-Aqua-MODIS_Ed4.1_Subset_20190301-20190730.nc')

toa_solar_all_1h_1 = np.array(observed_df_1.variables['toa_solar_all_1h'])

observed_df_1 = xr.open_dataset('C:/Work/Solar Project/observed_data/CERES_SYN1deg-1H_Terra-Aqua-MODIS_Ed4.1_Subset_20190301-20190730.nc')
observed_df_2 = xr.open_dataset('C:/Work/Solar Project/observed_data/CERES_SYN1deg-1H_Terra-Aqua-MODIS_Ed4.1_Subset_20190731-20191229.nc')
observed_df_3 = xr.open_dataset('C:/Work/Solar Project/observed_data/CERES_SYN1deg-1H_Terra-Aqua-MODIS_Ed4.1_Subset_20191230-20200331.nc')
    
toa_solar_all_1h_1 = np.array(observed_df_1.toa_solar_all_1h)
toa_solar_all_1h_2 = np.array(observed_df_2.toa_solar_all_1h)
toa_solar_all_1h_3 = np.array(observed_df_3.toa_solar_all_1h)


observed_df_angle_1 = xr.open_dataset('C:/Users/ABHIRUP/Downloads/CERES_SYN1deg-1H_Terra-Aqua-MODIS_Ed4.1_Subset_20190301-20190730.nc')
observed_df_angle_2 = xr.open_dataset('C:/Users/ABHIRUP/Downloads/CERES_SYN1deg-1H_Terra-Aqua-MODIS_Ed4.1_Subset_20190731-20191229.nc')
observed_df_angle_3 = xr.open_dataset('C:/Users/ABHIRUP/Downloads/CERES_SYN1deg-1H_Terra-Aqua-MODIS_Ed4.1_Subset_20191230-20200331.nc')

solar_zenith_observed_1 = np.array(observed_df_angle_1.solar_zen_angle_1h)
solar_zenith_observed_2 = np.array(observed_df_angle_2.solar_zen_angle_1h)
solar_zenith_observed_3 = np.array(observed_df_angle_3.solar_zen_angle_1h)


###############################################################################

def daily_observed_GHI_elev(lat, lon, month, day):
    

    if lat % 1 > 0.5:
        lat_idx = 90 - (int(lat) + 1)
    else:
        lat_idx = 90 - int(lat)
        
    lat_idx = 180 - lat_idx

    if lon < 0:
        if np.abs(lon) % 1 > 0.5:
            lon_idx = 360 + int(lon) + 1
        else:
            lon_idx = 360 + int(lon)
    else:
        if lon % 1 > 0.5:
            lon_idx = int(lon) + 1
        else:
            lon_idx = int(lon)
            
            
    if (month >= 3) and (month <= 7):
    
        if month == 3:
            day_idx_init = (day - 1) * 24 - int(tz)
            day_idx = np.arange(day_idx_init, day_idx_init+24, 1)
            TOA_solar_inso_day = toa_solar_all_1h_1[day_idx, lat_idx, lon_idx]
            solar_zenith_day = solar_zenith_observed_1[day_idx, lat_idx, lon_idx]
            solar_elev_day = 90 - solar_zenith_day
    
        elif month == 4:
            day_idx_init = (31 + day - 1) * 24 - int(tz)
            day_idx = np.arange(day_idx_init, day_idx_init+24, 1)
            TOA_solar_inso_day = toa_solar_all_1h_1[day_idx, lat_idx, lon_idx]
            solar_zenith_day = solar_zenith_observed_1[day_idx, lat_idx, lon_idx]
            solar_elev_day = 90 - solar_zenith_day

        elif month == 5:
            day_idx_init = (31 + 30 + day - 1) * 24 - int(tz)
            day_idx = np.arange(day_idx_init, day_idx_init+24, 1)
            TOA_solar_inso_day = toa_solar_all_1h_1[day_idx, lat_idx, lon_idx]   
            solar_zenith_day = solar_zenith_observed_1[day_idx, lat_idx, lon_idx]
            solar_elev_day = 90 - solar_zenith_day
        
        elif month == 6:
            day_idx_init = (31 + 30 + 31 + day - 1) * 24 - int(tz)
            day_idx = np.arange(day_idx_init, day_idx_init+24, 1)
            TOA_solar_inso_day = toa_solar_all_1h_1[day_idx, lat_idx, lon_idx]
            solar_zenith_day = solar_zenith_observed_1[day_idx, lat_idx, lon_idx]
            solar_elev_day = 90 - solar_zenith_day

        else:
            day_idx_init = (31 + 30 + 31 + 30 + day - 1) * 24 - int(tz)
            day_idx = np.arange(day_idx_init, day_idx_init+24, 1)
            TOA_solar_inso_day = toa_solar_all_1h_1[day_idx, lat_idx, lon_idx]
            solar_zenith_day = solar_zenith_observed_1[day_idx, lat_idx, lon_idx]
            solar_elev_day = 90 - solar_zenith_day


    if (month >= 8) and (month <= 12):
        
        if month == 8:
            day_idx_init = day * 24 - int(tz)
            day_idx = np.arange(day_idx_init, day_idx_init+24, 1)
            TOA_solar_inso_day = toa_solar_all_1h_2[day_idx, lat_idx, lon_idx]
            solar_zenith_day = solar_zenith_observed_2[day_idx, lat_idx, lon_idx]
            solar_elev_day = 90 - solar_zenith_day
   
        elif month == 9:
            day_idx_init = (31 + day) * 24 - int(tz)
            day_idx = np.arange(day_idx_init, day_idx_init+24, 1)
            TOA_solar_inso_day = toa_solar_all_1h_2[day_idx, lat_idx, lon_idx]
            solar_zenith_day = solar_zenith_observed_2[day_idx, lat_idx, lon_idx]
            solar_elev_day = 90 - solar_zenith_day

        elif month == 10:
            day_idx_init = (31 + 30 + day) * 24 - int(tz)
            day_idx = np.arange(day_idx_init, day_idx_init+24, 1)
            TOA_solar_inso_day = toa_solar_all_1h_2[day_idx, lat_idx, lon_idx]   
            solar_zenith_day = solar_zenith_observed_2[day_idx, lat_idx, lon_idx]
            solar_elev_day = 90 - solar_zenith_day
        
        elif month == 11:
            day_idx_init = (31 + 30 + 31 + day) * 24 - int(tz)
            day_idx = np.arange(day_idx_init, day_idx_init+24, 1)
            TOA_solar_inso_day = toa_solar_all_1h_2[day_idx, lat_idx, lon_idx]
            solar_zenith_day = solar_zenith_observed_2[day_idx, lat_idx, lon_idx]
            solar_elev_day = 90 - solar_zenith_day

        else:
            day_idx_init = (31 + 30 + 31 + 30 + day - 1) * 24 - int(tz)
            day_idx = np.arange(day_idx_init, day_idx_init+24, 1)
            TOA_solar_inso_day = toa_solar_all_1h_2[day_idx, lat_idx, lon_idx]
            solar_zenith_day = solar_zenith_observed_2[day_idx, lat_idx, lon_idx]
            solar_elev_day = 90 - solar_zenith_day


    if (month >= 1) and (month <= 3):
    
        if month == 1:
            day_idx_init = (day + 1) * 24 - int(tz)
            day_idx = np.arange(day_idx_init, day_idx_init+24, 1)
            TOA_solar_inso_day = toa_solar_all_1h_3[day_idx, lat_idx, lon_idx]
            solar_zenith_day = solar_zenith_observed_3[day_idx, lat_idx, lon_idx]
            solar_elev_day = 90 - solar_zenith_day
    
        elif month == 2:
            day_idx_init = (31 + day + 1) * 24 - int(tz)
            day_idx = np.arange(day_idx_init, day_idx_init+24, 1)
            TOA_solar_inso_day = toa_solar_all_1h_3[day_idx, lat_idx, lon_idx]
            solar_zenith_day = solar_zenith_observed_3[day_idx, lat_idx, lon_idx]
            solar_elev_day = 90 - solar_zenith_day

        else:
            day_idx_init = (31 + 29 + day + 1) * 24 - int(tz)
            day_idx = np.arange(day_idx_init, day_idx_init+24, 1)
            TOA_solar_inso_day = toa_solar_all_1h_3[day_idx, lat_idx, lon_idx]
            solar_zenith_day = solar_zenith_observed_3[day_idx, lat_idx, lon_idx]
            solar_elev_day = 90 - solar_zenith_day
  
    return TOA_solar_inso_day, solar_elev_day

###############################################################################

fig1 = plt.figure(figsize = (28, 28))

###############################################################################

City_Name = 'New Delhi'

day = 23
month = 9
year = 2019 
tz = 5.5

lat = 28.6138889   
lon = 77.2088889

GHI_observed, Elevation_observed = daily_observed_GHI_elev(lat, lon, month, day)

GHI_model, Elevation_model = daily_model_GHI_elev(lat, lon, year, month, day, tz)

#GHI_observed[12] = GHI_model[12]

plt.subplot(4,4,1)

plt.plot(Elevation_observed, label='Observed Elev')
plt.plot(Elevation_model, label='model Elev')
plt.legend(loc='best')
plt.grid()
plt.xticks(np.arange(0, 23, 4), ('01:00', '05:00', '09:00', '13:00', '17:00', '21:00'))
#plt.xlabel('Local time (hrs)', fontweight = 'bold', fontsize = 15)
plt.ylabel('Elevation (Degree)', fontweight = 'bold', fontsize = 15)
plt.title(f'{City_Name}   {day}-{month}-{year}', fontweight = 'bold', fontsize = 20)

plt.subplot(4,4,2)

X1 = Elevation_observed
X1 = sm.add_constant(X1)
y1 = Elevation_model

est = sm.OLS(y1,X1)
est = est.fit()
est.summary()
est.params
coef1 = np.sqrt(est.rsquared)

y1_hat = est.predict(X1)

plt.scatter(Elevation_observed, Elevation_model)
plt.plot(X1[:,1],y1_hat,'k',label=f'OLS fit\nCorr_coef = {coef1:.5f}')
plt.legend(loc='best')
plt.grid()
#plt.xlabel('Observed Elevation (in deg)', fontweight = 'bold', fontsize = 15)
plt.ylabel('Model Elevation (in deg)', fontweight = 'bold', fontsize = 15)
plt.title(f'{City_Name}   {day}-{month}-{year}', fontweight = 'bold', fontsize = 20)



plt.subplot(4,4,3)

plt.plot(GHI_observed, label='Observed GHI')
plt.plot(GHI_model, label='model GHI')
plt.legend(loc='best')
plt.grid()
plt.xticks(np.arange(0, 23, 4), ('01:00', '05:00', '09:00', '13:00', '17:00', '21:00'))
#plt.xlabel('Local time (hrs)', fontweight = 'bold', fontsize = 15)
plt.ylabel('GHI (W/m2)', fontweight = 'bold', fontsize = 15)
plt.title(f'{City_Name}   {day}-{month}-{year}', fontweight = 'bold', fontsize = 20)


plt.subplot(4,4,4)

X2 = GHI_observed
X2 = sm.add_constant(X2)
y2 = GHI_model

est = sm.OLS(y2,X2)
est = est.fit()
est.summary()
est.params
coef2 = np.sqrt(est.rsquared)

y2_hat = est.predict(X2)

plt.scatter(GHI_observed, GHI_model)
plt.plot(X2[:,1],y2_hat,'k',label=f'OLS fit\nCorr_coef = {coef2:.5f}')
plt.legend(loc='best')
plt.grid()
#plt.xlabel('Observed GHI (W/m2)', fontweight = 'bold', fontsize = 15)
plt.ylabel('Model GHI (W/m2)', fontweight = 'bold', fontsize = 15)
plt.title(f'{City_Name}   {day}-{month}-{year}', fontweight = 'bold', fontsize = 20)

###############################################################################


City_Name = 'Kolkata'

day = 23
month = 9
year = 2019 
tz = 5.5

lat = 22.5726    
lon = 88.3639

GHI_observed, Elevation_observed = daily_observed_GHI_elev(lat, lon, month, day)

GHI_model, Elevation_model = daily_model_GHI_elev(lat, lon, year, month, day, tz)

#GHI_observed[12] = GHI_model[12]

plt.subplot(4,4,5)

plt.plot(Elevation_observed, label='Observed Elev')
plt.plot(Elevation_model, label='model Elev')
plt.legend(loc='best')
plt.grid()
plt.xticks(np.arange(0, 23, 4), ('01:00', '05:00', '09:00', '13:00', '17:00', '21:00'))
#plt.xlabel('Local time (hrs)', fontweight = 'bold', fontsize = 15)
plt.ylabel('Elevation (Degree)', fontweight = 'bold', fontsize = 15)
plt.title(f'{City_Name}   {day}-{month}-{year}', fontweight = 'bold', fontsize = 20)

plt.subplot(4,4,6)

X1 = Elevation_observed
X1 = sm.add_constant(X1)
y1 = Elevation_model

est = sm.OLS(y1,X1)
est = est.fit()
est.summary()
est.params
coef1 = np.sqrt(est.rsquared)

y1_hat = est.predict(X1)

plt.scatter(Elevation_observed, Elevation_model)
plt.plot(X1[:,1],y1_hat,'k',label=f'OLS fit\nCorr_coef = {coef1:.5f}')
plt.legend(loc='best')
plt.grid()
#plt.xlabel('Observed Elevation (in deg)', fontweight = 'bold', fontsize = 15)
plt.ylabel('Model Elevation (in deg)', fontweight = 'bold', fontsize = 15)
plt.title(f'{City_Name}   {day}-{month}-{year}', fontweight = 'bold', fontsize = 20)



plt.subplot(4,4,7)

plt.plot(GHI_observed, label='Observed GHI')
plt.plot(GHI_model, label='model GHI')
plt.legend(loc='best')
plt.grid()
plt.xticks(np.arange(0, 23, 4), ('01:00', '05:00', '09:00', '13:00', '17:00', '21:00'))
#plt.xlabel('Local time (hrs)', fontweight = 'bold', fontsize = 15)
plt.ylabel('GHI (W/m2)', fontweight = 'bold', fontsize = 15)
plt.title(f'{City_Name}   {day}-{month}-{year}', fontweight = 'bold', fontsize = 20)


plt.subplot(4,4,8)

X2 = GHI_observed
X2 = sm.add_constant(X2)
y2 = GHI_model

est = sm.OLS(y2,X2)
est = est.fit()
est.summary()
est.params
coef2 = np.sqrt(est.rsquared)

y2_hat = est.predict(X2)

plt.scatter(GHI_observed, GHI_model)
plt.plot(X2[:,1],y2_hat,'k',label=f'OLS fit\nCorr_coef = {coef2:.5f}')
plt.legend(loc='best')
plt.grid()
#plt.xlabel('Observed GHI (W/m2)', fontweight = 'bold', fontsize = 15)
plt.ylabel('Model GHI (W/m2)', fontweight = 'bold', fontsize = 15)
plt.title(f'{City_Name}   {day}-{month}-{year}', fontweight = 'bold', fontsize = 20)

###############################################################################


City_Name = 'Mumbai'

day = 23
month = 9
year = 2019 
tz = 5.5

lat = 18.975     
lon = 72.8258

GHI_observed, Elevation_observed = daily_observed_GHI_elev(lat, lon, month, day)

GHI_model, Elevation_model = daily_model_GHI_elev(lat, lon, year, month, day, tz)

#GHI_observed[12] = GHI_model[12]

plt.subplot(4,4,9)

plt.plot(Elevation_observed, label='Observed Elev')
plt.plot(Elevation_model, label='model Elev')
plt.legend(loc='best')
plt.grid()
plt.xticks(np.arange(0, 23, 4), ('01:00', '05:00', '09:00', '13:00', '17:00', '21:00'))
#plt.xlabel('Local time (hrs)', fontweight = 'bold', fontsize = 15)
plt.ylabel('Elevation (Degree)', fontweight = 'bold', fontsize = 15)
plt.title(f'{City_Name}   {day}-{month}-{year}', fontweight = 'bold', fontsize = 20)

plt.subplot(4,4,10)

X1 = Elevation_observed
X1 = sm.add_constant(X1)
y1 = Elevation_model

est = sm.OLS(y1,X1)
est = est.fit()
est.summary()
est.params
coef1 = np.sqrt(est.rsquared)

y1_hat = est.predict(X1)

plt.scatter(Elevation_observed, Elevation_model)
plt.plot(X1[:,1],y1_hat,'k',label=f'OLS fit\nCorr_coef = {coef1:.5f}')
plt.legend(loc='best')
plt.grid()
#plt.xlabel('Observed Elevation (in deg)', fontweight = 'bold', fontsize = 15)
plt.ylabel('Model Elevation (in deg)', fontweight = 'bold', fontsize = 15)
plt.title(f'{City_Name}   {day}-{month}-{year}', fontweight = 'bold', fontsize = 20)



plt.subplot(4,4,11)

plt.plot(GHI_observed, label='Observed GHI')
plt.plot(GHI_model, label='model GHI')
plt.legend(loc='best')
plt.grid()
plt.xticks(np.arange(0, 23, 4), ('01:00', '05:00', '09:00', '13:00', '17:00', '21:00'))
#plt.xlabel('Local time (hrs)', fontweight = 'bold', fontsize = 15)
plt.ylabel('GHI (W/m2)', fontweight = 'bold', fontsize = 15)
plt.title(f'{City_Name}   {day}-{month}-{year}', fontweight = 'bold', fontsize = 20)


plt.subplot(4,4,12)

X2 = GHI_observed
X2 = sm.add_constant(X2)
y2 = GHI_model

est = sm.OLS(y2,X2)
est = est.fit()
est.summary()
est.params
coef2 = np.sqrt(est.rsquared)

y2_hat = est.predict(X2)

plt.scatter(GHI_observed, GHI_model)
plt.plot(X2[:,1],y2_hat,'k',label=f'OLS fit\nCorr_coef = {coef2:.5f}')
plt.legend(loc='best')
plt.grid()
#plt.xlabel('Observed GHI (W/m2)', fontweight = 'bold', fontsize = 15)
plt.ylabel('Model GHI (W/m2)', fontweight = 'bold', fontsize = 15)
plt.title(f'{City_Name}   {day}-{month}-{year}', fontweight = 'bold', fontsize = 20)

###############################################################################


City_Name = 'Chennai'

day = 23
month = 9
year = 2019 
tz = 5.5

lat = 13.5    
lon = 80.16

GHI_observed, Elevation_observed = daily_observed_GHI_elev(lat, lon, month, day)

GHI_model, Elevation_model = daily_model_GHI_elev(lat, lon, year, month, day, tz)

#GHI_observed[12] = GHI_model[12]

plt.subplot(4,4,13)

plt.plot(Elevation_observed, label='Observed Elev')
plt.plot(Elevation_model, label='model Elev')
plt.legend(loc='best')
plt.grid()
plt.xticks(np.arange(0, 23, 4), ('01:00', '05:00', '09:00', '13:00', '17:00', '21:00'))
plt.xlabel('Local time (hrs)', fontweight = 'bold', fontsize = 15)
plt.ylabel('Elevation (Degree)', fontweight = 'bold', fontsize = 15)
plt.title(f'{City_Name}   {day}-{month}-{year}', fontweight = 'bold', fontsize = 20)

plt.subplot(4,4,14)

X1 = Elevation_observed
X1 = sm.add_constant(X1)
y1 = Elevation_model

est = sm.OLS(y1,X1)
est = est.fit()
est.summary()
est.params
coef1 = np.sqrt(est.rsquared)

y1_hat = est.predict(X1)

plt.scatter(Elevation_observed, Elevation_model)
plt.plot(X1[:,1],y1_hat,'k',label=f'OLS fit\nCorr_coef = {coef1:.5f}')
plt.legend(loc='best')
plt.grid()
plt.xlabel('Observed Elevation (in deg)', fontweight = 'bold', fontsize = 15)
plt.ylabel('Model Elevation (in deg)', fontweight = 'bold', fontsize = 15)
plt.title(f'{City_Name}   {day}-{month}-{year}', fontweight = 'bold', fontsize = 20)



plt.subplot(4,4,15)

plt.plot(GHI_observed, label='Observed GHI')
plt.plot(GHI_model, label='model GHI')
plt.legend(loc='best')
plt.grid()
plt.xticks(np.arange(0, 23, 4), ('01:00', '05:00', '09:00', '13:00', '17:00', '21:00'))
plt.xlabel('Local time (hrs)', fontweight = 'bold', fontsize = 15)
plt.ylabel('GHI (W/m2)', fontweight = 'bold', fontsize = 15)
plt.title(f'{City_Name}   {day}-{month}-{year}', fontweight = 'bold', fontsize = 20)


plt.subplot(4,4,16)

X2 = GHI_observed
X2 = sm.add_constant(X2)
y2 = GHI_model

est = sm.OLS(y2,X2)
est = est.fit()
est.summary()
est.params
coef2 = np.sqrt(est.rsquared)

y2_hat = est.predict(X2)

plt.scatter(GHI_observed, GHI_model)
plt.plot(X2[:,1],y2_hat,'k',label=f'OLS fit\nCorr_coef = {coef2:.5f}')
plt.legend(loc='best')
plt.grid()
plt.xlabel('Observed GHI (W/m2)', fontweight = 'bold', fontsize = 15)
plt.ylabel('Model GHI (W/m2)', fontweight = 'bold', fontsize = 15)
plt.title(f'{City_Name}   {day}-{month}-{year}', fontweight = 'bold', fontsize = 20)

