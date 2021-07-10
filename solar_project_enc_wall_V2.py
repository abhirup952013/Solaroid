# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 00:03:34 2020

@author: ABHIRUP
"""


class solar_coords_and_GHI_loss(object):
    
    def __init__(self, City_Name, lat, lon, year, month, day, tz):
        
        self.City_Name = City_Name
        self.lat = lat
        self.lon = lon
        self.year = year
        self.month = month
        self.day = day
        self.tz = tz

    
    def Solar_Position_Specific_Location_Daily(self):
        
        import numpy as np
        import pandas as pd
        
        lat = self.lat
        lon = self.lon
        year = self.year
        month = self.month
        day = self.day
        tz = self.tz
        
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
    
    


    def calculating_GHI_loss(self, angle_df, h_target, r_obstacle, h_obstacle):
        
        import numpy as np
        
        Elevation = np.array(angle_df.Elevation)
        Azimuth = np.array(angle_df.Azimuth)
        # Time = angle_df.Time
        
        IDX = np.argwhere(Elevation > 0)
        spots = IDX.shape[0]
        IDX = IDX.reshape(spots,) 
                
        day_elevation = np.array(Elevation[IDX])
        day_azimuth = np.array(Azimuth[IDX])

        thet = np.arange(0, 360, 1)

        shadow_length_vec = (h_obstacle - h_target) / np.tan(np.radians(day_elevation))
        
        shadow_idx = []
        
        for i in range(len(shadow_length_vec)):    
            if shadow_length_vec[i] >= r_obstacle:
                for obs_azim in thet:
                    if np.abs(day_azimuth[i] - obs_azim) < 1:
                        shadow_idx.append(i)
                        
        shadow_idx = list(set(shadow_idx))
                            
        S0 = 1368        
                
        day_zenith = 90 - day_elevation
        
        for i in range(len(day_zenith)):
            if (day_zenith[i] > 90) or (day_zenith[i] < 0):
                day_zenith[i] = 90
                
        GHI_daily = S0 * np.cos(np.radians(day_zenith))
        GHI_daily_total = np.sum(GHI_daily)
        
        if len(shadow_idx) > 0:
            GHI_daily_lost = GHI_daily[shadow_idx]
            GHI_daily_lost_total = np.sum(GHI_daily_lost)
        else:
            GHI_daily_lost_total = 0

        GHI_daily_lost_percentage = (GHI_daily_lost_total / GHI_daily_total) * 100
        
        return GHI_daily_lost_percentage

    
    
    
    
    
    


