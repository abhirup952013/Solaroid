# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 12:24:54 2020

@author: ABHIRUP
"""

class solar_coords_and_shadow(object):
    
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



    def plot3D(self, angle_df, d, obs_azimuth, h_obs, r_obs, r_target, h_target):
        
        import numpy as np
        import matplotlib.pyplot as plt
        
        ### Extracting variables from dataframe
        
        Elevation = np.array(angle_df.Elevation)
        Azimuth = np.array(angle_df.Azimuth)
        Time = np.array(angle_df.Time)
        
        IDX = np.argwhere(Elevation > 0)
        spots = IDX.shape[0]
        IDX = IDX.reshape(spots,) 
                
        day_elevation = np.array(Elevation[IDX])
        day_azimuth = np.array(Azimuth[IDX])
        day_time = list(Time[IDX])
        
        
        ### Creating Celestial dome
        
        r = 350
        
        theta = np.radians(day_elevation)
        phi = np.radians(day_azimuth)
                    
        x = r * np.cos(theta) * np.cos(2*np.pi - phi)
        y = r * np.cos(theta) * np.sin(2*np.pi - phi) 
        z = r * np.sin(theta)
        
        theta_grid, phi_grid = np.meshgrid(theta, phi)
                    
        x_grid = r * np.cos(theta_grid) * np.cos(2*np.pi - phi_grid)
        y_grid = r * np.cos(theta_grid) * np.sin(2*np.pi - phi_grid) 
        z_grid = r * np.sin(theta_grid)    
        
        
        ### Creating the target structure
                
        theta_target = np.linspace(0, 2*np.pi, 100)
        z_target = np.linspace(0, h_target, 100)
        theta_target_grid, z_target_grid = np.meshgrid(theta_target, z_target)
        x_target_grid = r_target * np.cos(theta_target_grid)
        y_target_grid = r_target * np.sin(theta_target_grid)    
        
        
        ### Creating obstacle structures
        
        num_obs = len(d)
        
        x_obs = np.zeros(num_obs)
        y_obs = np.zeros(num_obs)
        z_obs = np.zeros((num_obs, 100))
        theta_obs = np.zeros((num_obs, 100))
        z_obs_grid = np.zeros((num_obs, 100, 100))
        theta_obs_grid = np.zeros((num_obs, 100, 100))
        x_obs_grid = np.zeros((num_obs, 100, 100))
        y_obs_grid = np.zeros((num_obs, 100, 100))
        
        for i in range(num_obs):
            
            x_obs[i] = d[i] * np.cos(np.radians(360 - obs_azimuth[i]))
            y_obs[i] = d[i] * np.sin(np.radians(360 - obs_azimuth[i]))
        
            z_obs[i] = np.linspace(0, h_obs[i], 100)
            theta_obs[i] = np.linspace(0, 2*np.pi, 100)
            theta_obs_grid[i], z_obs_grid[i] = np.meshgrid(theta_obs[i], z_obs[i])
        
            x_obs_grid[i] = r_obs[i] * np.cos(theta_obs_grid[i]) + x_obs[i]
            y_obs_grid[i] = r_obs[i] * np.sin(theta_obs_grid[i]) + y_obs[i]    
        
            
        ### Creating Shadow vector
         
        shadow_length_vec = np.zeros((num_obs, len(day_azimuth)))    
        x_shadow = np.zeros((num_obs, len(day_time)))
        y_shadow = np.zeros((num_obs, len(day_time)))
        
        xx_shadow = np.zeros((num_obs, len(day_time), 100))
        yy_shadow = np.zeros((num_obs, len(day_time), 100))
        
        tolerance_angle = np.zeros(num_obs)
        
        shadow_idx = []
        
        for j in range(num_obs):
            
            tolerance_angle[j] = np.degrees(2*np.arctan((r_target + r_obs[j]) / d[j]))
        
            if h_obs[j] > h_target:
                shadow_length_vec[j] = (h_obs[j] - h_target) / np.tan(np.radians(day_elevation))
        
            shadow_idx_obs = []
                
            for i in range(len(day_time)):
                
                if shadow_length_vec[j][i] >= d[j]:
                    if np.abs(day_azimuth[i] - obs_azimuth[j]) < tolerance_angle[j] / 2 :
                        shadow_idx_obs.append(i)
        
                if shadow_length_vec[j][i] > d.max():
                    shadow_length_vec[j][i] = d.max()
            
                if (day_azimuth[i] > 0) & (day_azimuth[i] <= 90):
                    a = day_azimuth[i]
                    x_shadow[j][i] = - shadow_length_vec[j][i] * np.cos(np.radians(a)) + x_obs[j]
                    y_shadow[j][i] = shadow_length_vec[j][i] * np.sin(np.radians(a)) + y_obs[j]
            
                elif (day_azimuth[i] > 90) & (day_azimuth[i] <= 180):
                    a = 180 - day_azimuth[i]
                    x_shadow[j][i] = shadow_length_vec[j][i] * np.cos(np.radians(a)) + x_obs[j]
                    y_shadow[j][i] = shadow_length_vec[j][i] * np.sin(np.radians(a)) + y_obs[j]
            
                elif (day_azimuth[i] > 180) & (day_azimuth[i] <= 270):
                    a = day_azimuth[i] - 180
                    x_shadow[j][i] = shadow_length_vec[j][i] * np.cos(np.radians(a)) + x_obs[j]
                    y_shadow[j][i] = - shadow_length_vec[j][i] * np.sin(np.radians(a)) + y_obs[j]
            
                else:
                    a = 360 - day_azimuth[i]
                    x_shadow[j][i] = - shadow_length_vec[j][i] * np.cos(np.radians(a)) + x_obs[j]
                    y_shadow[j][i] = - shadow_length_vec[j][i] * np.sin(np.radians(a)) + y_obs[j]
        
                
                xx_shadow[j][i] = np.linspace(x_obs[j], x_shadow[j][i], 100)
                yy_shadow[j][i] = np.linspace(y_obs[j], y_shadow[j][i], 100)
                
            shadow_idx.append(shadow_idx_obs)
            
            
        ### 3D Plotting
        
        from mpl_toolkits.mplot3d import Axes3D
            
        fig = plt.figure(figsize=(16,14))
        ax = fig.add_subplot(111, projection='3d')            
        
        ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.2, cmap='summer')
                    
        ax.scatter(x, y, z, s=10, marker = 'o', c='y', alpha=0.1)
        ax.scatter(x, y, s=10, marker = 'o', c='y', alpha=0.1)
        
        ax.plot_surface(x_target_grid, y_target_grid, z_target_grid, color='y', alpha=0.9)
        
        for i in range(num_obs):
            ax.plot_surface(x_obs_grid[i], y_obs_grid[i], z_obs_grid[i], color='r', alpha=0.9)
        
        
        for j in range(num_obs):    
            for i in range(len(day_time)):        
                if i in shadow_idx[j]:
                    ax.plot(xx_shadow[j][i], yy_shadow[j][i], 0*xx_shadow[j][i], lw=12, color='k', alpha=0.1)            
                else:           
                    ax.plot(xx_shadow[j][i], yy_shadow[j][i], 0*xx_shadow[j][i], lw=12, color='k', alpha=0.01)
        
           
        ### Setting axes labels in the 3D plot

        ax.set_xlabel('''\n\nCelestial dome length (in m)\n\n<-------South--------||--------North------->''', fontweight='bold', fontsize=12)        
       # ax.set_xlabel('''<-------South--------||--------North------->''', fontweight='bold', fontsize=12)
       # ax.set_ylabel('''<------Forenoon-----|Noon|-----Afternoon------>''', fontweight='bold', fontsize=12)
        ax.set_ylabel('''\n\nCelestial dome width (in m)\n\n<------Forenoon-----|Noon|-----Afternoon------>''', fontweight='bold', fontsize=12)
        ax.set_zlabel('Above Ground Level Height (in m)', fontweight='bold', fontsize=12)
        ax.set_xlim(-r, r)
        ax.set_ylim(-r, r)
        #ax.set_zlim(0, 4*h_obs.max())
        ax.set_zlim(0, 400)        
        ax.set_title(f'''{self.City_Name} ({self.lat:.5f} N, {self.lon:.5f} E)              Sun Path Diagram                     {self.day}-{self.month}-{self.year} 
-------------------------------------------------------------------------------------------------------------------------''', fontsize=18, fontweight='bold', loc='left')

        return fig


    def calculate_shadow_time(self, angle_df, d, obs_azimuth, h_obs, r_obs, r_target, h_target):
        
        import numpy as np
        
        ### Extracting variables from dataframe
        
        Elevation = np.array(angle_df.Elevation)
        Azimuth = np.array(angle_df.Azimuth)
        Time = np.array(angle_df.Time)
        
        IDX = np.argwhere(Elevation > 0)
        spots = IDX.shape[0]
        IDX = IDX.reshape(spots,) 
                
        day_elevation = np.array(Elevation[IDX])
        day_azimuth = np.array(Azimuth[IDX])
        day_time = list(Time[IDX])  
        
            
        ### Creating obstacle structures
        
        num_obs = len(d)
        
        x_obs = np.zeros(num_obs)
        y_obs = np.zeros(num_obs)
        z_obs = np.zeros((num_obs, 100))
        theta_obs = np.zeros((num_obs, 100))
        z_obs_grid = np.zeros((num_obs, 100, 100))
        theta_obs_grid = np.zeros((num_obs, 100, 100))
        x_obs_grid = np.zeros((num_obs, 100, 100))
        y_obs_grid = np.zeros((num_obs, 100, 100))
        
        for i in range(num_obs):
            
            x_obs[i] = d[i] * np.cos(np.radians(360 - obs_azimuth[i]))
            y_obs[i] = d[i] * np.sin(np.radians(360 - obs_azimuth[i]))
        
            z_obs[i] = np.linspace(0, h_obs[i], 100)
            theta_obs[i] = np.linspace(0, 2*np.pi, 100)
            theta_obs_grid[i], z_obs_grid[i] = np.meshgrid(theta_obs[i], z_obs[i])
        
            x_obs_grid[i] = r_obs[i] * np.cos(theta_obs_grid[i]) + x_obs[i]
            y_obs_grid[i] = r_obs[i] * np.sin(theta_obs_grid[i]) + y_obs[i]    
        
            
        ### Creating Shadow vector
         
        shadow_length_vec = np.zeros((num_obs, len(day_azimuth)))    
        x_shadow = np.zeros((num_obs, len(day_time)))
        y_shadow = np.zeros((num_obs, len(day_time)))
        
        xx_shadow = np.zeros((num_obs, len(day_time), 100))
        yy_shadow = np.zeros((num_obs, len(day_time), 100))
        
        tolerance_angle = np.zeros(num_obs)
        
        shadow_idx = []
        
        for j in range(num_obs):
            
            tolerance_angle[j] = np.degrees(2*np.arctan((r_target + r_obs[j]) / d[j]))
        
            if h_obs[j] > h_target:
                shadow_length_vec[j] = (h_obs[j] - h_target) / np.tan(np.radians(day_elevation))
        
            shadow_idx_obs = []
                
            for i in range(len(day_time)):
                
                if shadow_length_vec[j][i] >= d[j]:
                    if np.abs(day_azimuth[i] - obs_azimuth[j]) < tolerance_angle[j] / 2 :
                        shadow_idx_obs.append(i)
        
                if shadow_length_vec[j][i] > d.max():
                    shadow_length_vec[j][i] = d.max()
            
                if (day_azimuth[i] > 0) & (day_azimuth[i] <= 90):
                    a = day_azimuth[i]
                    x_shadow[j][i] = - shadow_length_vec[j][i] * np.cos(np.radians(a)) + x_obs[j]
                    y_shadow[j][i] = shadow_length_vec[j][i] * np.sin(np.radians(a)) + y_obs[j]
            
                elif (day_azimuth[i] > 90) & (day_azimuth[i] <= 180):
                    a = 180 - day_azimuth[i]
                    x_shadow[j][i] = shadow_length_vec[j][i] * np.cos(np.radians(a)) + x_obs[j]
                    y_shadow[j][i] = shadow_length_vec[j][i] * np.sin(np.radians(a)) + y_obs[j]
            
                elif (day_azimuth[i] > 180) & (day_azimuth[i] <= 270):
                    a = day_azimuth[i] - 180
                    x_shadow[j][i] = shadow_length_vec[j][i] * np.cos(np.radians(a)) + x_obs[j]
                    y_shadow[j][i] = - shadow_length_vec[j][i] * np.sin(np.radians(a)) + y_obs[j]
            
                else:
                    a = 360 - day_azimuth[i]
                    x_shadow[j][i] = - shadow_length_vec[j][i] * np.cos(np.radians(a)) + x_obs[j]
                    y_shadow[j][i] = - shadow_length_vec[j][i] * np.sin(np.radians(a)) + y_obs[j]
        
                
                xx_shadow[j][i] = np.linspace(x_obs[j], x_shadow[j][i], 100)
                yy_shadow[j][i] = np.linspace(y_obs[j], y_shadow[j][i], 100)
                
            shadow_idx.append(shadow_idx_obs)
    
    
        ### Extracting shadow times corresponding to each obstacle 
            
        shadow_time = []    
        for j in range(num_obs):
            shadow_time_obs = []
            for i in shadow_idx[j]:
                shadow_time_obs.append(day_time[i])
            shadow_time.append(shadow_time_obs)
        
        shadow_time_beg_end = []
        for j in range(num_obs):
            shadow_time_beg_end_obs = []
            if len(shadow_time[j]) != 0:
                shadow_time_beg_end_obs.append(shadow_time[j][0])
                shadow_time_beg_end_obs.append(shadow_time[j][-1])
                shadow_time_beg_end_obs.append(len(shadow_time[j]))        
            else:
                shadow_time_beg_end_obs.append('N/A')
                shadow_time_beg_end_obs.append('N/A')
                shadow_time_beg_end_obs.append(len(shadow_time[j])) 
            shadow_time_beg_end.append(shadow_time_beg_end_obs)
        
        
        ### Extracting shadow time corresponding to the target
               
        shadow_time_total = []
        for j in range(num_obs):
            stt = []
            if shadow_time_beg_end[j][2] != 0:
                stt.append(shadow_time_beg_end[j][0])
                end = shadow_time_beg_end[j][1]
                break
        
        if j+1 < num_obs:        
            for i in range(j+1, num_obs):                
                if shadow_time_beg_end[i][2] != 0: 
                    if int(shadow_time_beg_end[i][0].split(':')[0]) > int(end.split(':')[0]):
                        stt.append(end)
                        shadow_time_total.append(stt)
                        stt = []
                        stt.append(shadow_time_beg_end[i][0])
                        end = shadow_time_beg_end[i][1]
                    else:
                        if int(shadow_time_beg_end[i][1].split(':')[0]) > int(end.split(':')[0]):
                            end = shadow_time_beg_end[i][1] 
                        elif int(shadow_time_beg_end[i][1].split(':')[1]) > int(end.split(':')[1]):
                            end = shadow_time_beg_end[i][1]
            stt.append(end)
            shadow_time_total.append(stt)
        else:
            if len(stt) != 0:
                stt.append(end)
                shadow_time_total.append(stt)
    
    
        return shadow_time_beg_end, shadow_time_total




    def calculate_GHI(self, angle_df, shadow_time_total):
        
        import numpy as np
    
        ### Extracting variables from dataframe
        
        Elevation = np.array(angle_df.Elevation)
        Azimuth = np.array(angle_df.Azimuth)
        Time = np.array(angle_df.Time)
        
        IDX = np.argwhere(Elevation > 0)
        spots = IDX.shape[0]
        IDX = IDX.reshape(spots,) 
                
        day_elevation = np.array(Elevation[IDX])
        day_azimuth = np.array(Azimuth[IDX])
        day_time = list(Time[IDX])      
    
    
        ### Calculating GHI
        
        S0 = 1368        
                
        day_zenith = 90 - day_elevation
        
        for i in range(len(day_zenith)):
            if (day_zenith[i] > 90) or (day_zenith[i] < 0):
                day_zenith[i] = 90
                
        GHI_daily = S0 * np.cos(np.radians(day_zenith))
        GHI_daily_total = np.sum(GHI_daily)
        GHI_daily_avg = np.mean(GHI_daily)
        
        shadow_idx_total = []
        for i in range(len(shadow_time_total)):
            sit = []
            sit.append(day_time.index(shadow_time_total[i][0]))
            sit.append(day_time.index(shadow_time_total[i][1]))
            shadow_idx_total.append(sit)
            
        shadow_day_zenith = []
        for j in range(len(shadow_time_total)):
            for i in range(shadow_idx_total[j][0], shadow_idx_total[j][1]+1):
                shadow_day_zenith.append(day_zenith[i])
                
        shadow_day_zenith = np.array(shadow_day_zenith)       
        
        GHI_daily_total_loss = np.sum(S0 * np.cos(np.radians(shadow_day_zenith)))
        GHI_daily_avg_loss = GHI_daily_total_loss / len(day_time)
        GHI_daily_total_gain = GHI_daily_total - GHI_daily_total_loss
        GHI_daily_net = GHI_daily_total_gain / len(day_time)
    
        return GHI_daily_avg, GHI_daily_avg_loss, GHI_daily_net



    def save_Info(self, shadow_time_beg_end, shadow_time_total, GHI_daily_avg, GHI_daily_avg_loss, GHI_daily_net, h_target, area_target, df_obstacles):
        

        distance = df_obstacles.distance.to_numpy()
        obs_azim = df_obstacles.obs_azim.to_numpy()
        h_obs = df_obstacles.height.to_numpy()
        area_obs = df_obstacles.area.to_numpy()

        num_obs = len(distance)
    
        filename = f'C:/Work/Solar Project/{self.City_Name}_{self.day}-{self.month}-{self.year}_Shadow.txt'
        
        with open (filename, 'w') as file:
            file.seek(0,0)
            file.write(f'Location  : {self.City_Name}\nLatitude  : {self.lat} N\nLongitude : {self.lon} E\nDate      : {self.day}-{self.month}-{self.year}\n\n')
            file.write(f'Target height : {h_target} m\n')
            file.write(f'Target area   : {area_target} sq. m\n\n')
            file.write('---------------------------------------------------------------------------------------------------------------------------\n')
            file.write('\nObstacle id   Obstacle Height     Obstacle Area       Distance from Target       Obstacle Azimuth      Shadow Casting Time\n')
            file.write('                  (in m)            (in sq. m)                  (in m)          (in decimal degrees)     (local hrs) \n\n')
            file.write('                                                                                                          Start     Stop\n')    
            file.write('---------------------------------------------------------------------------------------------------------------------------\n\n')
            for i in range(num_obs):
                file.write(f'     {i+1}             {h_obs[i]}\t\t\t{area_obs[i]}\t\t\t{distance[i]}\t\t\t{obs_azim[i]}\t\t  {shadow_time_beg_end[i][0]}\t    {shadow_time_beg_end[i][1]}\n')        
            file.write('---------------------------------------------------------------------------------------------------------------------------\n\n')
            if len(shadow_time_total) == 0:
                file.write('The target will receive sunlight throughout the day in this scenario.\n')
            else:
                file.write('The target will NOT receive sunlight in the following time window(s) (in local time) :\n\n')
                for i in range(len(shadow_time_total)):
                    file.write(f'from {shadow_time_total[i][0]} hrs to {shadow_time_total[i][1]} hrs\n')
            file.write(f'\n\nGHI calculation on {self.day}-{self.month}-{self.year}: \n\n')
            file.write('Daily Avg. GHI\t   Daily Avg. Loss\tDaily Net Gain\n')
            file.write('--------------------------------------------------------\n')
            file.write(f'{GHI_daily_avg:.4f} W/m2       {GHI_daily_avg_loss:.4f} W/m2         {GHI_daily_net:.4f} W/m2\n')
    

        
        










