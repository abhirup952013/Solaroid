# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 13:53:42 2020

@author: ABHIRUP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import solar_project_V2 as spV2


City_Name = 'New Delhi'

# day = 18
# month = 10
year = 2020
tz = 5.5

lat = 28.6138889            
lon = 77.2088889  


### Obstacle features


df_obstacles = pd.DataFrame({
    'distance': [65, 120, 90, 75, 150],
    'obs_azim': [0, 90, 165, 225, 260], # Should be in increasing order
    'height': [45, 80, 75, 55, 90],
    'area': [1800, 2700, 2500, 2200, 3000]})


distance = df_obstacles.distance.to_numpy()
obs_azim = df_obstacles.obs_azim.to_numpy()
h_obs = df_obstacles.height.to_numpy()
r_obs = np.sqrt(df_obstacles.area.to_numpy() / np.pi)

### Target features

area_target = 1200
r_target = np.sqrt(area_target / np.pi)
h_target = 30


days = np.ones(24)
idx = np.arange(1,25,2)
days[idx] = 15
days = days.astype(int)

m = np.ones(2).astype(int)
months = np.concatenate((m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m, 8*m, 9*m, 10*m, 11*m, 12*m))

# s = spV2.solar_coords_and_shadow(City_Name, lat, lon, year, month, day, tz)

# angle_df = s.Solar_Position_Specific_Location_Daily()

# shadow_time_beg_end, shadow_time_total = s.calculate_shadow_time(angle_df, distance, obs_azim, h_obs, r_obs, r_target, h_target)
   
# GHI_daily_avg, GHI_daily_avg_loss, GHI_daily_net = s.calculate_GHI(angle_df, shadow_time_total)

# s.save_Info(shadow_time_beg_end, shadow_time_total, GHI_daily_avg, GHI_daily_avg_loss, GHI_daily_net, h_target, area_target, df_obstacles)

GHI, GHI_loss, GHI_net = [], [], []
    
for i in range(24):
    
    s = spV2.solar_coords_and_shadow(City_Name, lat, lon, year, months[i], days[i], tz)
    
    angle_df = s.Solar_Position_Specific_Location_Daily()
    
    shadow_time_beg_end, shadow_time_total = s.calculate_shadow_time(angle_df, distance, obs_azim, h_obs, r_obs, r_target, h_target)
    
    GHI_daily_avg, GHI_daily_avg_loss, GHI_daily_net = s.calculate_GHI(angle_df, shadow_time_total)

    GHI.append(GHI_daily_avg)
    GHI_loss.append(GHI_daily_avg_loss)
    GHI_net.append(GHI_daily_net)    

    
###############################################################################
# Plotting GHI

fig1 = plt.figure(figsize=(32,16))

barWidth = 0.25

br1 = np.arange(24)
br2 = [x + barWidth for x in br1] 
br3 = [x + barWidth for x in br2]

rects1 = plt.bar(br1, GHI_loss, width=barWidth, color='r', edgecolor='k', label='Lost GHI')
rects2 = plt.bar(br2, GHI, width=barWidth, color='c', edgecolor='k', label='GHI')
rects3 = plt.bar(br3, GHI_net, width=barWidth, color='g', edgecolor='k', label='Net GHI')

plt.legend(loc='best')

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        plt.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=90, fontweight='bold')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.xlabel('Day', fontsize=18, fontweight ='bold') 
plt.ylabel('GHI (W/m2)', fontsize=18, fontweight ='bold') 
plt.xticks([r + barWidth for r in range(24)], 
            ['Jan 01', 'Jan 15', 'Feb 01', 'Feb 15', 'Mar 01', 'Mar 15', 'Apr 01', 'Apr 15',
            'May 01', 'May 15', 'Jun 01', 'Jun 15', 'Jul 01', 'Jul 15', 'Aug 01', 'Aug 15',
            'Sep 01', 'Sep 15', 'Oct 01', 'Oct 15', 'Nov 01', 'Nov 15', 'Dec 01', 'Dec 15'], fontweight='bold') 

plt.title('New Delhi    Daily Total GHI throughout 2020', fontsize=18, fontweight='bold')

