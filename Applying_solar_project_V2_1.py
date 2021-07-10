# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 22:21:55 2020

@author: ABHIRUP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import solar_project_V2 as spV2


City_Name = 'New Delhi'

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


d31, m31 = list(np.arange(1, 32)), list(np.ones(31).astype(int))
d30, m30 = list(np.arange(1, 31)), list(np.ones(30).astype(int))
d28, m28 = list(np.arange(1, 29)), list(np.ones(28).astype(int))
d29, m29 = list(np.arange(1, 30)), list(np.ones(29).astype(int))

if year % 4 != 0:
    days = np.array(d31 + d28 + d31 + d30 + d31 + d30 + d31 + d31 + d30 + d31 + d30 + d31).astype(int)
    months = np.array(m31 + list(2*np.array(m28)) + list(3*np.array(m31)) + 
                      list(4*np.array(m30)) + list(5*np.array(m31)) + list(6*np.array(m30)) +
                      list(7*np.array(m31)) + list(8*np.array(m31)) + list(9*np.array(m30)) +
                      list(10*np.array(m31)) + list(11*np.array(m30)) + list(12*np.array(m31)))
else:
    days = np.array(d31 + d29 + d31 + d30 + d31 + d30 + d31 + d31 + d30 + d31 + d30 + d31).astype(int)  
    months = np.array(m31 + list(2*np.array(m29)) + list(3*np.array(m31)) + 
                      list(4*np.array(m30)) + list(5*np.array(m31)) + list(6*np.array(m30)) +
                      list(7*np.array(m31)) + list(8*np.array(m31)) + list(9*np.array(m30)) +
                      list(10*np.array(m31)) + list(11*np.array(m30)) + list(12*np.array(m31)))


    
for i in range(len(days)):
    
    s = spV2.solar_coords_and_shadow(City_Name, lat, lon, year, months[i], days[i], tz)
    
    angle_df = s.Solar_Position_Specific_Location_Daily()
        
    fig = s.plot3D(angle_df, distance, obs_azim, h_obs, r_obs, r_target, h_target)    
    fig.savefig(f'C:/Work/Solar Project/Animation_frames_04/frame_{i+1}.png')
