# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 00:17:00 2020

@author: ABHIRUP
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import solar_project_enc_wall_V2 as spewV2



City_Name = 'New Delhi'

# day = 9
# month = 7
year = 2020
tz = 5.5

lat = 28.6138889            
lon = 77.2088889  


days = np.ones(24)
idx = np.arange(1,25,2)
days[idx] = 15
days = days.astype(int)

m = np.ones(2).astype(int)
months = np.concatenate((m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m, 8*m, 9*m, 10*m, 11*m, 12*m))



area_target = 1200
r_target = np.sqrt(area_target / np.pi)
h_target = 30

r_wall = np.arange(10, 110, 10)
h_wall = np.arange(h_target+10, h_target+110, 10) 

rel_h_wall = h_wall - h_target
rel_h_target = 0


GHI_loss = np.zeros((len(r_wall), len(h_wall)))

for j in range(len(r_wall)):
    
    for k in range(len(h_wall)):

        Annual_GHI_loss_percentage = []
        
        for i in range(24):
            
            s = spewV2.solar_coords_and_GHI_loss(City_Name, lat, lon, year, months[i], days[i], tz)
            
            angle_df = s.Solar_Position_Specific_Location_Daily()
            
            GHI_daily_lost_percentage = s.calculating_GHI_loss(angle_df, r_target, rel_h_target, r_wall[j], rel_h_wall[k])
            
            Annual_GHI_loss_percentage.append(GHI_daily_lost_percentage)
        
        Annual_GHI_loss_percentage = np.array(Annual_GHI_loss_percentage)
        Annual_avg_GHI_loss_percentage = np.mean(Annual_GHI_loss_percentage)
        
        GHI_loss[j,k] = Annual_avg_GHI_loss_percentage


plt.figure(figsize=(12,10))
levels = np.arange(0, 105, 5)
contour = plt.contour(r_wall, rel_h_wall, GHI_loss.T, levels, colors='k')
plt.clabel(contour, colors = 'k', fmt='%d')
contour_filled = plt.contourf(r_wall, rel_h_wall, GHI_loss.T, levels, cmap='rainbow')
cbar=plt.colorbar(contour_filled)
labels = np.arange(0, 110, 10)
cbartick = cbar.set_ticks(labels)
cbarticklabels = cbar.set_ticklabels(labels)
cbar.set_label('% Loss in GHI', fontweight='bold', fontsize=16)
plt.xlabel('Obstacle distance (in m) from the target solar panel', fontsize=16, fontweight='bold')
plt.ylabel('Obstacle height (in m) relative to the target', fontsize=16, fontweight='bold')
plt.title(f'Annual average % GHI loss\nLocation - {City_Name}    Target Height : {rel_h_target} m   Target Area : {area_target} sq. m', fontsize=16, fontweight='bold')
plt.tight_layout()


