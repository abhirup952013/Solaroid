# Solaroid

Product Name : Solaroid
Owner        : Abhirup Bhattacharya (MTech, Centre for Atmospheric Sciences, IIT Delhi)

*********************************************************************************************************************************************************************************

This following product gives an accurate understanding of shading effects of any number of obstacles around an urban roof-top solar panel installation, placed horizontally atop any building at any place on Earth. 

It also gives the daily average available/lost/gained Global Horizontal Irradiance (GHI) throughout a year at that urban roof-top solar installation considering a clear-sky environment as well as the Annual average % loss in GHI for different heights of neighbouring obstacle buildings at different distance, that will give an idea about safe distance/dimension of obstacles around an urban roof-top solar installation.

And finally, last but not the least this product generates frames of daily shadow profiles of any given number of obstacle building(s) having any given height(s) and width(s) at any given distance(s) from the building atop which solar panel is located, throughout a year, those can be put one consecutively in a different software (e.g. Blender) to create a nice animation of shadow profiles of any real life neighbourhood on an urban roof top solar panel.

***Attention: All the codes/modules avaiable here are written and tested on Python 3.8***

1. solar_project_V2.py

	Calls upon  : None
	Called by   : Applying_solar_project_V2.py; Applying_solar_project_V2_1.py 

	Input	: None
	Output	: None

	Description:

		This python module contains a class, named 'solar_coords_and_shadow (object)', that has the following instances:

		(a) City_Name : Name of the city / location on Earth (String)
		(b) lat	      : City latitude (Float)
		(c) lon       : City longitude (Float)
		(d) year      : Year of interest (Integer)
		(e) month     : Month of interest (Integer)
		(f) day       : Day of interest (Integer)
		(g) tz	      : Timezone in which the city/location belongs (Float, e.g 5.5(-5.5) for UTC+5:30(UTC-5:30) timezone)

		This module has the following Class Methods:

		(a) Solar_Position_Specific_Location_Daily(*args)

			Input	: self - instances of the class 'solar_coords_and_shadow'.
			Output  : angle_df - a Pandas DataFrame containing Solar Angles (Elevation, Azimuth, Declination, Right Ascention) at every minute throughout the day.

		(b) plot3D(*args)

			Input	: self - instances of the class 'solar_coords_and_shadow'.
				  angle_df - output from the class method 'Solar_Position_Specific_Location_Daily'.
			          d - a numpy array containing distances of each obstacle fromthe target building.
				  obs_azimuth - a numpy array containing azimuthal angles of obstacle buildings around the target building (N - 0; E - 90, S - 180; W - 270)
				  h_obs - a numpy array containing heights(m) of the obstacle buildings.
				  r_obs - a numpy array containing widths(m) of the obstacle buildings (Note: This model considers all the buildings to have cylindrical shapes, hence width 				          	  resemblences to diameter of that building).
				  r_target - Floating point number denoting the width(m) of the target building atop which the solar panel is located.
				  h_target - Floating point number denoting the height(m) of the target building atop which the solar panel is located.

			Output	: a matplotlib 3D Figure depicting the shadow profiles of the obstacle buildings on that particular day.

		(c) calculate_shadow_time(*args)

			Input	: Same as plot3D

			Output  : shadow_time_beg_end - a list containing lists corresponding to each obstacle building showing start/stop time and total length of time (minutes) (local 				  	  			time) for that obstacle building to cast shadow on the target building (Obstacle perspective).
				  shadow_time_total - a list containing list(s) of start/stop time of the shadow received by the target building. It handles the case if there is any 						                      overlapping of shadow timings of two or more different obstacle buildings (Target perspective).

		(d) calculate_GHI(*args)

			Input	: self - instances of the class 'solar_coords_and_shadow'.
				  angle_df - output from the class method 'Solar_Position_Specific_Location_Daily'.
				  shadow_time_total - output from the class method 'calculate_shadow_time'.

			Output  : GHI_daily_avg - Floating point number denoting daily average avaiable GHI at the roof-top solar panel assuming a clear-sky environment.
				  GHI_daily_avg_loss - Floating point number denoting daily average avaiable GHI loss at the roof-top solar panel assuming a clear-sky environmentdue to 						       	       shading effects of the above mentioned obstacle scenario.
				  GHI_daily_net - Floating point number denoting daily net gained GHI at the urban roof-top solar panel given the above obstacle scenario.

		(e) save_Info(*args)

			Input	: self - instances of the class 'solar_coords_and_shadow'.
				  All outputs from the class method 'calculate_shadow_time'.
				  All outputs from the class method 'calculate_GHI'.
				  h_target - Floating point number denoting the height(m) of the target building atop which the solar panel is located.
				  area_target - Floating point number denoting the area (sq. m) of the target building atop which the solar panel is located (used to calculate width of the target building).
				  df_obstacles - a user defined Pandas Dataframe, containing distance(s), azimuthal angle(s), height(s) and area(s) of the obstacles around the target building atop which the solar panel is located (All units are in meters).

			Output	: A '.txt' file with name 'City_Name_day-month-year_Shadow.txt', automatically saved at any user defined location, containing all the above informations in a very comprehensive way. (Note: at line number 622 of the module 'solar_project_V2.py', please insert a valid path for saving this '.txt' file).


2. Applying_solar_project_V2.py

	Calls upon  : solar_project_V2 as spV2
	Called by   : None

	Application - This code can be utilised in two settings to generate different outputs.

	Setup #1 (Default)

	Input	: City_Name - Name of the city / location on Earth (String)
		  year - Year of interest (Integer)
		  tz - Timezone in which the city/location belongs (Float, e.g 5.5(-5.5) for UTC+5:30(UTC-5:30) timezone)
		  lat - City latitude (Float)
		  lon - City longitude (Float)
		  df_obstacles - a user defined Pandas Dataframe, containing distance(s), azimuthal angle(s), height(s) and area(s) of the obstacles around the target building atop which 				               the solar panel is located.
		  area_target - Floating point number denoting the area(sq. m) of the target building atop which the solar panel is located (used to calculate width of the target building).
		  h_target - Floating point number denoting the height(m) of the target building atop which the solar panel is located.

	Output	: A matplotlib 2D figure showing a barplot of daily average avaiable/lost/gained GHI at the city location for every 1st and 15th day of a month throughout the given year.

	Setup #2 : Uncomment - Line #16, #17 and #55 to #63
		   Comment   - Line #47 to #53 and #65 to #120

	Input	: City_Name - Name of the city / location on Earth (String)
		  day - Day of interest (Integer)
		  month - Month of interest (Integer)
		  year - Year of interest (Integer)
		  tz - Timezone in which the city/location belongs (Float, e.g 5.5(-5.5) for UTC+5:30(UTC-5:30) timezone)
		  lat - City latitude (Float)
		  lon - City longitude (Float)
		  df_obstacles - a user defined Pandas Dataframe, containing distance(s), azimuthal angle(s), height(s) and area(s) of the obstacles around the target building atop which 				         the solar panel is located.
		  area_target - Floating point number denoting the area(sq. m) of the target building atop which the solar panel is located (used to calculate width of the target building).
		  h_target - Floating point number denoting the height(m) of the target building atop which the solar panel is located.

	Output	: A '.txt' file with name 'City_Name_day-month-year_Shadow.txt', automatically saved at any user defined location, containing daily informations regarding the shadow timing for a given obstacle scenario along with avaiable/lost/ganied GHI very comprehensive way.


3. Applying_solar_project_V2_1.py

	Calls upon  : solar_project_V2 as spV2
	Called by   : None
		
	Application - This code is used to generate animation frames of obstacle shadow profiles for a given obstacle scenario at any location for every day throughout a given year.

	Input	: City_Name - Name of the city / location on Earth (String)
		  year - Year of interest (Integer)
		  tz - Timezone in which the city/location belongs (Float, e.g 5.5(-5.5) for UTC+5:30(UTC-5:30) timezone)
		  lat - City latitude (Float)
		  lon - City longitude (Float)
		  df_obstacles - a user defined Pandas Dataframe, containing distance(s), azimuthal angle(s), height(s) and area(s) of the obstacles around the target building atop which 				         the solar panel is located.
		  area_target - Floating point number denoting the area(sq. m) of the target building atop which the solar panel is located (used to calculate width of the target building).
		  h_target - Floating point number denoting the height(m) of the target building atop which the solar panel is located.

	Output	: Animation frames (matplotlib 2D figures) with file names 'frame_day_index' corresponding to shadow profiles at every day. (Note: at line number 72 of this code please insert a valid path for saving these '.png' files). These frames can be further put chronologically in a 3rd party software (e.g Blender) to create nice animations having 365 (or 366 depending upon leap years) frames.


4. solar_project_enc_wall_V2.py

	Calls upon  : None
	Called by   : Applying_solar_project_enc_wall_V2.py

	Input	: None
	Output	: None

	Description:
		
		This python module uses the concept of circular barrier, i.e., an encircling wall of certain height at a certain distance from the target building, that casts shadow on the target building depending upon the position of the Sun in the sky. As there is no chance for an obstacle building of missing the target while casting shadow due to non-coinciding azimuthal angles of the obstacle and the Sun, hence, the wall will cast shadow only when the shadow length is more than the radius of the encircling wall.

		This module uses same class, named 'solar_coords_and_shadow (object)' as 'solar_project_V2.py' having exactly same instances. The only difference is that this class contains only two methods:

			(a) Solar_Position_Specific_Location_Daily(*args)

				Same as is used in the module 'solar_project_V2.py'.

			(b) calculate_GHI_loss(*args)

				Input	: self - instances of the class 'solar_coords_and_shadow'.
					  angle_df - output from the class method 'Solar_Position_Specific_Location_Daily'.
					  h_target - Floating point number denoting the height(m) of the target building atop which the solar panel is located.
					  r_obstacle - Floating point number denoting the radius(m) of the encircling barrier around the target building.
					  h_obstacle - Floating point number denoting the height(m) of the encircling barrier around the target building.

				Output	: GHI_daily_lost_percentage - Floating point number denoting the daily percentage of GHI loss due to the shading effects of the encircling wall.


5. Applying_solar_project_enc_wall_V2.py

	Calls upon   : solar_project_enc_wall_V2.py
	Called by    : None

	Application - This code is used to generate a contour plot of annual average % loss in GHI due to shading effects of encircling obstacles with both heights relative to target and radii of the walls ranging from 10m to 100m. This plot gives an idea about safe distance/dimension of obstacles around an urban roof-top solar installation even before 		  construction of that obstacle.

	Input	: City_Name - Name of the city / location on Earth (String)
		  day - Day of interest (Integer)
		  month - Month of interest (Integer)
		  year - Year of interest (Integer)
		  tz - Timezone in which the city/location belongs (Float, e.g 5.5(-5.5) for UTC+5:30(UTC-5:30) timezone)
		  lat - City latitude (Float)
		  lon - City longitude (Float)
		  df_obstacles - a user defined Pandas Dataframe, containing distance(s), azimuthal angle(s), height(s) and area(s) of the obstacles around the target building atop which 				         the solar panel is located.
		  area_target - Floating point number denoting the area(sq. m) of the target building atop which the solar panel is located (used to calculate width of the target building).
		  h_target - Floating point number denoting the height(m) of the target building atop which the solar panel is located.

	Output	: a matplotlib 2D contour plot of annual average % loss in GHI due to shading effects of encircling obstacles with both heights relative to target and radii of the walls ranging from 10m to 100m.


*********************************************************************************************************************************************************************************
******************************************************************************** Codes for Model Evaluation**********************************************************************
*********************************************************************************************************************************************************************************


1. solar_project_evaluation.py

	Call upon   : None
	Called by   : None

	Description :

		This python code evaluates the solar angles (Elevation and Azimuth) and GHI caclculated by the model against the ground based observed data from Almeria, Spain on both 		solstices and both equinoxes for the year 2017. Where the data is avaiable at 15 minutes interval from 1st Jan 1991 till 31st Dec 2018, while the model data is avaiable at 		every minute, so, the data corresponding to the time of observed data has been fetched from the model output for the four spcial days of the year 2017.

		Note: At line #201 of this code, please insert a valid loaction of the observed data.

	Input	: Observed data at Almeria, Spain (https://solargis.com/products/evaluate/useful-resources)
		  Variables Fetched - Date, T(Time), SE(Solar Elevation), SA(Solar Azimuth), GHI(Observed GHI).

		  Model input - City_Name = Almeria (Spain)    
			        lat = 37.094416                   
			        lon = -2.359850
			        tz = 0 
			        Year = 2017
                                Month = March/June/September/December
 			        Day = 20/21/22/21

	Output	: A matplotlib 2D plot having Solar Elevation (both observed and model output), Solar Elevation (Observed vs Model Output), Solar Azimuth (Observed vs Model Output) and GHI(Observed vs Model Output), row-wise corresponding to each of the four special days of the year 2017.


2. solar_project_evaluation_V2.py

	Call upon   : None
	Called by   : None

	Description :

		This python code evaluates the solar elevation angle and GHI at four major Indian metro cities (Delhi, Mumbai, Kolkata, Chennai) on 2x Solstices and 2x equinoxes in 2019 		against the observed data fetched from CERES website.

		Note: At line #235 to #237 and #244 to #246 of this code, please insert a valid location of the observed data.

	Input	: Observed data CERES_SYN1deg-1H_Terra-Aqua-MODIS (https://ceres-tool.larc.nasa.gov/ord-tool/jsp/SYN1degEd41Selection.jsp)
		  Variables Fetched - Solar Insolation Flux(All sky), Elevation above sea level

		  Model Input - City_Name = Delhi/Mumbai/Kolkata/Chennai
		  		lat / lon = Latitudes/Longitudes of these selected cities
                  		tz = 5.5
                  		Year = 2019
                  		Month = March/June/September/December
                  		Day = 21/21/23/22

	Output	: A matplotlib 2D plot having Solar Elevation (both observed and model output), Solar Elevation (Observed vs Model Output), GHI (both Observed and Model Output) and GHI (Observed vs Model Output), row-wise corresponding to each city. Such 4x plots have been generated tinkering with the date and month for four special days of the year 2019.

