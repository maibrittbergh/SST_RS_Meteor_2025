#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 16:53:35 2025

@author: maibrittberghofer
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File to read and visualize the sounding data. 

@author: maibrittberghofer
"""


#read packages 


import numpy as np
import matplotlib.pyplot as plt
from metpy.plots import SkewT
from metpy.units import units
from eccodes import codes_bufr_new_from_file, codes_set, codes_get_array, codes_release
import numpy as np


#enter balloon specifications 



filename = '/Users/maibrittberghofer/Desktop/Arbeit /Meteor/Sounding Data/20250616_12UTC_Bufr_309057_all.dat'
date="20250616"
time="12UTC"


data = {
    "airTemperature": [],
    "pressure": [],
    "dewpointTemperature": [],
    "windDirection": [],
    "windSpeed": []
}

with open(filename, 'rb') as f:
    while True:
        bufr = codes_bufr_new_from_file(f)
        if bufr is None:
            break
        try:
            codes_set(bufr, 'unpack', 1)
            for key in data:
                try:
                    values = codes_get_array(bufr, key)
                    data[key].extend(values)
                except Exception:
                    pass
        except Exception as e:
            print("Skipping message:", e)
        finally:
            codes_release(bufr)

# Print first few entries
for key in data:
    print(f"{key}: {data[key][:5]}")



#Convert data: Pa to hPa, Kelvin to °C
pressure = np.array(data["pressure"]) * 0.01 * units.hPa  # Convert from Pa to hPa
temperature = (np.array(data["airTemperature"]) - 273.15) * units.degC
dewpoint = (np.array(data["dewpointTemperature"]) - 273.15) * units.degC


wind_speed = np.array(data['windSpeed']) * units('m/s')
wind_dir = np.array(data['windDirection']) * units.degree

# Calculate directional components of wind 
u_wind = -wind_speed * np.sin(np.radians(wind_dir.m))
v_wind = -wind_speed * np.cos(np.radians(wind_dir.m))



# Extract missing values 
mask = (
    np.isfinite(pressure.m) &
    np.isfinite(temperature.m) &
    np.isfinite(dewpoint.m) &
    (pressure.m > 50) &
    (pressure.m < 1100)
)

pressure = pressure[mask]
temperature = temperature[mask]
dewpoint = dewpoint[mask]

sort_idx = np.argsort(pressure.m)[::-1]
pressure = pressure[sort_idx]
temperature = temperature[sort_idx]
dewpoint = dewpoint[sort_idx]

u_wind = u_wind[sort_idx]
v_wind = v_wind[sort_idx]

# Subsample
barb_step = 20
pressure_barb = pressure[::barb_step]
u_barb = u_wind[::barb_step]
v_barb = v_wind[::barb_step]


# Plot
skew = SkewT()

skew.plot(pressure, temperature, 'r', linewidth=2, label='Temperature')
skew.plot(pressure, dewpoint, 'g', linewidth=2, label='Dew Point')
skew.plot_barbs(pressure_barb, u_barb, v_barb)

skew.ax.set_ylim(1050, 100)
skew.ax.set_xlim(-50, 50)
skew.ax.set_yscale('log')
skew.ax.grid(True)
skew.ax.legend()

plt.xlabel("Temperature [°C]")
plt.ylabel("Pressure [hPa]")
plt.title(f'Skew-T log-P: {date} at {time}')
plt.tight_layout()
plt.show()







#height in the atmosphere: 50 hpa ~500 m 
# height at 200 hPa ~ 800/50=16 -> 