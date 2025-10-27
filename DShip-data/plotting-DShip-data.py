# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 14:36:07 2025

@author: lotta
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

# create a folder for the plots
path = Path("plots")
path.mkdir(parents=True, exist_ok=True)

def calc_mixing_ratio(T_array,H_array,p_array): 
    # based on https://cran.r-project.org/web/packages/humidity/vignettes/humidity-measures.html

    # get T in K
    T_array=T_array+273.15
    
    # calculate saturation vapour pressure
    a = 17.2693882 #over water
    b = 35.86 #over water
    e_s = 6.1078 * np.exp(a*(T_array - 273.15)/(T_array - b)) # in hPa
    
    # calculate partial pressure of water vapour in air e
    e=H_array/100*e_s # in hPa
    
    # calculate specific humidity q (approx. formula)
    q=0.662*e/(p_array-0.378*e)
    
    # return mixing ratio w
    return q/(1-q)*1000 # in g/kg


def plot_DShip_data(file_name, CP_front_passage): 
    
    time = pd.read_csv(
        file_name,
        usecols=[0],
        skiprows=3,
        delimiter='\t',
        encoding='ANSI',
        parse_dates=[0]
    ).squeeze()
    
    
    # starboard air temperature [°C]
    T = pd.to_numeric(pd.read_csv(file_name, usecols=[4], skiprows=3, delimiter='\t', encoding='ANSI').squeeze(),
        errors='coerce')
    
    # starboard humidity [%]
    H = pd.to_numeric(pd.read_csv(file_name, usecols=[6], skiprows=3, delimiter='\t', encoding='ANSI').squeeze(),
        errors='coerce')
    
    # air pressure [hPa]
    p = pd.to_numeric(pd.read_csv(file_name, usecols=[1], skiprows=3, delimiter='\t', encoding='ANSI').squeeze(),
        errors='coerce')
    
    # wind speed (absolute) [m/s]
    u = pd.to_numeric(pd.read_csv(file_name, usecols=[9], skiprows=3, delimiter='\t', encoding='ANSI').squeeze(),
        errors='coerce')
    
    # wind direction (absolute) [°]
    phi = pd.to_numeric(pd.read_csv(file_name, usecols=[7], skiprows=3, delimiter='\t', encoding='ANSI').squeeze(),
        errors='coerce')
    
    # mixing ratio [g/kg]
    w= calc_mixing_ratio(T,H,p)
    
    # calculate averages
    T_pre=np.nanmean(T[:30*60])
    T_post=np.nanmean(T[30*60:])
    print("Temperature drop:", T_post-T_pre)
    
    w_pre=np.nanmean(w[:30*60])
    w_post=np.nanmean(w[30*60:])
    print("Mixing ratio change:", w_post-w_pre)
        
    # plotting
    fig, axs = plt.subplots(2,1, constrained_layout=True, figsize=(10, 6))
    
    # temperature and mixing ratio
    axs[0].set_title("Temperature and mixing ratio")
    axs[0].set_xlabel(r"Time")
    axs[0].set_xlim(min(time), max(time))
    
    # set major ticks every 5 minute
    axs[0].xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
    
    # format tick labels to show hour:minute
    axs[0].xaxis.set_major_formatter(DateFormatter("%H:%M"))
    
    color1 = 'tab:blue'
    axs[0].plot(time, T, color=color1)
    axs[0].set_ylabel(r"Temperature [°C]", color=color1)
    axs[0].tick_params(axis='y', labelcolor=color1)
    
    ax2 = axs[0].twinx()  # instantiate a second Axes that shares the same x-axis
    color2 = 'tab:green'
    ax2.plot(time, w, color=color2)
    ax2.set_ylabel('Mixing ratio [g/kg]', color=color2)  
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # add vertical line for CP front passage
    plt.axvline(x=pd.to_datetime(CP_front_passage), color='red', linestyle='--', linewidth=2)
    axs[0].text(pd.to_datetime(CP_front_passage)+pd.Timedelta(seconds=30), min(T), 'CP front passage',
             verticalalignment='bottom',
             horizontalalignment='left', 
             color='red')
    
    # show date in upper left corner
    axs[0].text(0.01, 0.96, time[0].strftime('%Y-%m-%d'), transform=axs[0].transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='left')
    
    # wind speed and direction
    axs[1].set_title("Wind speed and direction")
    axs[1].set_xlabel(r"Time")
    axs[1].set_xlim(min(time), max(time))
    
    # set major ticks every 5 minute
    axs[1].xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
    
    # format tick labels to show hour:minute
    axs[1].xaxis.set_major_formatter(DateFormatter("%H:%M"))
    
    color3 = 'tab:cyan'
    axs[1].plot(time, u, color=color3)
    axs[1].set_ylabel(r"Wind speed (absolute) [m/s]", color=color3)
    axs[1].tick_params(axis='y', labelcolor=color3)
    
    ax4 = axs[1].twinx()  # instantiate a second Axes that shares the same x-axis
    color4 = 'tab:purple'
    ax4.plot(time, phi, color=color4)
    ax4.set_ylabel('Wind direction (absolute) [°]', color=color4)  
    ax4.tick_params(axis='y', labelcolor=color4)
    
    # add vertical line for CP front passage
    plt.axvline(x=pd.to_datetime(CP_front_passage), color='red', linestyle='--', linewidth=2)
    axs[1].text(pd.to_datetime(CP_front_passage)+pd.Timedelta(seconds=30), min(u), 'CP front passage',
             verticalalignment='bottom',
             horizontalalignment='left', 
             color='red', zorder=10)
    
    plt.savefig('plots/'+Path(file_name).stem+'.png', dpi=1200) # save figure
    plt.show()


plot_DShip_data('data/CP-20250713-0947.dat', '2025-07-13 09:47:00')
plot_DShip_data('data/CP-20250714-1904.dat', '2025-07-14 19:04:00')
plot_DShip_data('data/CP-20250716-1010.dat', '2025-07-16 10:10:00')
plot_DShip_data('data/CP-20250716-1859.dat', '2025-07-16 18:59:00')
plot_DShip_data('data/CP-20250717-1609.dat', '2025-07-17 16:09:00')

