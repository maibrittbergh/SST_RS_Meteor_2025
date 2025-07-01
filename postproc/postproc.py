#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.constants import h, c, k
from scipy.integrate import quad
from scipy.interpolate import interp1d
import pandas as pd 
import matplotlib.pyplot as plt
from pathlib import Path


# In[ ]:


# === SETTINGS ===
date_str = "20250626" # Date in YYYYMMDD format
ID = "A2"             # Station ID
action_number = "17"  # 17 for Halobates
begin_meas = pd.to_datetime("2025-06-26 08:30")
end_meas = pd.to_datetime("2025-06-27 08:30")



# === File Paths ===
data_dir = Path("/Users/maibrittberghofer/Desktop/Arbeit /Meteor/python_scripts_2025_meteor/data/")
water_file = data_dir / f"{date_str}_{ID}_{action_number}_IR_Water.csv"
sky_file = data_dir / f"{date_str}_{ID}_{action_number}_IR_Sky.csv"

# Output path for the raw SST temperature
out_dir = Path("/Users/maibrittberghofer/Desktop/Arbeit /Meteor/python_scripts_2025_meteor/postproc/raw")
sst_output_file = out_dir / f"{date_str}_{ID}_{action_number}_IR_SST.csv"

# Output path for the averaged SST temperature
out_dir_avg = Path("/Users/maibrittberghofer/Desktop/Arbeit /Meteor/python_scripts_2025_meteor/postproc/1min")
sst_avg_file = out_dir_avg / f"{date_str}_{ID}_{action_number}_IR_SST_1min.csv"


# In[ ]:


# === Define constants ===
lambda_min = 8     # micrometers
lambda_max = 14    # micrometers
emissivity = 0.98  # assumed emissivity of sea surface CHECK CHECK CHECK

# === Define temperature range for lookup tables ===
T_min = 280 # Kelvin
T_max = 310 # Kelvin
n_values = (T_max - T_min) * 1000 # 0.001 K steps


# ### Define functions and create lookups tables

# In[ ]:


# === Planck function ===
def planck_lambda(T, lam):
    """Spectral radiance [W/m²/sr/μm] at temperature T [K] and wavelength λ [μm]"""
    lam_m = lam * 1e-6  # convert μm to meters
    return (2*h*c**2 / lam_m**5) / (np.exp(h*c / (lam_m * k * T)) - 1)

# === Convert spectral radiance to band-integrated radiance ===
def band_integrated_radiance(T, lam1=lambda_min, lam2=lambda_max):
    """Band-integrated Planck radiance over [λ1, λ2] for temperature T"""
    result, _ = quad(lambda l: planck_lambda(T, l), lam1, lam2)
    return result


# In[ ]:


# === Step 1:Generate lookup table of temperature-band intergated radiance values ===
T_range = np.linspace(T_min, T_max + 1, n_values)  # Kelvin
radiance_lookup = np.array([band_integrated_radiance(T) for T in T_range])

# Interpolation function to invert Planck's law
radiance_to_temp = interp1d(radiance_lookup, T_range, bounds_error=False, fill_value="extrapolate")


# ### Load and prepare the datasets

# In[53]:


# === Load water and sky data ===
df_water = pd.read_csv(water_file, sep=";", decimal=",")
df_sky = pd.read_csv(sky_file, sep=";", decimal=",")

# Remove white spaces from the columns titles
df_water.columns = df_water.columns.str.strip() 
df_sky.columns = df_sky.columns.str.strip()


# In[54]:


# === Parse datetime ===
df_water['Time'] = pd.to_datetime(df_water['Date'], format='%b %d %Y %I:%M:%S %p')
df_sky['Time'] = pd.to_datetime(df_sky['Date'], format='%b %d %Y %I:%M:%S %p')

# === Determine overlapping time window ===
start_time = max(df_water['Time'].min(), df_sky['Time'].min())
end_time = min(df_water['Time'].max(), df_sky['Time'].max())

print("Overlapping time window: \nStart time:", start_time, "\nEnd Time:", end_time)



# Remove the data before and after measurement begin and measurement ending 

#df_cut = df[(df['timestamp'] >= begin_meas) & (df['timestamp'] <= end_meas)]
#df_water=df_water[(df_water['Time'] >= begin_meas) & (df_water['Time'] <= end_meas)]
#df_sky=df_sky[(df_sky['Time'] >= begin_meas) & (df_sky['Time'] <= end_meas)]


# In[55]:


# === Filter data to overlapping time window ===
df_water_overlap = df_water[(df_water['Time'] >= start_time) & (df_water['Time'] <= end_time)].copy()
df_sky_overlap = df_sky[(df_sky['Time'] >= start_time) & (df_sky['Time'] <= end_time)].copy()

# === Reset index after filtering ===
df_water_overlap.reset_index(drop=True, inplace=True)
df_sky_overlap.reset_index(drop=True, inplace=True)

# === Compare lengths ===
len_water= len(df_water_overlap)
len_sky = len(df_sky_overlap)

print(f"Water rows: {len_water}, Sky rows: {len_sky}")

# === Compare time columns to check alignment ===
if len_water == len_sky:
    time_diff = df_water_overlap['Time'] - df_sky_overlap['Time']
    if all(time_diff == pd.Timedelta(0)):
        print("Timestamps are perfectly aligned.")
    else:
        print("Same length but timestamps do not match exactly.")
        print(time_diff.head())
else:
    print("Mismatch in number of data points after filtering.")


# #### Look for data gaps or duplicates 
# These cells have to be run only if the datasets are not perfecly aligned.
# 
# If they are, skip this and go directly to the next section "Calculate SST temperature"

# In[111]:


# === Find data gaps - if any ===
def find_time_gaps(df, name, expected_interval='1s'):
    df = df.sort_values('Time').reset_index(drop=True)
    time_diffs = df['Time'].diff()

    # Find where the time difference is greater than expected
    gaps = time_diffs[time_diffs > pd.to_timedelta(expected_interval)]

    if gaps.empty:
        print(f"No gaps detected in {name}.")
    else:
        print(f"Gaps found in {name}:")
        for idx in gaps.index:
            gap_start = df.loc[idx - 1, 'Time']
            gap_end = df.loc[idx, 'Time']
            print(f"   - Gap from {gap_start} to {gap_end} ({gap_end - gap_start})")

    return gaps

# === Find gaps in both datasets ===
gaps_water = find_time_gaps(df_water_overlap, "Water Camera")
gaps_sky = find_time_gaps(df_sky_overlap, "Sky Camera")


# In[112]:


# === Check sampling intervals and statitics ===
df_water_overlap['dt'] = df_water_overlap['Time'].diff().dt.total_seconds()
print("Water sampling stats:")
print(df_water_overlap['dt'].describe())

df_sky_overlap['dt'] = df_sky_overlap['Time'].diff().dt.total_seconds()
print("Sky sampling stats:")
print(df_sky_overlap['dt'].describe())


# In[113]:


# === Check for duplicates ===
def check_duplicates(df, name):
    dups = df[df.duplicated(subset='Time', keep=False)]
    if dups.empty:
        print(f"No duplicate timestamps in {name}.")
    else:
        print(f"Found {len(dups)} duplicate rows in {name}.")
        print(dups.head())
    return dups

dups_water = check_duplicates(df_water_overlap, "Water Camera")
dups_sky = check_duplicates(df_sky_overlap, "Sky Camera")












# === Drop rows with zero time difference and exact same temperature values === 

def drop_exact_duplicate_timestamps(df, name, value_column='Target(°C)'):
    df = df.sort_values('Time').reset_index(drop=True)
    df['dt'] = df['Time'].diff()

    # Find rows where the timestamp is duplicated (dt == 0)
    duplicates = df[df['dt'] == pd.Timedelta(0)]
    dropped_indices = []

    for idx in duplicates.index:
        curr_val = df.loc[idx, value_column]
        prev_val = df.loc[idx - 1, value_column]

        if curr_val == prev_val:
            dropped_indices.append(idx)
        else:
            print(f"Different values at duplicate timestamp (index {idx}): {prev_val} ≠ {curr_val}")

    df_cleaned = df.drop(index=dropped_indices).reset_index(drop=True)

    print(f"Dropped {len(dropped_indices)} exact duplicate rows from {name}.")

    return df_cleaned.drop(columns='dt')



df_water_overlap = drop_exact_duplicate_timestamps(df_water_overlap, "Water Camera")
df_sky_overlap = drop_exact_duplicate_timestamps(df_sky_overlap, "Sky Camera")



# In[115]:


# === Final alignment check ===
if len(df_water_overlap) != len(df_sky_overlap):
    raise ValueError(f"Mismatch in row counts: Water = {len(df_water_overlap)}, Sky = {len(df_sky_overlap)}")

# Check timestamp alignment
if not (df_water_overlap['Time'].values == df_sky_overlap['Time'].values).all():
    raise ValueError("Timestamps are not aligned between water and sky datasets.")

# Reset index just to be safe
df_water_overlap = df_water_overlap.reset_index(drop=True)
df_sky_overlap = df_sky_overlap.reset_index(drop=True)

print("Water and sky datasets are fully aligned and ready for SST calculation.")


# ### Calculate SST temperature

# In[56]:


# === Extract temperature arrays and convert to Kelvin ===
T_water = df_water_overlap['Target(°C)'].values + 273.15
T_sky = df_sky_overlap['Target(°C)'].values + 273.15


# In[ ]:


# Step 2: Convert temperatures to radiances
L_measured_array = np.array([band_integrated_radiance(T) for T in T_water])
L_sky_array = np.array([band_integrated_radiance(T) for T in T_sky])

# Step 3: Correct for reflection and emissivity
L_emit_array = (L_measured_array - (1 - emissivity) * L_sky_array) / emissivity

# Step 4: Invert radiance to get SST
T_sst_array = radiance_to_temp(L_emit_array)

# Convert back to Celsius
T_sst_array_C = T_sst_array - 273.15 

# === Create df with "Time" column ===
df_sst = pd.DataFrame({
    "Time": df_water_overlap["Time"].values,
    "SST (°C)": T_sst_array_C })


# In[ ]:


# Save SST data to CSV
df_sst.to_csv(sst_output_file, index=False)


# In[59]:


# === Plot ===
plt.figure(figsize=(10, 4))
plt.plot(df_sst["Time"], df_sst["SST (°C)"], label="SST")

plt.xlabel("Time (UTC)")
plt.ylabel("SST[°C]")

plt.ylim(20, 25)

plt.xticks(rotation=-90)
plt.title("Sea Surface Temperature")
plt.grid(True) 
#plt.legend()
plt.tight_layout()

plt.show()


# ### Calculate averages

# In[ ]:


# -- Calculate 1 min averages --
df_sst_avg = df_sst.copy() # Copy SST dataset

df_sst_avg = df_sst_avg.set_index("Time") # Set Time as index for resampling 

df_sst_avg = df_sst_avg.resample("1min").mean() # Average over 1 minute intervals

# OPTIONAL: save 1 min averages to CSV to the right folder
df_sst_avg.to_csv(sst_avg_file)


# In[ ]:


# PLOT
plt.figure(figsize=(10, 5))
plt.plot(df_sst["Time"], df_sst["SST (°C)"], label="Raw SST", alpha=0.5)
plt.plot(df_sst_avg.index, df_sst_avg["SST (°C)"], label="1-Min Average", linewidth=2, color='red')

plt.xlabel("Time (UTC)")
plt.ylabel("SST [°C]")
plt.title("Sea Surface Temperature (Raw vs. 1-Min Average)")
plt.xticks(rotation=-90)

plt.ylim(20, 25)


plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ### Subset

# In[ ]:


# === Option A: Subset for a specific date ===
date = pd.to_datetime("2025-06-26").date() # Select date for subsetting
subset = df_sst[df_sst["Time"].dt.date == date] # Subset for the specific date


# In[ ]:


# == Option B: Subset for a specific time range ===
start_time = pd.to_datetime("2025-06-26 16:30:00")
end_time = pd.to_datetime("2025-06-26 16:40:00")
subset = df_sst[(df_sst["Time"] >= start_time) & (df_sst["Time"] <= end_time)]


# In[ ]:


# === Plot ===
plt.figure(figsize=(10, 4))
plt.plot(subset["Time"], subset["SST (°C)"], label="SST")

plt.xlabel("Time (UTC)")
plt.ylabel("SST[°C]")

plt.ylim(22, 24)

plt.xticks(rotation=-90)
plt.title("Sea Surface Temperature")
plt.grid(True) 
#plt.legend()
plt.tight_layout()

plt.show()

