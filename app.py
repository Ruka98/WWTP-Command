import rasterio
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, simpledialog
from pyproj import Geod

# File upload dialog
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(filetypes=[('DEM Files', '*.sdat;*.tif;*.tiff')])

if not file_path:
    raise ValueError('No file selected. Please select a DEM file.')

# Open the selected DEM file using rasterio
with rasterio.open(file_path) as src:
    dem_data = src.read(1)
    transform = src.transform
    bounds = src.bounds
    pixel_size = transform[0]
    no_data_value = src.nodata if src.nodata else -99999.0
    dem_data = np.ma.masked_where(dem_data == no_data_value, dem_data)

# Initialize Geod for accurate distance calculations
geod = Geod(ellps='WGS84')

def geo_to_pixel(lon, lat, transform):
    col, row = ~transform * (lon, lat)
    return int(col), int(row)

def pixel_to_geo(x, y, transform):
    lon, lat = rasterio.transform.xy(transform, y, x, offset='center')
    return lon, lat

# Save logic
print('Application setup complete.')