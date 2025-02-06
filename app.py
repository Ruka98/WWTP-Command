import streamlit as st
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Geod

# Streamlit UI for file upload
st.title("DEM File Viewer")

uploaded_file = st.file_uploader("Upload a DEM file", type=["sdat", "tif", "tiff"])

if uploaded_file:
    # Read the uploaded file with rasterio
    with rasterio.open(uploaded_file) as src:
        dem_data = src.read(1)
        transform = src.transform
        bounds = src.bounds
        pixel_size = transform[0]
        no_data_value = src.nodata if src.nodata else -99999.0
        dem_data = np.ma.masked_where(dem_data == no_data_value, dem_data)

    # Initialize Geod for accurate distance calculations
    geod = Geod(ellps='WGS84')

    # Display basic info
    st.write(f"Bounds: {bounds}")
    st.write(f"Pixel size: {pixel_size}")

    # Display DEM
    fig, ax = plt.subplots()
    cax = ax.imshow(dem_data, cmap="terrain")
    fig.colorbar(cax, ax=ax)
    st.pyplot(fig)
else:
    st.warning("Please upload a DEM file.")
