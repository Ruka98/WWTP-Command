import streamlit as st
import rasterio
import numpy as np
from pyproj import Geod
import geopandas as gpd
from rasterio import features
import pandas as pd
from shapely.geometry import Point, LineString
import os
import tempfile
import shutil

# Coordinate conversion functions
def geo_to_pixel(lon, lat, transform):
    col, row = ~transform * (lon, lat)
    return int(col), int(row)

def pixel_to_geo(x, y, transform):
    lon, lat = rasterio.transform.xy(transform, y, x, offset='center')
    return lon, lat

def get_circular_pixels(lon, lat, radius_m, transform, geod, dem_shape, dem_data, wwtp_elev=None, elevation_threshold=None):
    x_center, y_center = geo_to_pixel(lon, lat, transform)
    if (x_center < 0 or x_center >= dem_shape[1] or 
        y_center < 0 or y_center >= dem_shape[0]):
        return []
    
    circular_pixels = []
    clon, clat = pixel_to_geo(x_center, y_center, transform)
    east_x = x_center + 1
    east_lon, east_lat = pixel_to_geo(east_x, y_center, transform)
    _, _, dist_east = geod.inv(clon, clat, east_lon, east_lat)
    
    max_dx = int(np.ceil(radius_m / dist_east)) if dist_east != 0 else 0
    
    for dx in range(-max_dx, max_dx + 1):
        for dy in range(-max_dx, max_dx + 1):
            x = x_center + dx
            y = y_center + dy
            if 0 <= x < dem_shape[1] and 0 <= y < dem_shape[0]:
                plon, plat = pixel_to_geo(x, y, transform)
                _, _, distance = geod.inv(lon, lat, plon, plat)
                if distance <= radius_m:
                    pixel_elev = dem_data[y, x]
                    elev_condition = True
                    if wwtp_elev is not None and elevation_threshold is not None:
                        elev_condition = (pixel_elev <= wwtp_elev + elevation_threshold)
                    if not np.ma.is_masked(pixel_elev) and elev_condition:
                        circular_pixels.append((y, x))
    
    return circular_pixels

def trace_downstream(start_lon, start_lat, initial_volume, transform, dem_data, geod):
    # Hydrological parameters
    evaporation_rate = 7e-3 / 86400  # 7 mm/day converted to m/s
    infiltration_rate = 1e-7       # 0.0001 mm/s converted to m/s
    canal_width = 10.0           # meters
    
    x, y = geo_to_pixel(start_lon, start_lat, transform)
    if x < 0 or x >= dem_data.shape[1] or y < 0 or y >= dem_data.shape[0]:
        return []
    current_elev = dem_data[y, x]
    if np.ma.is_masked(current_elev):
        return []
    path = [ (x, y) ]
    volume = initial_volume
    current_lon, current_lat = pixel_to_geo(x, y, transform)
    
    while volume > 0:
        min_elev = current_elev
        best_nx, best_ny = x, y
        # Find steepest downhill neighbor
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx = x + dx
                ny = y + dy
                if nx < 0 or nx >= dem_data.shape[1] or ny < 0 or ny >= dem_data.shape[0]:
                    continue
                elev = dem_data[ny, nx]
                if np.ma.is_masked(elev):
                    continue
                if elev < min_elev:
                    min_elev = elev
                    best_nx, best_ny = nx, ny
        if (best_nx, best_ny) == (x, y):
            break  # No downhill neighbor
        
        # Calculate distance to next pixel
        next_lon, next_lat = pixel_to_geo(best_nx, best_ny, transform)
        _, _, distance = geod.inv(current_lon, current_lat, next_lon, next_lat)
        # Elevation difference (next - current)
        elevation_diff = min_elev - current_elev
        
        # Calculate losses (downhill flow)
        slope = -elevation_diff / distance if distance > 0 else 0
        velocity = np.sqrt(9.81 * slope) if slope > 0 else 0
        time = distance / velocity if velocity > 0 else 0
        
        evaporation_loss = evaporation_rate * (canal_width * distance) * time
        infiltration_loss = infiltration_rate * (canal_width * distance) * time
        total_loss = evaporation_loss + infiltration_loss
        
        # Deduct losses from volume
        volume -= total_loss
        
        if volume <= 0:
            break  # Stop if volume is depleted
        
        # Update current position and elevation
        x, y = best_nx, best_ny
        current_elev = dem_data[y, x]
        current_lon, current_lat = next_lon, next_lat
        path.append( (x, y) )
    
    return path

def main():
    st.title("WWTP Model Interface")
    
    dem_file = st.file_uploader("Upload DEM file (.tif)", type=["tif"])
    excel_file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])
    output_dir = st.text_input("Output directory", "output")
    
    if st.button("Run Model"):
        if not dem_file or not excel_file:
            st.error("Please upload both DEM and Excel files.")
            return

        os.makedirs(output_dir, exist_ok=True)

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp_dem:
                tmp_dem.write(dem_file.getvalue())
                dem_path = tmp_dem.name
                st.write(f"Temporary DEM file saved to: {dem_path}")

            df = pd.read_excel(excel_file)
            wwtp_locations = df[['Longitude', 'Latitude', 'Volume']].values.tolist()

            with rasterio.open(dem_path) as src:
                dem_data = src.read(1, masked=True)
                transform = src.transform
                crs = src.crs
                
                geod = Geod(ellps="WGS84")
                command_areas = np.zeros_like(dem_data.filled(0), dtype=np.int32)
                wwtp_points = []
                streamlines = []

                for idx, (lon, lat, vol) in enumerate(wwtp_locations, 1):
                    try:
                        x_center, y_center = geo_to_pixel(lon, lat, transform)
                        if not (0 <= x_center < dem_data.shape[1] and 0 <= y_center < dem_data.shape[0]):
                            st.warning(f"WWTP {idx} is outside DEM bounds.")
                            continue
                        
                        wwtp_elev = dem_data[y_center, x_center]
                        if np.ma.is_masked(wwtp_elev):
                            st.warning(f"WWTP {idx} has no elevation data.")
                            continue
                        
                        # Part 1: 5km circular buffer with elevation up to 50m higher than WWTP
                        circular_pixels = get_circular_pixels(lon, lat, 5000, transform, geod, dem_data.shape, dem_data, wwtp_elev, elevation_threshold=50)
                        for (y, x) in circular_pixels:
                            command_areas[y, x] = idx
                        
                        # Part 2: Trace downstream until volume is depleted and buffer 1km around path
                        downstream_path = trace_downstream(lon, lat, vol, transform, dem_data, geod)
                        if downstream_path and len(downstream_path) >= 2:  # Ensure at least 2 points
                            # Convert downstream path to geographic coordinates
                            downstream_coords = [pixel_to_geo(x, y, transform) for (x, y) in downstream_path]
                            streamlines.append(downstream_coords)
                            # Create 1km buffer around downstream path (no elevation check)
                            for (x_p, y_p) in downstream_path:
                                plon, plat = pixel_to_geo(x_p, y_p, transform)
                                buffer_pixels = get_circular_pixels(plon, plat, 1000, transform, geod, dem_data.shape, dem_data)
                                for (y_b, x_b) in buffer_pixels:
                                    command_areas[y_b, x_b] = idx
                        
                        # Save WWTP location as a point
                        wwtp_points.append(Point(lon, lat))
                    except Exception as e:
                        st.error(f"Error processing WWTP {idx}: {e}")
                
                mask = command_areas > 0
                features_list = []
                wwtp_names = df['WWTP'].tolist()
                
                for geom, value in features.shapes(command_areas, mask=mask, transform=transform):
                    if value == 0:
                        continue
                    wwtp_index = int(value)
                    if 1 <= wwtp_index <= len(wwtp_names):
                        features_list.append({
                            'geometry': geom,
                            'properties': {'name': wwtp_names[wwtp_index-1], 'id': wwtp_index}
                        })
                
                if features_list:
                    gdf = gpd.GeoDataFrame.from_features(features_list, crs=crs)
                    gdf = gdf.dissolve(by='id').reset_index()
                    gdf = gdf.to_crs(epsg=4326)
                    output_file = os.path.join(output_dir, "OP2wwtp_command_areas.geojson")
                    gdf.to_file(output_file, driver='GeoJSON')
                    st.success(f"Command areas saved to {output_file}")
                
                if wwtp_points:
                    wwtp_gdf = gpd.GeoDataFrame(geometry=wwtp_points, crs='EPSG:4326')
                    wwtp_gdf['name'] = df['WWTP']
                    output_file = os.path.join(output_dir, "OP2wwtp_locations.geojson")
                    wwtp_gdf.to_file(output_file, driver='GeoJSON')
                    st.success(f"WWTP locations saved to {output_file}")
                
                if streamlines:
                    streamline_geometries = [LineString(coords) for coords in streamlines if len(coords) >= 2]  # Ensure valid LineString
                    if streamline_geometries:
                        streamline_gdf = gpd.GeoDataFrame(geometry=streamline_geometries, crs=crs)
                        streamline_gdf = streamline_gdf.to_crs(epsg=4326)
                        output_file = os.path.join(output_dir, "OP2streamlines.geojson")
                        streamline_gdf.to_file(output_file, driver='GeoJSON')
                        st.success(f"Streamlines saved to {output_file}")
                    else:
                        st.warning("No valid streamlines to save.")
                
            os.unlink(dem_path)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

        # Display output files and provide download links
        st.subheader("Output Files")
        output_files = os.listdir(output_dir)
        if output_files:
            for file in output_files:
                file_path = os.path.join(output_dir, file)
                with open(file_path, "rb") as f:
                    st.download_button(
                        label=f"Download {file}",
                        data=f,
                        file_name=file,
                        mime="application/octet-stream"
                    )
        else:
            st.warning("No output files generated.")

if __name__ == "__main__":
    main()
