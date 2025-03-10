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

def geo_to_pixel(lon, lat, transform):
    col, row = ~transform * (lon, lat)
    return int(col), int(row)

def pixel_to_geo(x, y, transform):
    lon, lat = rasterio.transform.xy(transform, y, x, offset='center')
    return lon, lat

def get_circular_pixels(lon, lat, radius_m, transform, geod, dem_shape):
    x_center, y_center = geo_to_pixel(lon, lat, transform)
    if (x_center < 0 or x_center >= dem_shape[1] or 
        y_center < 0 or y_center >= dem_shape[0]):
        return []
    clon, clat = pixel_to_geo(x_center, y_center, transform)
    east_x = x_center + 1
    east_lon, east_lat = pixel_to_geo(east_x, y_center, transform)
    _, _, dist_east = geod.inv(clon, clat, east_lon, east_lat)
    north_y = y_center + 1
    north_lon, north_lat = pixel_to_geo(x_center, north_y, transform)
    _, _, dist_north = geod.inv(clon, clat, north_lon, north_lat)
    max_dx = int(np.ceil(radius_m / dist_east)) if dist_east !=0 else 0
    max_dy = int(np.ceil(radius_m / dist_north)) if dist_north !=0 else 0
    circular_pixels = []
    for dx in range(-max_dx, max_dx +1):
        for dy in range(-max_dy, max_dy +1):
            x = x_center + dx
            y = y_center + dy
            if x <0 or x >= dem_shape[1] or y <0 or y >= dem_shape[0]:
                continue
            plon, plat = pixel_to_geo(x, y, transform)
            _, _, distance = geod.inv(lon, lat, plon, plat)
            if distance <= radius_m:
                circular_pixels.append( (y, x) )
    return circular_pixels

def trace_downstream(start_lon, start_lat, initial_volume, transform, dem_data, geod):
     evaporation_rate = 5.787e-8  # m/s (5mm/day)
     infiltration_rate = 1e-7     # m/s (0.0001mm/s)
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
        
        next_lon, next_lat = pixel_to_geo(best_nx, best_ny, transform)
        _, _, distance = geod.inv(current_lon, current_lat, next_lon, next_lat)
        elevation_diff = min_elev - current_elev
        
        slope = -elevation_diff / distance if distance > 0 else 0
        velocity = np.sqrt(9.81 * slope) if slope > 0 else 0
        time = distance / velocity if velocity > 0 else 0
        
        evaporation_loss = evaporation_rate * (canal_width * distance) * time
        infiltration_loss = infiltration_rate * (canal_width * distance) * time
        total_loss = evaporation_loss + infiltration_loss
        
        volume -= total_loss
        
        if volume <= 0:
            break
        
        x, y = best_nx, best_ny
        current_elev = dem_data[y, x]
        current_lon, current_lat = next_lon, next_lat
        path.append( (x, y) )
    
    return path

def main():
    st.title("WWTP Model Interface")

    # File uploaders
    dem_file = st.file_uploader("Upload DEM file (.sdat)", type=["sdat"])
    excel_file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])
    output_dir = st.text_input("Output directory", "output")

    if st.button("Run Model"):
        if not dem_file or not excel_file:
            st.error("Please upload both DEM and Excel files.")
            return

        os.makedirs(output_dir, exist_ok=True)

        try:
            # Save DEM to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".sdat") as tmp_dem:
                tmp_dem.write(dem_file.getvalue())
                dem_path = tmp_dem.name

            # Read Excel data
            df = pd.read_excel(excel_file)
            wwtp_locations = []
            for index, row in df.iterrows():
                wwtp_locations.append((row['Longitude'], row['Latitude'], row['Volume']))

            # Open DEM and process
            with rasterio.open(dem_path) as src:
                dem_data = src.read(1)
                transform = src.transform
                crs = src.crs
                no_data_value = src.nodata if src.nodata else -99999.0
                dem_data = np.ma.masked_where(dem_data == no_data_value, dem_data)

                geod = Geod(ellps="WGS84")
                command_areas = np.zeros_like(dem_data, dtype=np.int32)
                streamlines = []
                wwtp_points = []

                for idx, (lon, lat, vol) in enumerate(wwtp_locations, 1):
                    try:
                        x_center, y_center = geo_to_pixel(lon, lat, transform)
                        if (x_center < 0 or x_center >= dem_data.shape[1] or 
                            y_center < 0 or y_center >= dem_data.shape[0]):
                            st.warning(f"WWTP {idx} is outside DEM bounds.")
                            continue
                        wwtp_elev = dem_data[y_center, x_center]
                        if np.ma.is_masked(wwtp_elev):
                            st.warning(f"WWTP {idx} has no elevation data.")
                            continue

                        # Part 1: 5km circular buffer with elevation within 50m
                        circular_pixels = get_circular_pixels(lon, lat, 5000, transform, geod, dem_data.shape)
                        for (y, x) in circular_pixels:
                            if dem_data[y, x] <= wwtp_elev + 50:
                                command_areas[y, x] = idx

                        # Part 2: Trace downstream
                        downstream_path = trace_downstream(lon, lat, vol, transform, dem_data, geod)
                        if downstream_path:
                            downstream_coords = [pixel_to_geo(x, y, transform) for (x, y) in downstream_path]
                            streamlines.append(downstream_coords)
                            # Create 1km buffer around downstream path
                            for (x_p, y_p) in downstream_path:
                                plon, plat = pixel_to_geo(x_p, y_p, transform)
                                buffer_pixels = get_circular_pixels(plon, plat, 1000, transform, geod, dem_data.shape)
                                for (y_b, x_b) in buffer_pixels:
                                    command_areas[y_b, x_b] = idx

                        wwtp_points.append(Point(lon, lat))
                    except Exception as e:
                        st.error(f"Error processing WWTP {idx}: {e}")

                # Generate command areas GeoJSON
                features_list = []
                wwtp_names = df['WWTP'].tolist()
                for geom, value in features.shapes(command_areas, transform=transform):
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
                    if crs != 'EPSG:4326':
                        gdf = gdf.to_crs(epsg=4326)
                    output_file = os.path.join(output_dir, "OP2wwtp_command_areas.geojson")
                    gdf.to_file(output_file, driver='GeoJSON')
                    st.success(f"Command areas saved to {output_file}")

                # Save streamlines
                if streamlines:
                    streamline_geometries = [LineString(coords) for coords in streamlines]
                    streamline_gdf = gpd.GeoDataFrame(geometry=streamline_geometries, crs=crs)
                    if crs != 'EPSG:4326':
                        streamline_gdf = streamline_gdf.to_crs(epsg=4326)
                    output_file = os.path.join(output_dir, "OP2streamlines.geojson")
                    streamline_gdf.to_file(output_file, driver='GeoJSON')
                    st.success(f"Streamlines saved to {output_file}")

                # Save WWTP points
                if wwtp_points:
                    wwtp_gdf = gpd.GeoDataFrame(geometry=wwtp_points, crs='EPSG:4326')
                    wwtp_gdf['name'] = df['WWTP']
                    output_file = os.path.join(output_dir, "OP2wwtp_locations.geojson")
                    wwtp_gdf.to_file(output_file, driver='GeoJSON')
                    st.success(f"WWTP locations saved to {output_file}")

            # Clean up temp DEM file
            os.unlink(dem_path)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
