import srtm
import requests
import numpy as np
import os


def get_elevation(lat, lon, srtm_path):
    elevation_data = srtm.get_data(local_cache_dir=srtm_path)
    elevation = elevation_data.get_elevation(lat, lon)
    if elevation is None:
        lat = round(round(lat * 1200, 4) / 1200.0, 2)
        lon = round(round(lon * 1200, 4) / 1200.0, 2)
        elevation = elevation_data.get_elevation(lat, lon)
    if elevation is None:
        lat_prefix = 'N' if lat >= 0 else 'S'
        lon_prefix = 'E' if lon >= 0 else 'W'
        hgt_file = f"{lat_prefix}{int(abs(lat)):02d}{lon_prefix}{int(abs(lon)):03d}.hgt"
        file_path = os.path.join(srtm_path, hgt_file)

        if not os.path.exists(file_path):
            elevation = 0
        else:
            elevation = get_elevation_online(lat, lon)

    return elevation


def get_elevation_online(lat, lon):
    print("Online Querying")
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat}, {lon}"
    response = requests.get(url)
    if response.status_code == 200:
        elevation_data = response.json()
        return elevation_data['results'][0]['elevation']
    else:
        raise Exception("Error in API request: " + str(response.status_code))


def interpolate_points(point1, point2, num_points=15):
    lat1, lon1 = point1
    lat2, lon2 = point2

    lats = np.linspace(lat1, lat2, num_points)
    lons = np.linspace(lon1, lon2, num_points)

    return np.array([lats, lons]).T


if __name__ == "__main__":
    latitude = 35.5
    longitude = 113.5
    elevation = get_elevation(latitude, longitude, "../dataset/srtm")
    print(elevation)