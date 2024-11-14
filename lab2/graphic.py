from geopy.geocoders import Nominatim
from gmplot import gmplot
import time


def get_coordinates(city_names):
    """ Finds coordinates of the cities."""
    geolocator = Nominatim(user_agent="city_mapper")
    coordinates = []

    for city in city_names:
        location = geolocator.geocode(city + ", Poland")
        if location:
            coordinates.append((location.latitude, location.longitude))
        else:
            print(f"Warning: Could not find coordinates for {city}")
            coordinates.append((None, None))
        time.sleep(1)

    return coordinates


def graphic(city_names, solution_path):
    """Visualises the algorithm solution on the map."""
    ordered_city_names = [city_names[i] for i in solution_path]

    coordinates = get_coordinates(ordered_city_names)
    coordinates = [(lat, lon) for lat, lon in coordinates if lat is not None and lon is not None]

    if not coordinates:
        print("Error: No valid coordinates found for the route.")
        return

    gmap = gmplot.GoogleMapPlotter(coordinates[0][0], coordinates[0][1], 7)

    latitudes, longitudes = zip(*coordinates)
    gmap.plot(latitudes, longitudes, 'yellow', edge_width=2)
    gmap.scatter(latitudes, longitudes, color='red', size=40, marker=False)
    gmap.marker(latitudes[0], longitudes[0], color='blue')  # start flag
    gmap.marker(latitudes[-1], longitudes[-1], color='green')  # finish flag

    gmap.draw("route_map.html")
    print("Map has been saved as route_map.html")
