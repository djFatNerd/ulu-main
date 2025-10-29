"""Dataset of major world city centers with approximate radii.

This script provides a structured list of 100 major global cities, including the
city name, latitude, longitude, and an indicative radius in kilometers that
roughly captures the central urban area. When executed as a script it prints the
collection in JSON format for convenient downstream consumption.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import List


@dataclass(frozen=True)
class CityBoundary:
    """Represents a city's approximate center point and coverage radius."""

    name: str
    latitude: float
    longitude: float
    radius_km: float


CITY_BOUNDARIES: List[CityBoundary] = [
    CityBoundary("New York City, USA", 40.7128, -74.006, 30.0),
    CityBoundary("Los Angeles, USA", 34.0522, -118.2437, 35.0),
    CityBoundary("Chicago, USA", 41.8781, -87.6298, 28.0),
    CityBoundary("Houston, USA", 29.7604, -95.3698, 35.0),
    CityBoundary("Phoenix, USA", 33.4484, -112.074, 30.0),
    CityBoundary("Philadelphia, USA", 39.9526, -75.1652, 25.0),
    CityBoundary("San Antonio, USA", 29.4241, -98.4936, 28.0),
    CityBoundary("San Diego, USA", 32.7157, -117.1611, 25.0),
    CityBoundary("Dallas, USA", 32.7767, -96.797, 28.0),
    CityBoundary("San Jose, USA", 37.3382, -121.8863, 22.0),
    CityBoundary("Austin, USA", 30.2672, -97.7431, 25.0),
    CityBoundary("Jacksonville, USA", 30.3322, -81.6557, 28.0),
    CityBoundary("San Francisco, USA", 37.7749, -122.4194, 18.0),
    CityBoundary("Columbus, USA", 39.9612, -82.9988, 25.0),
    CityBoundary("Fort Worth, USA", 32.7555, -97.3308, 26.0),
    CityBoundary("Indianapolis, USA", 39.7684, -86.1581, 24.0),
    CityBoundary("Charlotte, USA", 35.2271, -80.8431, 25.0),
    CityBoundary("Seattle, USA", 47.6062, -122.3321, 20.0),
    CityBoundary("Denver, USA", 39.7392, -104.9903, 22.0),
    CityBoundary("Boston, USA", 42.3601, -71.0589, 20.0),
    CityBoundary("Washington, D.C., USA", 38.9072, -77.0369, 20.0),
    CityBoundary("Miami, USA", 25.7617, -80.1918, 20.0),
    CityBoundary("Atlanta, USA", 33.749, -84.388, 24.0),
    CityBoundary("Detroit, USA", 42.3314, -83.0458, 22.0),
    CityBoundary("Toronto, Canada", 43.6532, -79.3832, 24.0),
    CityBoundary("Montreal, Canada", 45.5017, -73.5673, 22.0),
    CityBoundary("Vancouver, Canada", 49.2827, -123.1207, 20.0),
    CityBoundary("Mexico City, Mexico", 19.4326, -99.1332, 35.0),
    CityBoundary("Guadalajara, Mexico", 20.6597, -103.3496, 25.0),
    CityBoundary("Monterrey, Mexico", 25.6866, -100.3161, 25.0),
    CityBoundary("São Paulo, Brazil", -23.5505, -46.6333, 35.0),
    CityBoundary("Rio de Janeiro, Brazil", -22.9068, -43.1729, 28.0),
    CityBoundary("Brasília, Brazil", -15.8267, -47.9218, 25.0),
    CityBoundary("Buenos Aires, Argentina", -34.6037, -58.3816, 30.0),
    CityBoundary("Santiago, Chile", -33.4489, -70.6693, 28.0),
    CityBoundary("Lima, Peru", -12.0464, -77.0428, 28.0),
    CityBoundary("Bogotá, Colombia", 4.711, -74.0721, 28.0),
    CityBoundary("Medellín, Colombia", 6.2442, -75.5812, 22.0),
    CityBoundary("Quito, Ecuador", -0.1807, -78.4678, 20.0),
    CityBoundary("Caracas, Venezuela", 10.4806, -66.9036, 20.0),
    CityBoundary("London, United Kingdom", 51.5074, -0.1278, 25.0),
    CityBoundary("Manchester, United Kingdom", 53.4808, -2.2426, 20.0),
    CityBoundary("Birmingham, United Kingdom", 52.4862, -1.8904, 20.0),
    CityBoundary("Paris, France", 48.8566, 2.3522, 22.0),
    CityBoundary("Marseille, France", 43.2965, 5.3698, 20.0),
    CityBoundary("Lyon, France", 45.764, 4.8357, 18.0),
    CityBoundary("Berlin, Germany", 52.52, 13.405, 25.0),
    CityBoundary("Munich, Germany", 48.1351, 11.582, 20.0),
    CityBoundary("Hamburg, Germany", 53.5511, 9.9937, 22.0),
    CityBoundary("Frankfurt, Germany", 50.1109, 8.6821, 18.0),
    CityBoundary("Madrid, Spain", 40.4168, -3.7038, 25.0),
    CityBoundary("Barcelona, Spain", 41.3851, 2.1734, 22.0),
    CityBoundary("Rome, Italy", 41.9028, 12.4964, 25.0),
    CityBoundary("Milan, Italy", 45.4642, 9.19, 20.0),
    CityBoundary("Naples, Italy", 40.8518, 14.2681, 20.0),
    CityBoundary("Warsaw, Poland", 52.2297, 21.0122, 22.0),
    CityBoundary("Vienna, Austria", 48.2082, 16.3738, 20.0),
    CityBoundary("Zurich, Switzerland", 47.3769, 8.5417, 15.0),
    CityBoundary("Brussels, Belgium", 50.8503, 4.3517, 18.0),
    CityBoundary("Amsterdam, Netherlands", 52.3676, 4.9041, 18.0),
    CityBoundary("Copenhagen, Denmark", 55.6761, 12.5683, 18.0),
    CityBoundary("Stockholm, Sweden", 59.3293, 18.0686, 20.0),
    CityBoundary("Oslo, Norway", 59.9139, 10.7522, 18.0),
    CityBoundary("Helsinki, Finland", 60.1699, 24.9384, 18.0),
    CityBoundary("Moscow, Russia", 55.7558, 37.6173, 30.0),
    CityBoundary("Saint Petersburg, Russia", 59.9311, 30.3609, 25.0),
    CityBoundary("Istanbul, Turkey", 41.0082, 28.9784, 28.0),
    CityBoundary("Athens, Greece", 37.9838, 23.7275, 20.0),
    CityBoundary("Cairo, Egypt", 30.0444, 31.2357, 30.0),
    CityBoundary("Johannesburg, South Africa", -26.2041, 28.0473, 25.0),
    CityBoundary("Cape Town, South Africa", -33.9249, 18.4241, 22.0),
    CityBoundary("Nairobi, Kenya", -1.2921, 36.8219, 22.0),
    CityBoundary("Lagos, Nigeria", 6.5244, 3.3792, 28.0),
    CityBoundary("Accra, Ghana", 5.6037, -0.187, 20.0),
    CityBoundary("Addis Ababa, Ethiopia", 8.9806, 38.7578, 22.0),
    CityBoundary("Casablanca, Morocco", 33.5731, -7.5898, 22.0),
    CityBoundary("Algiers, Algeria", 36.7538, 3.0588, 22.0),
    CityBoundary("Tunis, Tunisia", 36.8065, 10.1815, 18.0),
    CityBoundary("Dubai, United Arab Emirates", 25.2048, 55.2708, 25.0),
    CityBoundary("Abu Dhabi, United Arab Emirates", 24.4539, 54.3773, 22.0),
    CityBoundary("Riyadh, Saudi Arabia", 24.7136, 46.6753, 28.0),
    CityBoundary("Jeddah, Saudi Arabia", 21.4858, 39.1925, 25.0),
    CityBoundary("Doha, Qatar", 25.2854, 51.531, 18.0),
    CityBoundary("Kuwait City, Kuwait", 29.3759, 47.9774, 18.0),
    CityBoundary("Muscat, Oman", 23.5859, 58.4059, 20.0),
    CityBoundary("Tehran, Iran", 35.6892, 51.389, 28.0),
    CityBoundary("Baghdad, Iraq", 33.3128, 44.3615, 25.0),
    CityBoundary("Karachi, Pakistan", 24.8607, 67.0011, 30.0),
    CityBoundary("Lahore, Pakistan", 31.5204, 74.3587, 25.0),
    CityBoundary("Delhi, India", 28.7041, 77.1025, 30.0),
    CityBoundary("Mumbai, India", 19.076, 72.8777, 28.0),
    CityBoundary("Bengaluru, India", 12.9716, 77.5946, 25.0),
    CityBoundary("Kolkata, India", 22.5726, 88.3639, 25.0),
    CityBoundary("Chennai, India", 13.0827, 80.2707, 24.0),
    CityBoundary("Dhaka, Bangladesh", 23.8103, 90.4125, 28.0),
    CityBoundary("Colombo, Sri Lanka", 6.9271, 79.8612, 20.0),
    CityBoundary("Bangkok, Thailand", 13.7563, 100.5018, 28.0),
    CityBoundary("Kuala Lumpur, Malaysia", 3.139, 101.6869, 22.0),
    CityBoundary("Singapore, Singapore", 1.3521, 103.8198, 18.0),
    CityBoundary("Jakarta, Indonesia", -6.2088, 106.8456, 30.0),
]

# Ensure the dataset maintains the expected coverage.
assert len(CITY_BOUNDARIES) == 100, "The city boundary dataset must include 100 entries."


def export_city_boundaries() -> List[dict]:
    """Return the city boundaries as a list of dictionaries."""

    return [asdict(city) for city in CITY_BOUNDARIES]


def main() -> None:
    """Print the city boundary collection as formatted JSON."""

    print(json.dumps(export_city_boundaries(), indent=2))


if __name__ == "__main__":
    main()
