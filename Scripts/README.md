# BART Digital Twin

The BART Digital Twin is a simulation of the Bay Area Rapid Transit system. This Unity repo contains the 3d environment that connects to a webservices backend with various historical and realtime data sources. 

The simulation focuses on four key areas: train control modernization to reduce delays and optimize routing, passenger flow management to handle station crowding, power grid management for improved energy efficiency, and fleet management for optimized maintenance. 

By creating an accurate virtual replica of BART's entire system, the project aims to improve service reliability while reducing operational costs through predictive analytics and real-time optimization.

## Getting Started

To get started, clone the repo and open the project in Unity. The project is built with Unity 2021.3.29f1.

Configure the Cesium ion API key in the CesiumManager script. There may be other geospatial data sources that need to be configured as well.

## Networking

The characters in the BART simulation are controlled by the webservices backend via the NetworkingController. Each NPC should have a default day plan and set of actions to be able to run offline, but can be more realistically controlled by the webservices backend when available.

## Data Sources

* Cesium Google Maps Photorealistic Tiles Terrain is used for the base terrain. 
* OSM Streets data is used for the base map. 
* Census data is used for the population density and other demographic data. 
* BART station and train data is used for the train stations and train locations. 
* OSGS Weather data is used for historical weather data
* Crime data is used for the crime locations and types. 