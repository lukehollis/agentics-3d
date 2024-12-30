using UnityEngine;
using System.Collections.Generic;

public class VehicleManager : MonoBehaviour
{
    private static VehicleManager instance;
    public static VehicleManager Instance
    {
        get
        {
            if (instance == null)
            {
                var go = new GameObject("VehicleManager");
                instance = go.AddComponent<VehicleManager>();
                DontDestroyOnLoad(go);
            }
            return instance;
        }
    }

    private Dictionary<Road, List<BaseVehicle>> vehiclesOnRoad = new Dictionary<Road, List<BaseVehicle>>();
    private Dictionary<Road, float> lastSpawnDistances = new Dictionary<Road, float>();
    private const float MIN_VEHICLE_SPACING = 20f; // Minimum distance between vehicles
    private const float MAX_VEHICLE_SPACING = 50f; // Maximum distance between vehicles

    public void RegisterVehicle(BaseVehicle vehicle, Road road)
    {
        if (!vehiclesOnRoad.ContainsKey(road))
        {
            vehiclesOnRoad[road] = new List<BaseVehicle>();
        }
        vehiclesOnRoad[road].Add(vehicle);
    }

    public void UnregisterVehicle(BaseVehicle vehicle, Road road)
    {
        if (vehiclesOnRoad.ContainsKey(road))
        {
            vehiclesOnRoad[road].Remove(vehicle);
        }
    }

    public List<BaseVehicle> GetVehiclesOnRoad(Road road)
    {
        if (vehiclesOnRoad.ContainsKey(road))
        {
            return vehiclesOnRoad[road];
        }
        return new List<BaseVehicle>();
    }

    public float GetRandomStartingDistance(Road road)
    {
        if (!lastSpawnDistances.ContainsKey(road))
        {
            lastSpawnDistances[road] = 0f;
        }

        float randomSpacing = Random.Range(MIN_VEHICLE_SPACING, MAX_VEHICLE_SPACING);
        float startingDistance = lastSpawnDistances[road] + randomSpacing;
        lastSpawnDistances[road] = startingDistance;
        
        return startingDistance;
    }
} 