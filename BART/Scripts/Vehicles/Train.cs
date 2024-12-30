using UnityEngine;
using System.Collections.Generic;
using System.Linq;
using Agentics;

public class Train : BaseVehicle
{
    public RailTrack track;
    public List<TrainCar> cars = new List<TrainCar>();
    public List<BartStation> stationStops = new List<BartStation>();
    
    private float distanceTraveled = 0f;
    private List<float> carDistances = new List<float>();
    private bool isAtStation = false;
    private float stationWaitTime = 2f; // Time to wait at each station
    private float currentStationTimer = 0f;
    private Dictionary<BartStation, float> stationCooldowns = new Dictionary<BartStation, float>();
    private float stationCooldownDuration = 10f; // Time before train can stop at same station again
    private List<AgenticController> passengers = new List<AgenticController>();
    private BartStation currentStation = null;

    protected override void Start()
    {
        base.Start();
        speed = 35f; // ~126 km/h
        rotationSpeed = 2f; // Reduced rotation speed for smoother turns
        
        // Initialize distances for each car
        carDistances.Clear();
        for (int i = 0; i < cars.Count; i++)
        {
            // Each car starts further back based on car length
            float totalOffset = 0f;
            for (int j = 0; j < i; j++)
            {
                totalOffset += cars[j].carLength;
            }
            carDistances.Add(-totalOffset);
        }
    }

    public override void UpdatePosition(float deltaTime)
    {
        if (track == null || isPaused) return;

        // Update station cooldowns
        List<BartStation> finishedCooldowns = new List<BartStation>();
        List<KeyValuePair<BartStation, float>> cooldownsToUpdate = new List<KeyValuePair<BartStation, float>>();
        
        // First, collect all updates needed
        foreach (var kvp in stationCooldowns)
        {
            float newCooldown = kvp.Value - deltaTime;
            if (newCooldown <= 0)
            {
                finishedCooldowns.Add(kvp.Key);
            }
            else
            {
                cooldownsToUpdate.Add(new KeyValuePair<BartStation, float>(kvp.Key, newCooldown));
            }
        }
        
        // Then apply the updates
        foreach (var station in finishedCooldowns)
        {
            stationCooldowns.Remove(station);
        }
        
        foreach (var update in cooldownsToUpdate)
        {
            stationCooldowns[update.Key] = update.Value;
        }

        // Handle station stop logic
        if (isAtStation)
        {
            currentStationTimer += deltaTime;
            if (currentStationTimer >= stationWaitTime)
            {
                isAtStation = false;
                currentStationTimer = 0f;

                // Add cooldown for the current station
                if (currentStation != null)
                {
                    stationCooldowns[currentStation] = stationCooldownDuration;
                    currentStation = null;
                }
            }
            return;
        }
        
        // Check if we're approaching a station
        if (cars.Count > 0)
        {
            foreach (var station in stationStops)
            {
                // Skip if station is on cooldown
                if (stationCooldowns.ContainsKey(station)) continue;
                
                // Calculate distance to platform instead of station center
                Vector3 platformPosition = station.GetPlatformPosition();
                float distance = Vector3.Distance(cars[0].transform.position, platformPosition);
                
                if (distance < 10f && !isAtStation)
                {
                    Debug.Log($"Stopping at station {station.name}");
                    isAtStation = true;
                    currentStationTimer = 0f;
                    currentStation = station;
                    return;
                }
            }

            // Update train transform to follow lead car
            transform.position = cars[0].transform.position;
            transform.rotation = cars[0].transform.rotation;
        }

        // Move lead car
        distanceTraveled += speed * deltaTime;
        UpdateCarPosition(0, distanceTraveled);
        
        // Update following cars
        for (int i = 1; i < cars.Count; i++)
        {
            float carDistance = distanceTraveled + carDistances[i];
            UpdateCarPosition(i, carDistance);
        }
    }

    private void UpdateCarPosition(int carIndex, float distance)
    {
        TrainCar car = cars[carIndex];
        if (car == null) return;

        // Get current and look-ahead positions
        Vector3 targetPosition = track.GetPositionAtDistance(distance);
        Vector3 lookAheadPosition = track.GetPositionAtDistance(distance + 1f);
        Vector3 direction = (lookAheadPosition - targetPosition).normalized;
        
        // Update position and rotation
        car.transform.position = targetPosition;
        if (direction != Vector3.zero)
        {
            Quaternion targetRotation = Quaternion.LookRotation(direction);
            car.transform.rotation = Quaternion.Lerp(
                car.transform.rotation,
                targetRotation,
                rotationSpeed * Time.deltaTime
            );
        }
    }

    // Editor helper to setup train cars
    public void SetupCars()
    {
        cars.Clear();
        TrainCar[] foundCars = GetComponentsInChildren<TrainCar>();
        cars.AddRange(foundCars);
        
        // Initialize distances
        Start();
    }

    protected override BasePath GetPath() => track;

    // Add method to check if train serves these stations
    public bool ServesStations(BartStation station1, BartStation station2)
    {
        return stationStops.Contains(station1) && stationStops.Contains(station2) && 
               stationStops.IndexOf(station1) < stationStops.IndexOf(station2);
    }

    public BartStation GetCurrentStation() => currentStation;

}