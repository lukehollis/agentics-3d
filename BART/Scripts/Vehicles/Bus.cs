using UnityEngine;

public class Bus : BaseVehicle
{
    public Road road;
    public int currentLane = 0;
    public float distanceTraveled = 0f;
    
    // Bus-specific properties
    public float stopDuration = 15f; // Time in seconds to wait at each stop
    public float stopDistance = 5f;  // Distance threshold to detect if we're at a stop
    private float currentStopTimer = 0f;
    private bool isAtStop = false;

    protected override BasePath GetPath() => road;

    protected override void Start()
    {
        base.Start();
        distanceTraveled = VehicleManager.Instance.GetRandomStartingDistance(road);
        VehicleManager.Instance.RegisterVehicle(this, road);
    }

    protected void OnDestroy()
    {
        VehicleManager.Instance.UnregisterVehicle(this, road);
    }

    public override void UpdatePosition(float deltaTime)
    {
        if (road == null || isPaused) return;

        // Handle bus stop logic
        if (isAtStop)
        {
            currentStopTimer += deltaTime;
            if (currentStopTimer >= stopDuration)
            {
                isAtStop = false;
                currentStopTimer = 0f;
            }
            return;
        }

        // Check distance to vehicle ahead
        float forwardDistance = CheckForwardDistance(road, distanceTraveled, currentLane);
        
        // Adjust speed based on forward distance
        currentSpeed = AdjustSpeed(forwardDistance);

        // Move the bus
        distanceTraveled += currentSpeed * deltaTime;
        Vector3 targetPosition = road.GetPositionAtDistance(distanceTraveled, currentLane);
        Vector3 lookAheadPos = road.GetPositionAtDistance(distanceTraveled + 1f, currentLane);
        
        transform.position = targetPosition;
        
        if ((lookAheadPos - targetPosition).sqrMagnitude > 0.001f)
        {
            Quaternion targetRotation = Quaternion.LookRotation(lookAheadPos - targetPosition);
            transform.rotation = Quaternion.Lerp(transform.rotation, targetRotation, rotationSpeed * deltaTime);
        }
    }

    // Method to trigger a bus stop
    public void TriggerBusStop()
    {
        if (!isAtStop)
        {
            isAtStop = true;
            currentStopTimer = 0f;
        }
    }
} 