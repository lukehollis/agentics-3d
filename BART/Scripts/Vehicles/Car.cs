using UnityEngine;

public class Car : BaseVehicle
{
    public Road road;
    public int currentLane = 0;
    public bool startAtRandomPositionOnRoad = false;
    
    private bool isActive = false;
    private Vector3 startPosition;
    private Vector3 targetPosition;
    private float distanceTraveled = 0f;
    private const float ARRIVAL_THRESHOLD = 2f;

    public void StartJourneyToPositions(Vector3 start, Vector3 target)
    {
        Debug.Log($"Car starting journey from {start} to {target}");
        startPosition = start;
        targetPosition = target;
        
        if (!startAtRandomPositionOnRoad)
        {
            distanceTraveled = road.GetDistanceAlongPath(start);
            transform.position = road.GetPositionAtDistance(distanceTraveled, currentLane);
            Debug.Log($"Starting at distance: {distanceTraveled}");
        }
        
        isActive = true;
        currentSpeed = speed;
        
        // Only register if we haven't already in Start
        if (!startAtRandomPositionOnRoad)
        {
            VehicleManager.Instance.RegisterVehicle(this, road);
        }
    }

    public void EndJourney()
    {
        isActive = false;
        currentSpeed = 0;
        VehicleManager.Instance.UnregisterVehicle(this, road);
    }

    protected override void Start()
    {
        base.Start();
        currentSpeed = speed;
        
        if (startAtRandomPositionOnRoad && road != null)
        {
            distanceTraveled = VehicleManager.Instance.GetRandomStartingDistance(road);
            transform.position = road.GetPositionAtDistance(distanceTraveled, currentLane);
            VehicleManager.Instance.RegisterVehicle(this, road);
        }
    }

    protected void OnDestroy()
    {
        VehicleManager.Instance.UnregisterVehicle(this, road);
    }

    protected override BasePath GetPath() => road;

    public override void UpdatePosition(float deltaTime)
    {
        if (road == null || isPaused) return;

        // Check distance to vehicle ahead
        float forwardDistance = CheckForwardDistance(road, distanceTraveled, currentLane);
        
        // Adjust speed based on forward distance
        currentSpeed = AdjustSpeed(forwardDistance);

        // Move the car 
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

    public bool HasReachedDestination()
    {
        // return !isActive || Vector3.Distance(transform.position, targetPosition) <= ARRIVAL_THRESHOLD;
        return false;
    }
} 