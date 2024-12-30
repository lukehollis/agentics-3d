using UnityEngine;

public class Car : BaseVehicle
{
    public Road road;
    public int currentLane = 0;
    
    private bool isActive = false;
    private Vector3 startPosition;
    private Vector3 targetPosition;
    private float distanceTraveled = 0f;
    private const float ARRIVAL_THRESHOLD = 2f;

    public void StartJourneyToPositions(Vector3 start, Vector3 target)
    {
        startPosition = start;
        targetPosition = target;
        
        // Find the initial distance along the road that corresponds to our start position
        distanceTraveled = road.GetDistanceAlongPath(start);
        transform.position = road.GetPositionAtDistance(distanceTraveled, currentLane);
        
        isActive = true;
        currentSpeed = speed;
        
        // Register with vehicle manager
        VehicleManager.Instance.RegisterVehicle(this, road);
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
    }

    protected void OnDestroy()
    {
        VehicleManager.Instance.UnregisterVehicle(this, road);
    }

    protected override BasePath GetPath() => road;

    public override void UpdatePosition(float deltaTime)
    {
        if (road == null || isPaused || !isActive) return;

        // Check if we've reached the target position
        if (Vector3.Distance(transform.position, targetPosition) <= ARRIVAL_THRESHOLD)
        {
            Debug.Log("Reached target position");
            EndJourney();
            return;
        }

        // Check distance to vehicle ahead
        float forwardDistance = CheckForwardDistance(road, distanceTraveled, currentLane);
        
        // Adjust speed based on forward distance
        currentSpeed = AdjustSpeed(forwardDistance);

        // Move along the road
        distanceTraveled += currentSpeed * deltaTime;
        Vector3 currentTargetPos = road.GetPositionAtDistance(distanceTraveled, currentLane);
        Vector3 lookAheadPos = road.GetPositionAtDistance(distanceTraveled + 1f, currentLane);
        
        transform.position = currentTargetPos;
        
        if ((lookAheadPos - currentTargetPos).sqrMagnitude > 0.001f)
        {
            Quaternion targetRotation = Quaternion.LookRotation(lookAheadPos - currentTargetPos);
            transform.rotation = Quaternion.Lerp(transform.rotation, targetRotation, rotationSpeed * deltaTime);
        }
    }

    public bool HasReachedDestination()
    {
        return !isActive || Vector3.Distance(transform.position, targetPosition) <= ARRIVAL_THRESHOLD;
    }
} 