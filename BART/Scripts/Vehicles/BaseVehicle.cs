using UnityEngine;

public abstract class BaseVehicle : MonoBehaviour, IVehicle
{
    public float speed = 25f;
    public float rotationSpeed = 10f;
    public float vehicleLength = 5f;
    public float safetyDistance = 5f;
    protected bool isPaused = false;
    protected float currentSpeed;

    protected virtual void Start()
    {
        currentSpeed = speed;
    }

    protected float CheckForwardDistance(BasePath path, float currentDistance, int lane)
    {
        var vehicles = VehicleManager.Instance.GetVehiclesOnRoad(path as Road);
        float minDistance = float.MaxValue;

        foreach (var vehicle in vehicles)
        {
            if (vehicle == this) continue;

            // Only check vehicles in the same lane
            if (vehicle is Car car && car.currentLane != lane) continue;
            if (vehicle is Bus bus && bus.currentLane != lane) continue;

            // Calculate actual world distance between vehicles
            float distance = Vector3.Distance(vehicle.transform.position, transform.position);
            
            // Only consider vehicles ahead of us (in the direction we're moving)
            Vector3 toOtherVehicle = vehicle.transform.position - transform.position;
            if (Vector3.Dot(transform.forward, toOtherVehicle) > 0 && distance < minDistance)
            {
                minDistance = distance;
            }
        }

        return minDistance;
    }

    protected float AdjustSpeed(float forwardDistance)
    {
        if (forwardDistance < safetyDistance)
        {
            // Use quadratic easing instead of linear for more natural deceleration
            float t = forwardDistance / safetyDistance;
            return speed * (t * t);
        }
        return speed;
    }

    public virtual void UpdatePosition(float deltaTime)
    {
        if (isPaused) return;
        // Implement in derived classes
    }

    public virtual void Pause() => isPaused = true;
    public virtual void Resume() => isPaused = false;

    protected virtual void Update()
    {
        UpdatePosition(Time.deltaTime);
    }

    protected abstract BasePath GetPath(); // Each vehicle type must specify its path
} 