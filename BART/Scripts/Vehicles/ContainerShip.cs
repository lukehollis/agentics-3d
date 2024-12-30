using UnityEngine;

public class ContainerShip : BaseVehicle
{
    public WaterRoute waterRoute;
    private float distanceTraveled = 0f;
    
    // Container ship specific properties
    public float length = 300f;     // Length of the ship in meters
    public float beam = 40f;        // Width of the ship in meters
    public float draft = 12f;       // Depth below waterline

    protected override BasePath GetPath() => waterRoute;

    protected override void Start()
    {
        base.Start();
        // Container ships move quite slowly (typical speed ~20 knots = ~10 m/s)
        speed = 10f;
        rotationSpeed = 0.5f; // Very slow rotation for large ships
    }

    public override void UpdatePosition(float deltaTime)
    {
        if (waterRoute == null || isPaused) return;

        distanceTraveled += speed * deltaTime;
        Vector3 targetPosition = waterRoute.GetPositionAtDistance(distanceTraveled);
        Vector3 lookAheadPos = waterRoute.GetPositionAtDistance(distanceTraveled + length/2); // Look ahead by half ship length
        
        transform.position = targetPosition;
        
        if ((lookAheadPos - targetPosition).sqrMagnitude > 0.001f)
        {
            Quaternion targetRotation = Quaternion.LookRotation(lookAheadPos - targetPosition);
            transform.rotation = Quaternion.Lerp(
                transform.rotation, 
                targetRotation, 
                rotationSpeed * deltaTime
            );
        }
    }
} 