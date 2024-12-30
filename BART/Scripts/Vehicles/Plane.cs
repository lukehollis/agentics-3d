using UnityEngine;

public class Plane : BaseVehicle
{
    public AirRoute airRoute;
    private float distanceTraveled = 0f;

    protected override BasePath GetPath() => airRoute;

    public override void UpdatePosition(float deltaTime)
    {
        if (airRoute == null || isPaused) return;

        distanceTraveled += speed * deltaTime;
        Vector3 targetPosition = airRoute.GetPositionAtDistance(distanceTraveled);
        Vector3 lookAheadPos = airRoute.GetPositionAtDistance(distanceTraveled + 1f);
        
        transform.position = Vector3.MoveTowards(
            transform.position, 
            targetPosition, 
            speed * deltaTime
        );
        
        if ((lookAheadPos - targetPosition).sqrMagnitude > 0.001f)
        {
            Quaternion targetRotation = Quaternion.LookRotation(lookAheadPos - targetPosition);
            transform.rotation = Quaternion.Lerp(transform.rotation, targetRotation, rotationSpeed * deltaTime);
        }
    }
} 