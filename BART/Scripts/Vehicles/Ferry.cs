using UnityEngine;

public class Ferry : BaseVehicle
{
    public WaterRoute waterRoute;
    private float distanceTraveled = 0f;

    protected override BasePath GetPath() => waterRoute;

    public override void UpdatePosition(float deltaTime)
    {
        if (waterRoute == null || isPaused) return;

        distanceTraveled += speed * deltaTime;
        Vector3 targetPosition = waterRoute.GetPositionAtDistance(distanceTraveled);
        Vector3 lookAheadPos = waterRoute.GetPositionAtDistance(distanceTraveled + 1f);
        
        transform.position = targetPosition;
        
        if ((lookAheadPos - targetPosition).sqrMagnitude > 0.001f)
        {
            Quaternion targetRotation = Quaternion.LookRotation(lookAheadPos - targetPosition);
            transform.rotation = Quaternion.Lerp(transform.rotation, targetRotation, rotationSpeed * deltaTime);
        }
    }
} 