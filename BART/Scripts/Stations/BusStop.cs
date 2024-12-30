using UnityEngine;

public class BusStop : BaseStation
{
    public string routeId;
    public string stopId;
    public string direction;
    
    protected override void OnDrawGizmos()
    {
        // Use a different color for bus stops to distinguish them from BART stations
        stationColor = new Color(0.2f, 0.8f, 0.2f); // Green color
        base.OnDrawGizmos();
        
        // Draw a small square to distinguish bus stops from BART stations
        Gizmos.color = stationColor;
        Vector3 size = new Vector3(visualRadius, visualRadius, visualRadius) * 0.5f;
        Gizmos.DrawWireCube(transform.position, size);
    }
} 