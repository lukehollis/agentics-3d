using UnityEngine;

public abstract class BaseStation : MonoBehaviour
{
    public string stationId;
    public string stationName;
    public Vector2 coordinates; // Longitude, Latitude
    
    // For visualization in editor
    public float visualRadius = 5f;
    public Color stationColor = Color.white;
    
    protected virtual void OnDrawGizmos()
    {
        Gizmos.color = stationColor;
        Gizmos.DrawWireSphere(transform.position, visualRadius);
    }
} 