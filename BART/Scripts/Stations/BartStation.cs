using UnityEngine;

public class BartStation : BaseStation
{
    public string[] servingLines; // Array of line colors/names this station serves
    public bool isTransferStation; // If this station serves multiple lines
    
    [Header("Platform")]
    public Transform platform;          // Transform for the waiting area
    public float platformWidth = 3f;    // Width of the platform
    public float platformLength = 20f;  // Length of the platform

    [Header("Movement")]
    public bool allowRuntimeMovement = true;
    private bool isDragging = false;
    private Vector3 offset;

    private Vector3 lastPosition;

    private void Update()
    {
        // Check if position changed
        if (transform.position != lastPosition)
        {
            Debug.Log("Station moved");
            lastPosition = transform.position;
            NotifyCharactersOfStationMove();
        }
    }

    private void OnValidate()
    {
        // Create platform if it doesn't exist
        if (platform == null)
        {
            GameObject p = new GameObject("Platform");
            p.transform.parent = transform;
            platform = p.transform;
        }
    }

    private void OnTransformParentChanged()
    {
        NotifyCharactersOfStationMove();
    }

    private void OnMouseDown()
    {
        if (!allowRuntimeMovement) return;
        isDragging = true;
        offset = transform.position - GetMouseWorldPosition();
    }

    private void OnMouseDrag()
    {
        if (!isDragging) return;
        Vector3 newPosition = GetMouseWorldPosition() + offset;
        transform.position = new Vector3(newPosition.x, transform.position.y, newPosition.z);
        
        // Update platform position
        if (platform != null)
        {
            platform.position = transform.position;
        }

        NotifyCharactersOfStationMove();
    }

    private void OnMouseUp()
    {
        isDragging = false;
    }

    private Vector3 GetMouseWorldPosition()
    {
        Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
        UnityEngine.Plane groundPlane = new UnityEngine.Plane(Vector3.up, 0f);
        float distance;
        groundPlane.Raycast(ray, out distance);
        return ray.GetPoint(distance);
    }

    private void NotifyCharactersOfStationMove()
    {
        Debug.Log("Notifying characters of station move");

        var characters = FindObjectsOfType<Agentics.TransportationController>();
        foreach (var character in characters)
        {
            character.OnStationMoved(this);
        }
    }

    protected override void OnDrawGizmos()
    {
        base.OnDrawGizmos();
        
        if (isTransferStation)
        {
            Gizmos.DrawWireSphere(transform.position, visualRadius * 1.5f);
        }
        
        // Draw platform
        if (platform != null)
        {
            Gizmos.color = Color.yellow;
            Gizmos.DrawWireCube(platform.position, 
                new Vector3(platformWidth, 0.1f, platformLength));
        }
    }

    public Vector3 GetPlatformPosition()
    {
        if (platform != null)
        {
            return platform.position;
        }
        // If platform is null, return station position as fallback
        return transform.position;
    }
} 