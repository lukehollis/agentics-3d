using UnityEngine;
using System.Collections.Generic;
using System.Linq;


public class TaskWaypoints : MonoBehaviour
{
    public static TaskWaypoints Instance { get; private set; }

    private List<Waypoint> waypoints = new List<Waypoint>();

    private void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
        }
        else
        {
            Destroy(gameObject);
            return;
        }

        CollectWaypoints();
    }

    private void CollectWaypoints()
    {
        Transform[] childTransforms = GetComponentsInChildren<Transform>();

        foreach (Transform childTransform in childTransforms)
        {
            if (childTransform != transform)
            {
                Waypoint waypoint = new Waypoint
                {
                    Name = childTransform.name,
                    Location = childTransform.position
                };

                waypoints.Add(waypoint);
            }
        }
    }

    public Vector3 GetWaypointLocation(string waypointName)
    {
        Waypoint waypoint = waypoints.Find(w => w.Name == waypointName);

        if (waypoint != null)
        {
            return waypoint.Location;
        }

        Debug.LogWarning($"Waypoint '{waypointName}' not found.");
        return Vector3.zero;
    }

    public List<string> GetNames()
    {
        return waypoints.Select(w => w.Name).ToList();
    }


}

public class Waypoint
{
    public string Name { get; set; }
    public Vector3 Location { get; set; }
}