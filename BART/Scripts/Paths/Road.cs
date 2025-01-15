using UnityEngine;
using System.Collections.Generic;


public class Road : BasePath
{
    public float laneWidth = 4f; // Standard lane width in meters
    public int numLanes = 2;     // Number of lanes in each direction
    
    private LineRenderer[] laneRenderers;
    private LineRenderer[] laneDividers;
    
    public override void Awake()
    {
        base.Awake();  // Call BasePath's Awake first
        
        // Create line renderers for each lane
        laneRenderers = new LineRenderer[numLanes];
        // Create dividers between lanes (numLanes - 1 dividers needed)
        laneDividers = new LineRenderer[numLanes - 1];
        
        // Setup lane renderers
        for (int i = 0; i < numLanes; i++)
        {
            GameObject laneObj = new GameObject($"Lane_{i}");
            laneObj.transform.parent = transform;
            
            LineRenderer renderer = laneObj.AddComponent<LineRenderer>();
            renderer.startWidth = 0.2f;
            renderer.endWidth = 0.2f;
            renderer.material = new Material(Shader.Find("Unlit/Color"));
            renderer.material.color = Color.white;
            
            // Make sure the line is always visible
            renderer.receiveShadows = false;
            renderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            renderer.allowOcclusionWhenDynamic = false;
            renderer.material.renderQueue = 4000;
            
            laneRenderers[i] = renderer;
        }
        
        // Setup lane dividers
        for (int i = 0; i < numLanes - 1; i++)
        {
            GameObject dividerObj = new GameObject($"LaneDivider_{i}");
            dividerObj.transform.parent = transform;
            
            LineRenderer renderer = dividerObj.AddComponent<LineRenderer>();
            renderer.startWidth = 0.1f; // Thinner than the lanes
            renderer.endWidth = 0.1f;
            renderer.material = new Material(Shader.Find("Unlit/Color"));
            renderer.material.color = Color.white; // White divider lines
            
            // Make sure the line is always visible
            renderer.receiveShadows = false;
            renderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            renderer.allowOcclusionWhenDynamic = false;
            renderer.material.renderQueue = 4000;
            
            laneDividers[i] = renderer;
        }
        
        UpdateLaneRenderers();
    }
    
    private void UpdateLaneRenderers()
    {
        if (nodes.Count < 2) return;
        
        // Update lane renderers
        for (int lane = 0; lane < numLanes; lane++)
        {
            LineRenderer renderer = laneRenderers[lane];
            List<Vector3> points = new List<Vector3>();
            
            for (int i = 0; i < nodes.Count - 1; i++)
            {
                if (nodes[i] != null && nodes[i + 1] != null)
                {
                    Vector3 direction = (nodes[i + 1].position - nodes[i].position).normalized;
                    Vector3 right = Vector3.Cross(Vector3.up, direction);
                    float offset = (lane - (numLanes - 1) * 0.5f) * laneWidth;
                    Vector3 laneOffset = right * offset;
                    
                    points.Add(nodes[i].position + laneOffset);
                    points.Add(nodes[i + 1].position + laneOffset);
                }
            }
            
            renderer.positionCount = points.Count;
            renderer.SetPositions(points.ToArray());
        }
        
        // Update lane dividers
        for (int divider = 0; divider < numLanes - 1; divider++)
        {
            LineRenderer renderer = laneDividers[divider];
            List<Vector3> points = new List<Vector3>();
            
            for (int i = 0; i < nodes.Count - 1; i++)
            {
                if (nodes[i] != null && nodes[i + 1] != null)
                {
                    Vector3 direction = (nodes[i + 1].position - nodes[i].position).normalized;
                    Vector3 right = Vector3.Cross(Vector3.up, direction);
                    float offset = (divider + 1 - (numLanes - 1) * 0.5f) * laneWidth;
                    Vector3 dividerOffset = right * offset;
                    
                    points.Add(nodes[i].position + dividerOffset);
                    points.Add(nodes[i + 1].position + dividerOffset);
                }
            }
            
            renderer.positionCount = points.Count;
            renderer.SetPositions(points.ToArray());
        }
    }
    
    private void OnValidate()
    {
        if (laneRenderers != null)
        {
            UpdateLaneRenderers();
        }
    }
    

    public Vector3 GetNearestPoint(Vector3 position)
    {
        if (nodes.Count < 2) return transform.position;
        
        Vector3 nearestPoint = nodes[0].position;
        float nearestDistance = float.MaxValue;

        // Check each road segment
        for (int i = 0; i < nodes.Count - 1; i++)
        {
            Vector3 start = nodes[i].position;
            Vector3 end = nodes[i + 1].position;
            Vector3 point = GetNearestPointOnSegment(position, start, end);
            
            float distance = Vector3.Distance(position, point);
            if (distance < nearestDistance)
            {
                nearestDistance = distance;
                nearestPoint = point;
            }
        }

        return nearestPoint;
    }

    private Vector3 GetNearestPointOnSegment(Vector3 point, Vector3 start, Vector3 end)
    {
        Vector3 segment = end - start;
        Vector3 pointVector = point - start;
        
        float segmentLength = segment.magnitude;
        Vector3 segmentDirection = segment / segmentLength;
        
        float projection = Vector3.Dot(pointVector, segmentDirection);
        
        if (projection <= 0)
            return start;
        if (projection >= segmentLength)
            return end;
            
        return start + segmentDirection * projection;
    }

    public Vector3 GetPositionAtDistance(float distance, int lane)
    {
        Vector3 basePosition = base.GetPositionAtDistance(distance);
        
        // Get direction at this point to calculate lane offset
        Vector3 direction = GetDirectionAtDistance(distance);
        Vector3 right = Vector3.Cross(Vector3.up, direction);
        
        // Calculate lane offset from center
        float offset = (lane - (numLanes - 1) * 0.5f) * laneWidth;
        return basePosition + right * offset;
    }

    private Vector3 GetDirectionAtDistance(float distance)
    {
        // Find segment that contains this distance
        float accumulatedDistance = 0f;
        
        for (int i = 0; i < nodes.Count - 1; i++)
        {
            float segmentLength = Vector3.Distance(nodes[i].position, nodes[i + 1].position);
            
            if (accumulatedDistance + segmentLength >= distance)
            {
                // Found the right segment
                return (nodes[i + 1].position - nodes[i].position).normalized;
            }
            
            accumulatedDistance += segmentLength;
        }
        
        // If we're past the end, use direction of last segment
        return (nodes[nodes.Count - 1].position - nodes[nodes.Count - 2].position).normalized;
    }

    public float GetDistanceAlongPath(Vector3 worldPosition)
    {
        if (nodes.Count < 2) return 0f;
        
        float nearestDistance = float.MaxValue;
        float distanceAlongPath = 0f;
        float totalDistance = 0f;
        
        // Check each road segment
        for (int i = 0; i < nodes.Count - 1; i++)
        {
            Vector3 start = nodes[i].position;
            Vector3 end = nodes[i + 1].position;
            Vector3 segment = end - start;
            float segmentLength = segment.magnitude;
            
            Vector3 pointVector = worldPosition - start;
            float projection = Vector3.Dot(pointVector, segment.normalized);
            Vector3 projectedPoint = start + segment.normalized * Mathf.Clamp(projection, 0f, segmentLength);
            
            float distanceToPoint = Vector3.Distance(worldPosition, projectedPoint);
            
            if (distanceToPoint < nearestDistance)
            {
                nearestDistance = distanceToPoint;
                distanceAlongPath = totalDistance + Mathf.Clamp(projection, 0f, segmentLength);
            }
            
            totalDistance += segmentLength;
        }
        
        return distanceAlongPath;
    }
}