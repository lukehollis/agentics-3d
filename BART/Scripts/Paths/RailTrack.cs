using UnityEngine;
using System.Collections.Generic;

public class RailTrack : BasePath
{
    public float gaugeWidth = 1.435f; // Standard gauge in meters
    public Color lineColor = Color.white; // Color for visualization
    
    private void OnValidate()
    {
        pathColor = lineColor; // Keep colors in sync when changed in inspector
    }
    
    public override void Awake()
    {
        pathColor = lineColor; // Ensure color is set before base.Awake()
        base.Awake();
            
        if (pathRenderer != null)
        {
            pathRenderer.material.color = lineColor;
        }
    }

}