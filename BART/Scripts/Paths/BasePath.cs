using UnityEngine;
using System.Collections.Generic;


public abstract class BasePath : MonoBehaviour
{
    public List<Transform> nodes = new List<Transform>();
    public bool isLoop = true;
    public Color pathColor = Color.yellow;
    public float nodeSize = 1f;
    
    protected LineRenderer pathRenderer;
    protected GameObject[] nodeVisuals;

    public virtual void Awake()
    {
        // Setup main path renderer
        GameObject pathObj = new GameObject("PathLine");
        pathObj.transform.parent = transform;
        
        pathRenderer = pathObj.AddComponent<LineRenderer>();
        pathRenderer.startWidth = 0.2f;
        pathRenderer.endWidth = 0.2f;
        Material pathMaterial = new Material(Shader.Find("Unlit/Color"));
        pathMaterial.color = pathColor;
        pathRenderer.sharedMaterial = pathMaterial;
        
        // Make sure the line is always visible
        pathRenderer.receiveShadows = false;
        pathRenderer.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        pathRenderer.allowOcclusionWhenDynamic = false;
        pathRenderer.sharedMaterial.renderQueue = 4000;
        
        CreateNodeVisuals();
        UpdatePathVisualization();
    }

    private void CreateNodeVisuals()
    {
        if (nodeVisuals != null)
        {
            foreach (var visual in nodeVisuals)
            {
                if (visual != null) DestroyImmediate(visual);
            }
        }
        
        nodeVisuals = new GameObject[nodes.Count];
        for (int i = 0; i < nodes.Count; i++)
        {
            GameObject nodeObj = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            nodeObj.name = $"NodeVisual_{i}";
            nodeObj.transform.parent = transform;
            nodeObj.transform.localScale = Vector3.one * nodeSize;
            
            Material nodeMaterial = new Material(Shader.Find("Unlit/Color"));
            nodeMaterial.color = pathColor;
            nodeMaterial.renderQueue = 4000;
            nodeObj.GetComponent<MeshRenderer>().sharedMaterial = nodeMaterial;
            
            DestroyImmediate(nodeObj.GetComponent<SphereCollider>());
            nodeVisuals[i] = nodeObj;
        }
    }

    protected virtual void UpdatePathVisualization()
    {
        if (nodes.Count < 2) return;
        
        List<Vector3> points = new List<Vector3>();
        
        for (int i = 0; i < nodes.Count; i++)
        {
            if (nodes[i] != null)
            {
                points.Add(nodes[i].position);
                if (nodeVisuals != null && i < nodeVisuals.Length && nodeVisuals[i] != null)
                {
                    nodeVisuals[i].transform.position = nodes[i].position;
                }
            }
        }
        
        if (isLoop && nodes.Count > 0 && nodes[0] != null)
        {
            points.Add(nodes[0].position);
        }
        
        pathRenderer.positionCount = points.Count;
        pathRenderer.SetPositions(points.ToArray());
    }

    private void OnValidate()
    {
        if (!Application.isPlaying) return;
        if (pathRenderer == null) Awake();
        CreateNodeVisuals();
        UpdatePathVisualization();
    }

    protected virtual void Update()
    {
        UpdatePathVisualization();
    }

    // Keep all existing methods from the original BasePath
    public Vector3 GetDirectionAtDistance(float distance)
    {
        if (nodes.Count < 2) return transform.forward;
        
        Vector3 currentPos = GetPositionAtDistance(distance);
        Vector3 nextPos = GetPositionAtDistance(distance + 0.1f);
        
        return (nextPos - currentPos).normalized;
    }

    public Vector3 GetPositionAtDistance(float distance)
    {
        if (nodes.Count < 2) return transform.position;
        
        float totalDistance = 0f;
        float[] distances = new float[nodes.Count - 1];
        
        for (int i = 0; i < nodes.Count - 1; i++)
        {
            distances[i] = Vector3.Distance(nodes[i].position, nodes[i + 1].position);
            totalDistance += distances[i];
        }
        
        if (isLoop)
        {
            distances[distances.Length - 1] = Vector3.Distance(
                nodes[nodes.Count - 1].position, 
                nodes[0].position
            );
            totalDistance += distances[distances.Length - 1];
        }
        
        distance = distance % totalDistance;
        
        float currentDistance = 0f;
        for (int i = 0; i < distances.Length; i++)
        {
            if (currentDistance + distances[i] >= distance)
            {
                float t = (distance - currentDistance) / distances[i];
                return Vector3.Lerp(
                    nodes[i].position,
                    nodes[(i + 1) % nodes.Count].position,
                    t
                );
            }
            currentDistance += distances[i];
        }
        
        return nodes[0].position;
    }
} 