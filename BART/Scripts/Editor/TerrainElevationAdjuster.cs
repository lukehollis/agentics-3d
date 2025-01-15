using UnityEngine;
using UnityEditor;
using System.Collections.Generic;

public class TerrainElevationAdjuster : EditorWindow
{
    private GameObject targetParent;
    private GameObject environmentParent;
    private float heightOffset = 2f; // meters above ground
    private float maxRaycastDistance = 1000f;
    private bool adjustOnlySelected = false;
    private bool previewChanges = true;
    private List<Vector3> originalPositions = new List<Vector3>();
    private List<Transform> adjustableObjects = new List<Transform>();

    [MenuItem("Tools/BART/Terrain Elevation Adjuster")]
    public static void ShowWindow()
    {
        GetWindow<TerrainElevationAdjuster>("Terrain Elevation Adjuster");
    }

    private void OnEnable()
    {
        SceneView.duringSceneGui += OnSceneGUI;
    }

    private void OnDisable()
    {
        SceneView.duringSceneGui -= OnSceneGUI;
        RestoreOriginalPositions();
    }

    private void OnGUI()
    {
        EditorGUILayout.LabelField("Terrain Elevation Adjuster", EditorStyles.boldLabel);
        
        using (new EditorGUILayout.VerticalScope("box"))
        {
            targetParent = (GameObject)EditorGUILayout.ObjectField(
                "Parent Object", 
                targetParent, 
                typeof(GameObject), 
                true
            );

            environmentParent = (GameObject)EditorGUILayout.ObjectField(
                "Environment Parent (3D Tiles)", 
                environmentParent, 
                typeof(GameObject), 
                true
            );

            heightOffset = EditorGUILayout.FloatField("Height Offset (m)", heightOffset);
            maxRaycastDistance = EditorGUILayout.FloatField("Max Raycast Distance", maxRaycastDistance);
            adjustOnlySelected = EditorGUILayout.Toggle("Adjust Only Selected", adjustOnlySelected);
            previewChanges = EditorGUILayout.Toggle("Preview Changes", previewChanges);
        }

        EditorGUILayout.Space();

        using (new EditorGUILayout.VerticalScope("box"))
        {
            if (GUILayout.Button("Scan Objects"))
            {
                ScanObjects();
            }

            EditorGUI.BeginDisabledGroup(adjustableObjects.Count == 0);
            if (GUILayout.Button("Apply Elevation Adjustments"))
            {
                ApplyElevationAdjustments();
            }
            EditorGUI.EndDisabledGroup();
        }

        if (adjustableObjects.Count > 0)
        {
            EditorGUILayout.HelpBox(
                $"Found {adjustableObjects.Count} objects to adjust.", 
                MessageType.Info
            );
        }
    }

    private void TagEnvironmentObjects()
    {
        if (environmentParent == null)
        {
            EditorUtility.DisplayDialog("Error", "Please assign the Environment Parent object", "OK");
            return;
        }

        // Get all children including nested ones
        Transform[] allChildren = environmentParent.GetComponentsInChildren<Transform>();
        
        Undo.RecordObjects(allChildren, "Tag Environment Objects");
        
        foreach (Transform child in allChildren)
        {
            // Skip the parent itself and any objects that are children of targetParent
            if (child.gameObject != environmentParent && !child.IsChildOf(targetParent.transform))
            {
                child.gameObject.tag = "Environment";
            }
        }
    }

    private void ScanObjects()
    {
        if (targetParent == null)
        {
            EditorUtility.DisplayDialog("Error", "Please select a parent object", "OK");
            return;
        }

        // Tag environment objects first
        TagEnvironmentObjects();

        adjustableObjects.Clear();
        originalPositions.Clear();

        if (adjustOnlySelected)
        {
            foreach (GameObject obj in Selection.gameObjects)
            {
                if (obj.transform.IsChildOf(targetParent.transform))
                {
                    adjustableObjects.Add(obj.transform);
                    originalPositions.Add(obj.transform.position);
                }
            }
        }
        else
        {
            foreach (Transform child in targetParent.GetComponentsInChildren<Transform>())
            {
                if (child != targetParent.transform)
                {
                    adjustableObjects.Add(child);
                    originalPositions.Add(child.position);
                }
            }
        }

        if (previewChanges)
        {
            PreviewAdjustments();
        }
    }

    private void PreviewAdjustments()
    {
        for (int i = 0; i < adjustableObjects.Count; i++)
        {
            Transform obj = adjustableObjects[i];
            Vector3 rayStart = obj.position + Vector3.up * maxRaycastDistance / 2f;
            
            // Get all hits along the ray
            RaycastHit[] hits = Physics.RaycastAll(rayStart, Vector3.down, maxRaycastDistance);
            
            // Find the lowest (smallest y value) hit that's tagged Environment
            float lowestPoint = float.MaxValue;
            bool foundEnvironment = false;
            
            foreach (RaycastHit hit in hits)
            {
                // Skip if this is part of the object we're adjusting
                if (hit.transform.IsChildOf(obj))
                    continue;
                    
                if (hit.collider.CompareTag("Environment"))
                {
                    if (hit.point.y < lowestPoint)
                    {
                        lowestPoint = hit.point.y;
                        foundEnvironment = true;
                    }
                }
            }

            if (foundEnvironment)
            {
                Vector3 newPosition = obj.position;
                newPosition.y = lowestPoint + heightOffset;
                obj.position = newPosition;
            }
        }
        SceneView.RepaintAll();
    }

    private void RestoreOriginalPositions()
    {
        for (int i = 0; i < adjustableObjects.Count; i++)
        {
            if (adjustableObjects[i] != null)
            {
                adjustableObjects[i].position = originalPositions[i];
            }
        }
        SceneView.RepaintAll();
    }

    private void ApplyElevationAdjustments()
    {
        TagEnvironmentObjects();
        
        Undo.RecordObjects(adjustableObjects.ToArray(), "Adjust Terrain Elevations");
        
        foreach (Transform obj in adjustableObjects)
        {
            Vector3 rayStart = obj.position + Vector3.up * maxRaycastDistance / 2f;
            
            // Get all hits along the ray
            RaycastHit[] hits = Physics.RaycastAll(rayStart, Vector3.down, maxRaycastDistance);
            
            // Find the lowest (smallest y value) hit that's tagged Environment
            float lowestPoint = float.MaxValue;
            bool foundEnvironment = false;
            
            foreach (RaycastHit hit in hits)
            {
                // Skip if this is part of the object we're adjusting
                if (hit.transform.IsChildOf(obj))
                    continue;
                    
                if (hit.collider.CompareTag("Environment"))
                {
                    if (hit.point.y < lowestPoint)
                    {
                        lowestPoint = hit.point.y;
                        foundEnvironment = true;
                    }
                }
            }

            if (foundEnvironment)
            {
                Vector3 newPosition = obj.position;
                newPosition.y = lowestPoint + heightOffset;
                obj.position = newPosition;
            }
        }
        
        EditorUtility.DisplayDialog("Success", "Elevation adjustments applied!", "OK");
    }

    private void OnSceneGUI(SceneView sceneView)
    {
        if (!previewChanges || adjustableObjects.Count == 0) return;

        Handles.color = Color.green;
        foreach (Transform obj in adjustableObjects)
        {
            Handles.DrawWireCube(obj.position, Vector3.one * 0.5f);
            Handles.DrawLine(obj.position, obj.position + Vector3.down * maxRaycastDistance);
        }
    }
} 