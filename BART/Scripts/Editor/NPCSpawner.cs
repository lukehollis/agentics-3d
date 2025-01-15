using UnityEngine;
using UnityEditor;
using UnityEngine.AI;
using System.Collections.Generic;
using Unity.EditorCoroutines.Editor;
using Agentics;
using System.Linq;

public class NPCSpawner : EditorWindow
{
    private GameObject targetParent;
    private GameObject prefabToSpawn;
    private GameObject waypointsParent;
    private GameObject spawnArea;
    private int spawnCount = 1;
    private float maxRaycastDistance = 100f;

    [MenuItem("Tools/BART/NPC Spawner")]
    public static void ShowWindow()
    {
        GetWindow<NPCSpawner>("NPC Spawner");
    }

    private void OnGUI()
    {
        EditorGUILayout.LabelField("NPC Spawner", EditorStyles.boldLabel);
        
        using (new EditorGUILayout.VerticalScope("box"))
        {
            targetParent = (GameObject)EditorGUILayout.ObjectField(
                "Parent Object", 
                targetParent, 
                typeof(GameObject), 
                true
            );

            prefabToSpawn = (GameObject)EditorGUILayout.ObjectField(
                "Prefab to Spawn", 
                prefabToSpawn, 
                typeof(GameObject), 
                true
            );

            waypointsParent = (GameObject)EditorGUILayout.ObjectField(
                "Waypoints Parent", 
                waypointsParent, 
                typeof(GameObject), 
                true
            );

            spawnArea = (GameObject)EditorGUILayout.ObjectField(
                "Spawn Area Cube", 
                spawnArea, 
                typeof(GameObject), 
                true
            );

            spawnCount = EditorGUILayout.IntField("Number to Spawn", spawnCount);
        }

        EditorGUILayout.Space();

        using (new EditorGUILayout.VerticalScope("box"))
        {
            EditorGUI.BeginDisabledGroup(targetParent == null || prefabToSpawn == null || waypointsParent == null || spawnArea == null);
            if (GUILayout.Button("Spawn NPCs"))
            {
                SpawnPrefabs();
            }
            EditorGUI.EndDisabledGroup();
        }
    }

    private void SpawnPrefabs()
    {
        if (targetParent == null || prefabToSpawn == null || waypointsParent == null || spawnArea == null)
        {
            EditorUtility.DisplayDialog("Error", "Please assign Parent Object, Prefab, Waypoints Parent, and Spawn Area Cube", "OK");
            return;
        }

        spawnArea.SetActive(true);

        // Get all waypoint locations
        List<Transform> waypoints = new List<Transform>();
        foreach (Transform child in waypointsParent.transform)
        {
            waypoints.Add(child);
        }

        if (waypoints.Count == 0)
        {
            EditorUtility.DisplayDialog("Error", "No waypoints found in Waypoints Parent", "OK");
            return;
        }

        // Delete all existing children
        while (targetParent.transform.childCount > 0)
        {
            DestroyImmediate(targetParent.transform.GetChild(0).gameObject);
        }

        Undo.RecordObject(targetParent, "Spawn NPCs");
        Debug.Log($"Starting spawn of {spawnCount} NPCs...");

        EditorCoroutineUtility.StartCoroutine(SpawnPrefabsCoroutine(waypoints), this);
    }

    private System.Collections.IEnumerator SpawnPrefabsCoroutine(List<Transform> waypoints)
    {
        int successfulSpawns = 0;
        int maxAttempts = spawnCount * 10;
        int attempts = 0;
        int batchSize = 10; // Number of attempts per frame
        int currentBatch = 0;

        while (successfulSpawns < spawnCount && attempts < maxAttempts)
        {
            currentBatch = 0;
            while (currentBatch < batchSize && successfulSpawns < spawnCount && attempts < maxAttempts)
            {
                Vector3 spawnPoint;
                if (GetRandomPointInBounds(out spawnPoint))
                {
                    GameObject spawnedObject = PrefabUtility.InstantiatePrefab(prefabToSpawn) as GameObject;
                    spawnedObject.transform.SetParent(targetParent.transform);
                    spawnedObject.transform.position = spawnPoint;

                    var controller = spawnedObject.GetComponent<AgenticController>();
                    if (controller == null)
                    {
                        controller = spawnedObject.AddComponent<AgenticController>();
                    }

                    var character = spawnedObject.GetComponent<AgenticCharacter>();
                    if (character != null)
                    {
                        character.ID = Random.Range(1, 11);
                    }

                    string dayPlan = GenerateRandomDayPlan(waypoints);
                    controller.initialDayPlanJson = dayPlan;

                    Undo.RegisterCreatedObjectUndo(spawnedObject, "Spawn NPC");
                    successfulSpawns++;
                    Debug.Log($"Successfully spawned NPC {successfulSpawns}/{spawnCount} at {spawnPoint}");
                }
                attempts++;
                currentBatch++;
            }

            // Update progress bar
            float progress = (float)successfulSpawns / spawnCount;
            EditorUtility.DisplayProgressBar("Spawning NPCs", 
                $"Spawned {successfulSpawns}/{spawnCount} NPCs...", 
                progress);

            yield return new EditorWaitForSeconds(0.1f);
        }

        EditorUtility.ClearProgressBar();

        if (successfulSpawns < spawnCount)
        {
            Debug.LogWarning($"Only able to spawn {successfulSpawns} NPCs out of {spawnCount} requested.");
            EditorUtility.DisplayDialog("Warning", 
                $"Only able to spawn {successfulSpawns} NPCs out of {spawnCount} requested. " +
                "This might be due to limited valid NavMesh positions in the spawn area.", 
                "OK");
        }
        else
        {
            Debug.Log($"Successfully spawned all {spawnCount} NPCs!");
        }

        spawnArea.SetActive(false);
    }

    private string GenerateRandomDayPlan(List<Transform> waypoints)
    {
        DayPlan plan = new DayPlan();
        plan.day_overview = "Live a good day going to work, hitting the gym, and returning home";
        plan.actions = new List<DayPlanAction>();

        // Get all waypoints by type
        var homes = waypoints.Where(w => w.name.Contains("Home")).ToList();
        var works = waypoints.Where(w => w.name.Contains("Work")).ToList();
        var gyms = waypoints.Where(w => w.name.Contains("Gym")).ToList();

        // Randomly select specific locations
        Transform home = homes[Random.Range(0, homes.Count)];
        Transform work = works[Random.Range(0, works.Count)];
        Transform gym = gyms[Random.Range(0, gyms.Count)];

        // Create the fixed sequence with random locations
        var sequence = new[]
        {
            new { location = home, action = "Start the day at home", emoji = "🏠" },
            new { location = work, action = "Work at the office", emoji = "💼" },
            new { location = gym, action = "Exercise at the gym", emoji = "🏋️" },
            new { location = home, action = "Return home to rest", emoji = "🏠" }
        };

        foreach (var item in sequence)
        {
            DayPlanAction action = new DayPlanAction
            {
                action = item.action,
                emoji = item.emoji,
                location = item.location.name
            };

            plan.actions.Add(action);
        }

        return JsonUtility.ToJson(plan, true);
    }

    private bool GetRandomPointInBounds(out Vector3 result)
    {
        if (spawnArea == null)
        {
            result = Vector3.zero;
            return false;
        }

        Bounds bounds = spawnArea.GetComponent<Collider>().bounds;
        Vector3 randomPoint = new Vector3(
            Random.Range(bounds.min.x, bounds.max.x),
            bounds.center.y,
            Random.Range(bounds.min.z, bounds.max.z)
        );

        // Start the ray from high above the random point
        Vector3 rayStart = randomPoint + Vector3.up * maxRaycastDistance / 2f;
        
        // Get all hits along the ray
        RaycastHit[] hits = Physics.RaycastAll(rayStart, Vector3.down, maxRaycastDistance);
        
        // Find the highest (largest y value) hit that's tagged Environment
        float highestPoint = float.MinValue;
        bool foundEnvironment = false;
        
        foreach (RaycastHit hit in hits)
        {
            if (hit.collider.CompareTag("Environment"))
            {
                if (hit.point.y > highestPoint)
                {
                    highestPoint = hit.point.y;
                    foundEnvironment = true;
                }
            }
        }

        if (foundEnvironment)
        {
            result = new Vector3(randomPoint.x, highestPoint + 2f, randomPoint.z); // 2f is height offset
            return true;
        }

        result = Vector3.zero;
        return false;
    }

} 