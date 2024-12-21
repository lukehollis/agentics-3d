using UnityEngine;
using UnityEditor;
using Agentics;
using System.Collections.Generic;

public class AgenticsEditor : EditorWindow
{
    private List<AgenticCharacter> characters = new List<AgenticCharacter>();
    private bool isConnected = false;
    
    // Network settings
    private string backendUrl = "http://localhost:8000";
    private AgenticsEditorService editorService;
    
    // World settings
    private int worldId = 1;  // Default world ID
    private AgenticsEditorService.WorldData worldData;
    
    // Spawning settings
    private GameObject characterPrefab;
    private int numberOfCharactersToSpawn = 1;

    [MenuItem("Tools/Agentics/Agent Manager")]
    public static void ShowWindow()
    {
        GetWindow<AgenticsEditor>("Agent Manager");
    }

    void OnEnable()
    {
        editorService = new AgenticsEditorService(backendUrl);
    }

    void OnGUI()
    {
        GUILayout.Label("Agentics Network Manager", EditorStyles.boldLabel);

        EditorGUILayout.Space(10);
        
        // Network Configuration Section
        GUILayout.Label("Network Configuration", EditorStyles.boldLabel);
        using (new EditorGUILayout.VerticalScope("box"))
        {
            EditorGUI.BeginChangeCheck();
            backendUrl = EditorGUILayout.TextField("Backend URL", backendUrl);
            if (EditorGUI.EndChangeCheck())
            {
                editorService = new AgenticsEditorService(backendUrl);
            }
            
            worldId = EditorGUILayout.IntField("World ID", worldId);
            
            using (new EditorGUILayout.HorizontalScope())
            {
                EditorGUILayout.LabelField("Connection Status:");
                GUIContent iconContent = EditorGUIUtility.IconContent(
                    isConnected ? "d_winbtn_mac_max" : "d_winbtn_mac_close");
                GUILayout.Label(iconContent, GUILayout.Width(20));
            }

            if (GUILayout.Button("Test Connection"))
            {
                TestConnection();
            }

            if (isConnected && GUILayout.Button("Fetch World Data"))
            {
                FetchWorldData();
            }
        }

        EditorGUILayout.Space(10);

        // Character Management Section
        GUILayout.Label("Character Management", EditorStyles.boldLabel);
        using (new EditorGUILayout.VerticalScope("box"))
        {
            if (GUILayout.Button("Refresh Character List"))
            {
                RefreshCharacterList();
            }
            
            EditorGUILayout.Space(5);
            EditorGUILayout.LabelField("Active Characters:", characters.Count.ToString());

            foreach (var character in characters)
            {
                EditorGUILayout.ObjectField(character.name, character, typeof(AgenticCharacter), true);
            }
        }

        EditorGUILayout.Space(10);

        // Character Spawning Section
        GUILayout.Label("Character Spawning", EditorStyles.boldLabel);
        using (new EditorGUILayout.VerticalScope("box"))
        {
            characterPrefab = (GameObject)EditorGUILayout.ObjectField(
                "Character Prefab", characterPrefab, typeof(GameObject), false);

            numberOfCharactersToSpawn = EditorGUILayout.IntSlider(
                "Number to Spawn", numberOfCharactersToSpawn, 1, 100);

            if (worldData != null)
            {
                EditorGUILayout.LabelField("World:", worldData.name);
            }

            GUI.enabled = isConnected && worldData != null && characterPrefab != null;
            if (GUILayout.Button("Spawn Characters"))
            {
                SpawnCharacters();
            }
            GUI.enabled = true;
        }
    }

    private async void TestConnection()
    {
        isConnected = await editorService.TestConnection();
        Repaint();
    }

    private async void FetchWorldData()
    {
        worldData = await editorService.GetWorldData(worldId);
        if (worldData != null)
        {
            Debug.Log($"Fetched world data: {worldData.name}");
        }
        Repaint();
    }

    private void SpawnCharacters()
    {
        if (characterPrefab == null || worldData == null)
        {
            Debug.LogError("Missing required data for spawning!");
            return;
        }

        GameObject parent = new GameObject("Spawned Characters");
        
        for (int i = 0; i < numberOfCharactersToSpawn; i++)
        {
            Vector3 randomPosition = new Vector3(
                Random.Range(worldData.bounds.minX, worldData.bounds.maxX),
                worldData.bounds.y,
                Random.Range(worldData.bounds.minZ, worldData.bounds.maxZ)
            );

            GameObject instance = PrefabUtility.InstantiatePrefab(characterPrefab) as GameObject;
            instance.transform.parent = parent.transform;
            instance.transform.position = randomPosition;
            instance.transform.rotation = Quaternion.Euler(0, Random.Range(0, 360), 0);
            
            if (!instance.GetComponent<AgenticCharacter>())
            {
                Debug.LogWarning($"Adding AgenticCharacter component to {instance.name}");
                instance.AddComponent<AgenticCharacter>();
            }
        }

        RefreshCharacterList();
    }

    private void RefreshCharacterList()
    {
        characters.Clear();
        characters.AddRange(Object.FindObjectsOfType<AgenticCharacter>());
        Debug.Log($"Found {characters.Count} AgenticCharacters in scene");
    }
}
