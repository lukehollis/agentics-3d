using UnityEngine;
using UnityEditor;
using System.Collections.Generic;

public class MaterialSetterWindow : EditorWindow
{
    private GameObject targetParent;
    private Material materialToApply;
    private bool previewChanges = true;
    private List<Material> originalMaterials = new List<Material>();
    private List<Renderer> adjustableObjects = new List<Renderer>();

    [MenuItem("Tools/BART/Material Setter")]
    public static void ShowWindow()
    {
        GetWindow<MaterialSetterWindow>("Material Setter");
    }

    private void OnEnable()
    {
        SceneView.duringSceneGui += OnSceneGUI;
    }

    private void OnDisable()
    {
        SceneView.duringSceneGui -= OnSceneGUI;
        RestoreOriginalMaterials();
    }

    private void OnGUI()
    {
        EditorGUILayout.LabelField("Material Setter", EditorStyles.boldLabel);
        
        using (new EditorGUILayout.VerticalScope("box"))
        {
            targetParent = (GameObject)EditorGUILayout.ObjectField(
                "Parent Object", 
                targetParent, 
                typeof(GameObject), 
                true
            );

            materialToApply = (Material)EditorGUILayout.ObjectField(
                "Material to Apply", 
                materialToApply, 
                typeof(Material), 
                false
            );

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
            if (GUILayout.Button("Apply Material"))
            {
                ApplyMaterial();
            }
            EditorGUI.EndDisabledGroup();
        }

        if (adjustableObjects.Count > 0)
        {
            EditorGUILayout.HelpBox(
                $"Found {adjustableObjects.Count} objects to modify.", 
                MessageType.Info
            );
        }
    }

    private void ScanObjects()
    {
        if (targetParent == null)
        {
            EditorUtility.DisplayDialog("Error", "Please select a parent object", "OK");
            return;
        }

        adjustableObjects.Clear();
        originalMaterials.Clear();

        Renderer[] renderers = targetParent.GetComponentsInChildren<Renderer>(true);
        foreach (Renderer renderer in renderers)
        {
            if (renderer.gameObject != targetParent)
            {
                adjustableObjects.Add(renderer);
                originalMaterials.Add(renderer.sharedMaterial);
            }
        }

        if (previewChanges && materialToApply != null)
        {
            PreviewMaterial();
        }
    }

    private void PreviewMaterial()
    {
        foreach (Renderer obj in adjustableObjects)
        {
            if (obj != null)
            {
                obj.sharedMaterial = materialToApply;
            }
        }
        SceneView.RepaintAll();
    }

    private void RestoreOriginalMaterials()
    {
        for (int i = 0; i < adjustableObjects.Count; i++)
        {
            if (adjustableObjects[i] != null && i < originalMaterials.Count)
            {
                adjustableObjects[i].sharedMaterial = originalMaterials[i];
            }
        }
        SceneView.RepaintAll();
    }

    private void ApplyMaterial()
    {
        if (materialToApply == null)
        {
            EditorUtility.DisplayDialog("Error", "Please select a material to apply", "OK");
            return;
        }

        Undo.RecordObjects(adjustableObjects.ToArray(), "Apply Material to Objects");
        
        foreach (Renderer obj in adjustableObjects)
        {
            if (obj != null)
            {
                obj.sharedMaterial = materialToApply;
            }
        }
        
        EditorUtility.DisplayDialog("Success", "Material applied successfully!", "OK");
    }

    private void OnSceneGUI(SceneView sceneView)
    {
        if (!previewChanges || adjustableObjects.Count == 0) return;

        Handles.color = Color.green;
        foreach (Renderer obj in adjustableObjects)
        {
            if (obj != null)
            {
                Handles.DrawWireCube(obj.bounds.center, obj.bounds.size);
            }
        }
    }
} 