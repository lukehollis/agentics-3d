using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(BasePath), true)]  // true allows it to work with derived classes
public class BasePathEditor : Editor
{
    public override void OnInspectorGUI()
    {
        BasePath path = (BasePath)target;
        
        DrawDefaultInspector();
        
        EditorGUILayout.Space(10);
        
        if (GUILayout.Button("Add Node"))
        {
            GameObject node = new GameObject($"Node_{path.nodes.Count}");
            node.transform.parent = path.transform;
            
            // Position new node relative to last node or parent
            if (path.nodes.Count > 0)
            {
                node.transform.position = path.nodes[path.nodes.Count - 1].position + Vector3.right * 5f;
            }
            else
            {
                node.transform.position = path.transform.position;
            }
            
            path.nodes.Add(node.transform);
            EditorUtility.SetDirty(path);
        }

        if (GUILayout.Button("Refresh Path"))
        {
            // Force recreation of renderers
            var renderers = path.GetComponentsInChildren<LineRenderer>();
            foreach (var renderer in renderers)
            {
                DestroyImmediate(renderer.gameObject);
            }
            
            // Call Awake to recreate everything
            path.Awake();
            EditorUtility.SetDirty(path);
        }
    }
} 