using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(Train))]
public class TrainEditor : Editor
{
    public override void OnInspectorGUI()
    {
        Train train = (Train)target;
        
        DrawDefaultInspector();
        
        if (GUILayout.Button("Setup Cars"))
        {
            train.SetupCars();
        }
    }
}