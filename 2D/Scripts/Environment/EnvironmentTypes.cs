using UnityEngine;


[System.Serializable]
public struct CropData
{
    public Vector3Int Position;
    public Crop GrowingCrop;
    public Vector3 WorldPosition;  // Added for save/load
    public float GrowthRatio;
    public bool IsWatered;
    
    // Helper method for serialization
    public string GetPlantPrefabName() => GrowingCrop.DisplayName;
}

[System.Serializable]
public enum TileStateType
{
    Default,
    Plowed,
    Planted
}

[System.Serializable]
public struct FarmTileData
{
    public Vector3Int Position;
    public TileStateType StateType;
    public bool IsWatered;
    public bool IsFertilized;
    // Could add other states like:
    // public float Fertility;
    // public float Moisture;
    // public bool HasWeeds;
}
