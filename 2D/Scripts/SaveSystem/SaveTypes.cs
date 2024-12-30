using UnityEngine;
using CbAutorenTool.Tools;


[System.Serializable]
public class PlayerData
{
    public Vector3 Position;
    public int Money;
    public int Health;
    public int Stamina;
}

[System.Serializable]
public class DayCycleHandlerSaveData
{
    public CHugeDateTime CurrentDate;
    public bool IsDayTime;
    public bool LightsActive;
}

[System.Serializable]
public class EnvironmentData
{
    public FarmTileData[] ModifiedTiles;
    public CropData[] Crops;
}

[System.Serializable]
public class TimeSaveData
{
    public CHugeDateTime CurrentDate;
    public bool IsDayTime;
    public bool LightsActive;
}

[System.Serializable]
public class SaveData 
{
    public PlayerData PlayerData;
    public TimeSaveData TimeSaveData;
    public EnvironmentData EnvironmentData;
    public string SceneName;
}