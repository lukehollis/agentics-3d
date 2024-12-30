using UnityEngine;
using UnityEngine.Tilemaps;
using Agentics;

[CreateAssetMenu(fileName = "Crop", menuName = "2D Farming/Crop")]
public class Crop : Item 
{
    [Header("Basic Info")]
    public Tile[] growthStages;
    public float daysPerGrowthStage = 1f;
    
    [Header("Harvest Settings")]
    public bool isMultiHarvest;
    public int maxHarvestCount = 1;
    public Tile harvestedProduct;
    public int productPerHarvest = 1;
    
    [Header("Seed Settings")]
    public Tile seedItem;
    
    // Implement abstract methods from Item class
    public override bool CanUse(Vector3Int target)
    {
        return GameController.Instance.EnvironmentManager.IsPlantable(target);
    }

    public override bool Use(Vector3Int target)
    {
        GameController.Instance.EnvironmentManager.PlantAt(target, this);
        return true;
    }
}
