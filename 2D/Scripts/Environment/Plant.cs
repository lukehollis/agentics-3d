using System.Collections;
using UnityEngine;
using UnityEngine.Tilemaps;
using Agentics;

public class Plant : MonoBehaviour
{
    private Crop cropData;
    private int currentStage = 0;
    private int harvestCount = 0;
    private bool isWatered = false;
    private Vector3Int tilePosition;
    private Tilemap tilemap;

    public void Initialize(Crop crop, Vector3Int position, Tilemap targetTilemap)
    {
        cropData = crop;
        tilePosition = position + new Vector3Int(0, 1, 0);
        tilemap = targetTilemap;
        
        if (cropData.growthStages.Length > 0)
        {
            tilemap.SetTile(tilePosition, cropData.growthStages[0]);
            StartCoroutine(GrowPlant());
        }
    }

    private IEnumerator GrowPlant()
    {
        while (currentStage < cropData.growthStages.Length - 1)
        {
            yield return new WaitForSeconds(cropData.daysPerGrowthStage * GameController.Instance.timeline.gameDayDuration);
            
            if (!isWatered)
            {
                Debug.Log("TODO: withered crop state");
            }
            
            currentStage++;
            tilemap.SetTile(tilePosition, cropData.growthStages[currentStage]);
            isWatered = false; // Reset water status each day
        }
    }

    public bool Harvest()
    {
        if (currentStage < cropData.growthStages.Length - 1)
            return false;

        harvestCount++;
        
        // Add harvested items to inventory
        GameController.Instance.player.inventory.AddItem(cropData, cropData.productPerHarvest);
        
        if (cropData.isMultiHarvest && harvestCount < cropData.maxHarvestCount)
        {
            // Reset to previous growth stage for multi-harvest crops
            currentStage--;
            tilemap.SetTile(tilePosition, cropData.growthStages[currentStage]);
            StartCoroutine(GrowPlant());
            return true;
        }
        
        // Destroy plant if single harvest or max harvests reached
        GameController.Instance.EnvironmentManager.RemovePlant(tilePosition);
        tilemap.SetTile(tilePosition, null);  // Clear the tile
        Destroy(gameObject);
        return true;
    }

    public void Water()
    {
        isWatered = true;
    }
}
