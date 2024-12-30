using UnityEngine;
using UnityEngine.Tilemaps;
using System.Collections.Generic;
using Agentics;


public class EnvironmentManager : MonoBehaviour, Interactable
{
    [SerializeField] public Tilemap interactableMap;
    [SerializeField] public Tilemap plantMap;
    [SerializeField] private Tile hiddenInteractableTile;
    [SerializeField] private RuleTile plowedTile;
    [SerializeField] private GameObject plantPrefab;
    [SerializeField] private AudioClip plowSound;
    [SerializeField] private AudioClip waterSound;
    [SerializeField] private AudioClip plantSound;
    [SerializeField] private AudioClip harvestSound;
    private AudioSource audioSource;
    
    private Dictionary<Vector3Int, GameObject> plantsOnTiles = new Dictionary<Vector3Int, GameObject>();
    private Dictionary<Vector3Int, CropData> cropData = new Dictionary<Vector3Int, CropData>();
    private Dictionary<Vector3Int, FarmTileData> farmTiles = new Dictionary<Vector3Int, FarmTileData>();

    private void Start()
    {
        InitializeInteractableMap();
        audioSource = gameObject.AddComponent<AudioSource>();
    }

    private void InitializeInteractableMap()
    {
        // Get all non-empty tile positions in the map
        BoundsInt bounds = interactableMap.cellBounds;
        
        TileBase[] allTiles = interactableMap.GetTilesBlock(bounds);
        int tileCount = 0;

        for (int x = bounds.xMin; x < bounds.xMax; x++)
        {
            for (int y = bounds.yMin; y < bounds.yMax; y++)
            {
                Vector3Int tilePosition = new Vector3Int(x, y, 0);
                TileBase currentTile = interactableMap.GetTile(tilePosition);
                
                if (currentTile != null)
                {
                    // Set hidden interactable tile on base map
                    interactableMap.SetTile(tilePosition, hiddenInteractableTile);
                    // Initialize empty plant map at the same position
                    plantMap.SetTile(tilePosition, null);
                    tileCount++;
                }
            }
        }
    }

    public bool IsInteractable(Vector3Int position)
    {
        // set z = 0
        position.z = 0;
        
        TileBase tile = interactableMap.GetTile(position);
        
        if (tile != null)
        {
            return true;
        }
        
        return tile != null && (tile.name == "Interactable_invisible" || 
                              tile.name == "Dirt" || tile.name == "Interactable_visible");
    }

    public void PlowTile(Vector3Int position)
    {
        // Define the 3x3 grid around the center position
        for (int x = -1; x <= 1; x++)
        {
            for (int y = -1; y <= 1; y++)
            {
                Vector3Int adjacentPosition = position + new Vector3Int(x, y, 0);
                
                if (IsTillable(adjacentPosition))
                {
                    if (audioSource && plowSound) audioSource.PlayOneShot(plowSound, 0.5f);
                    interactableMap.SetTile(adjacentPosition, plowedTile);
                    
                    farmTiles[adjacentPosition] = new FarmTileData
                    {
                        Position = adjacentPosition,
                        StateType = TileStateType.Plowed,
                        IsWatered = false,
                        IsFertilized = false
                    };
                    
                    // Clear any existing tile flags that might affect rendering
                    interactableMap.SetTileFlags(adjacentPosition, TileFlags.None);

                    // This is vital to ensure the tile displays correctly, do not remove!
                    // Ensure the tile is visible
                    Color visibleColor = Color.white;
                    interactableMap.SetColor(adjacentPosition, visibleColor);

                    interactableMap.RefreshTile(adjacentPosition);
                    Player player = GameController.Instance.player; 
                    // player.TriggerAnimation("hacking");
                }
            }
        }
        
        // After all tiles are plowed
        SaveSystem.Instance.Save();
    }

    public void PlantAt(Vector3Int position, Crop crop)
    {
        if (IsPlantable(position))
        {
            // Remove one seed from inventory before planting
            if (!GameController.Instance.player.inventory.RemoveItem(crop, 1))
            {
                return;
            }
            if (audioSource && plantSound) audioSource.PlayOneShot(plantSound, 0.5f);
            Vector3 worldPos = plantMap.GetCellCenterWorld(position);
            GameObject plantObj = Instantiate(plantPrefab, worldPos, Quaternion.identity);
            
            Plant plant = plantObj.GetComponent<Plant>();
            plant.Initialize(crop, position, plantMap);
            
            // Check if tile is watered and update plant accordingly
            if (farmTiles.TryGetValue(position, out FarmTileData tileData) && tileData.IsWatered)
            {
                plant.Water();
            }
            
            plantsOnTiles[position] = plantObj;
            cropData[position] = new CropData
            {
                Position = position,
                GrowingCrop = crop,
                WorldPosition = worldPos,
                GrowthRatio = 0f,
                IsWatered = farmTiles.TryGetValue(position, out FarmTileData data) && data.IsWatered
            };
        }
        
        // After successful planting
        SaveSystem.Instance.Save();
    }

    public void RemovePlant(Vector3Int position)
    {
        if (plantsOnTiles.TryGetValue(position, out GameObject plant))
        {
            plantsOnTiles.Remove(position);
            cropData.Remove(position);
            plantMap.SetTile(position, null);
            interactableMap.SetTile(position, plowedTile);
            // Don't remove tileState - the tile is still plowed
            Destroy(plant);
        }
    }

    public void WaterAt(Vector3Int position)
    {
        if (IsTilled(position))
        {
            if (audioSource && waterSound) audioSource.PlayOneShot(waterSound, 0.5f);
            Player player = GameController.Instance.player; 
            // player.TriggerAnimation("watering");
            // player.PlayWateringAnimation();

            if (farmTiles.TryGetValue(position, out FarmTileData tileData))
            {
                tileData.IsWatered = true;
                farmTiles[position] = tileData;
            }

            interactableMap.SetTileFlags(position, TileFlags.None);
            Color darkenedColor = new Color(0.7f, 0.7f, 0.7f, 1f);
            interactableMap.SetColor(position, darkenedColor);
            
            if (plantsOnTiles.TryGetValue(position, out GameObject plantObj))
            {
                Plant plant = plantObj.GetComponent<Plant>();
                plant.Water();
                
                if (cropData.TryGetValue(position, out CropData data))
                {
                    data.IsWatered = true;
                    cropData[position] = data;
                }
            }
        }
        
        // After successful watering
        SaveSystem.Instance.Save();
    }

    public Crop HarvestAt(Vector3Int position)
    {
        if (plantsOnTiles.TryGetValue(position, out GameObject plantObj))
        {
            Plant plant = plantObj.GetComponent<Plant>();
            if (plant.Harvest())
            {
                if (audioSource && harvestSound) audioSource.PlayOneShot(harvestSound, 0.5f);
                Player player = GameController.Instance.player;
                // player.TriggerAnimation("lifting");

                var cropDataValue = cropData[position];
                // After successful harvesting
                SaveSystem.Instance.Save();
                return cropDataValue.GrowingCrop;
            }
        }
        return null;
    }

    public Vector3Int WorldToCell(Vector3 worldPosition) => interactableMap.WorldToCell(worldPosition);
    public bool IsTillable(Vector3Int position) => IsInteractable(position) && !cropData.ContainsKey(position);
    public bool IsTilled(Vector3Int position) => interactableMap.GetTile(position)?.name == "Dirt";
    public bool IsPlantable(Vector3Int position) => IsTilled(position) && !cropData.ContainsKey(position);
    public CropData? GetCropDataAt(Vector3Int position) => cropData.TryGetValue(position, out CropData data) ? data : null;

    public EnvironmentData Serialize()
    {
        var crops = new List<CropData>();
        var modifiedTiles = new List<FarmTileData>();

        // Save all farm tiles
        foreach (var kvp in farmTiles)
        {
            modifiedTiles.Add(kvp.Value);
        }

        // Save crops
        foreach (var kvp in cropData)
        {
            crops.Add(kvp.Value);
        }

        return new EnvironmentData
        {
            ModifiedTiles = modifiedTiles.ToArray(),
            Crops = crops.ToArray()
        };
    }

    public void Load(EnvironmentData data)
    {
        // Clear existing data
        foreach (var plant in plantsOnTiles.Values)
        {
            Destroy(plant);
        }
        plantsOnTiles.Clear();
        cropData.Clear();
        farmTiles.Clear();
        
        // Load farm tiles
        if (data.ModifiedTiles != null)
        {
            foreach (var tileData in data.ModifiedTiles)
            {
                farmTiles[tileData.Position] = tileData;
                
                if (tileData.StateType == TileStateType.Plowed)
                {
                    interactableMap.SetTile(tileData.Position, plowedTile);
                    
                    // Add these lines to ensure the tile is visible
                    interactableMap.SetTileFlags(tileData.Position, TileFlags.None);
                    interactableMap.SetColor(tileData.Position, Color.white);
                    interactableMap.RefreshTile(tileData.Position);
                }
                
                if (tileData.IsWatered)
                {
                    interactableMap.SetTileFlags(tileData.Position, TileFlags.None);
                    interactableMap.SetColor(tileData.Position, new Color(0.7f, 0.7f, 0.7f, 1f));
                }
            }
        }

        // Load crops (existing code)
        if (data.Crops != null)
        {
            foreach (var crop in data.Crops)
            {
                GameObject plantObj = Instantiate(plantPrefab, crop.WorldPosition, Quaternion.identity);
                Plant plant = plantObj.GetComponent<Plant>();
                plant.Initialize(crop.GrowingCrop, crop.Position, interactableMap);
                
                if (crop.IsWatered)
                {
                    plant.Water();
                }
                
                plantsOnTiles[crop.Position] = plantObj;
                cropData[crop.Position] = crop;
            }
        }
    }

    public void Interact()
    {
        // must be included for implementation of interactable
    }

    public void TileInteract(Vector3Int cellPosition)
    {
        cellPosition.z = 0;
        
        // Get active item from player's inventory
        Item activeItem = GameController.Instance.player.inventory.GetActiveItem();
        
        if (activeItem != null && activeItem.CanUse(cellPosition))
        {
            activeItem.Use(cellPosition);
        }
        else
        {
            // Default interaction if no item is selected or item can't be used
            InteractWithTile(cellPosition);
        }
    }

    public void InteractWithTile(Vector3Int position)
    {
        // If no plant, check if we can plant (tile is plowed but empty)
        if (IsPlantable(position))
        {
            // Get active item from player's inventory
            Item activeItem = GameController.Instance.player.inventory.GetActiveItem();
            if (activeItem) {
                if (activeItem is Crop crop)
                {
                    PlantAt(position, crop);
                    return;
                }
            }
        }
        // If not plowed, plow it
        else if (IsTillable(position))
        {
            Item activeItem = GameController.Instance.player.inventory.GetActiveItem();
            if (activeItem != null) {
                if (activeItem is Hoe hoe) {
                    PlowTile(position);
                    return;
                }
            }
        }
        // Check if there's a plant at this position
        else if (plantsOnTiles.TryGetValue(position, out GameObject plantObj))
        {
            Plant plant = plantObj.GetComponent<Plant>();
            
            // If plant is ready for harvest, harvest it
            if (plant.Harvest())
            {
                return;
            }
        }

        // Check if tile can be watered (is plowed)
        else if (IsTilled(position))
        {
            // Check if the tile has a plant
            bool hasPlant = cropData.TryGetValue(position, out CropData data);
            
            // Water if: has no plant, OR has plant but isn't watered
            if (!hasPlant || (hasPlant && !data.IsWatered))
            {
                Item activeItem = GameController.Instance.player.inventory.GetActiveItem();
                if (activeItem is WateringCan)  // Assuming you have a WateringCan item type
                {
                    WaterAt(position);
                    return;
                }
            }
        }
    }

    public bool IsTileWatered(Vector3Int position)
    {
        return farmTiles.TryGetValue(position, out FarmTileData tileData) && tileData.IsWatered;
    }

    public bool IsTileFertilized(Vector3Int position)
    {
        return farmTiles.TryGetValue(position, out FarmTileData tileData) && tileData.IsFertilized;
    }
}
