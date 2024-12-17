using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Agentics;
using UnityEngine.Tilemaps;


public enum GameState { FreeRoam, Dialog, Battle }

public class SimulationController : MonoBehaviour
{
    // Singleton instance
    private static SimulationController instance;
    public static SimulationController Instance
    {
        get
        {
            if (instance == null)
            {
                instance = FindObjectOfType<SimulationController>();
                if (instance == null)
                {
                    var singleton = new GameObject();
                    instance = singleton.AddComponent<SimulationController>();
                    singleton.name = typeof(SimulationController).ToString();
                    DontDestroyOnLoad(singleton);
                }
            }
            return instance;
        }
    }

    // Game State Management
    [Header("Game State")]
    public GameState state;

    // Inventory System References
    [Header("Inventory System")]
    [SerializeField] public ItemDatabase itemDatabase;
    public Player player;
    public UIManager uiManager;

    // Time Management
    [Header("Time Management")]
    public Timeline timeline;

    // Add environment reference
    [Header("Environment")]
    [SerializeField] private EnvironmentManager environmentManager;
    public EnvironmentManager EnvironmentManager => environmentManager;

    // Add this with other serialized fields around line 37-42
    [SerializeField] private InventoryController inventoryController;
    public InventoryController InventoryController => inventoryController;


    private void Awake()
    {
        // Singleton pattern implementation
        if (instance != null && instance != this)
        {
            Destroy(gameObject);
            return;
        }
        
        instance = this;
        DontDestroyOnLoad(gameObject);

        // Initialize the database instead of getting ItemManager component
        itemDatabase.Init();
        
        // Initialize inventory controller
        inventoryController = GetComponent<InventoryController>();
        
        // Get other components
        player = FindObjectOfType<Player3D>();

        // Get time-related components
        timeline = GetComponent<Timeline>();

        // Get terrain component
        environmentManager = GetComponent<EnvironmentManager>();
        if (environmentManager == null)
        {
            environmentManager = FindObjectOfType<EnvironmentManager>();
        }

    }

    private void Start()
    {
        // Dialog state management
        DialogManager.Instance.OnShowDialog += () =>
        {
            state = GameState.Dialog;
        };
        
        DialogManager.Instance.OnHideDialog += () =>
        {
            state = GameState.FreeRoam;
        };
    }

    private void Update()
    {
        if (state == GameState.FreeRoam)
        {
            if (player)
            {
                player.HandleUpdate();
            }
        }
        else if (state == GameState.Dialog)
        {
            DialogManager.Instance.HandleUpdate();

            if (player)
            {
                player.HandleUpdate();
            }
        }
        else if (state == GameState.Battle)
        {
            // Battle state logic here
        }
    }

    // Add helper method to get items
    public Item GetItem(string itemId)
    {
        return itemDatabase.GetFromID(itemId);
    }

    public void Pause()
    {
        Time.timeScale = 0f;
    }

    public void Resume()
    {
        Time.timeScale = 1f;
    }
}
