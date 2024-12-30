using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Agentics;
using UnityEngine.Tilemaps;
using CbAutorenTool.Tools; // For CHugeDateTime


public enum GameState { FreeRoam, Dialog, Battle, Sleeping }

public class GameController : MonoBehaviour
{
    // Singleton instance
    private static GameController instance;
    public static GameController Instance
    {
        get
        {
            if (instance == null)
            {
                instance = FindObjectOfType<GameController>();
                if (instance == null)
                {
                    var singleton = new GameObject();
                    instance = singleton.AddComponent<GameController>();
                    singleton.name = typeof(GameController).ToString();
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
    public DayNightCycle dayCycleHandler;
    public DayNightAudioManager dayNightAudioManager;
    public GameObject BlackWindowUI;

    // Add environment reference
    [Header("Environment")]
    [SerializeField] private EnvironmentManager environmentManager;
    public EnvironmentManager EnvironmentManager => environmentManager;

    [Header("Start Screen")]
    public GameObject startScreen;
    public StartScreenController startScreenController;
    public AudioSource startScreenBackgroundMusic;
    public AudioSource startingSound;

    [System.NonSerialized]
    public bool isOfflineMode = false;
    [System.NonSerialized]
    public bool hasStarted = false;

    // Add this field with the other serialized fields
    [SerializeField] private Tilemap walkSurfaceTilemap;
    public Tilemap WalkSurfaceTilemap => walkSurfaceTilemap;

    // Add this property after the other property declarations (around line 52)
    public SceneData LoadedSceneData { get; set; }

    // Add this with other serialized fields around line 37-42
    [SerializeField] private InventoryController inventoryController;
    public InventoryController InventoryController => inventoryController;

    // Add this field near other UI-related fields
    private CanvasGroup blackWindowCanvasGroup;

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
        player = FindObjectOfType<Player>();

        // Get time-related components
        timeline = GetComponent<Timeline>();
        dayCycleHandler = GetComponent<DayNightCycle>();
        dayNightAudioManager = GetComponent<DayNightAudioManager>();

        // Get terrain component
        environmentManager = GetComponent<EnvironmentManager>();
        if (environmentManager == null)
        {
            environmentManager = FindObjectOfType<EnvironmentManager>();
        }

        // Add this in the Awake method after the other component gets
        walkSurfaceTilemap = GetComponent<Tilemap>();
        if (walkSurfaceTilemap == null)
        {
            walkSurfaceTilemap = FindObjectOfType<Tilemap>();
        }

        // get the gameobject of the start screen controller and ensure it's active
        startScreen.SetActive(true);

        // Add this at the end of Awake
        blackWindowCanvasGroup = BlackWindowUI.GetComponent<CanvasGroup>();
        if (blackWindowCanvasGroup == null)
            blackWindowCanvasGroup = BlackWindowUI.AddComponent<CanvasGroup>();
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
        }
        else if (state == GameState.Battle)
        {
            // Battle state logic here
        }
        else if (state == GameState.Sleeping)
        {
            // No updates needed while sleeping
            // Time passage is handled by Timeline
        }
    }

    // Add helper method to get items
    public Item GetItem(string itemId)
    {
        return itemDatabase.GetFromID(itemId);
    }

    public void HandleGameLoaded()
    {
        // play the starting sound
        if (startingSound != null)  
        {
            startingSound.Play();
        }

        if (!isOfflineMode && !hasStarted)
        {
            startScreenController.StartGame();
        }

    }

    public void Pause()
    {
        Time.timeScale = 0f;
    }

    public void Resume()
    {
        Time.timeScale = 1f;
    }

    public void SetOfflineMode(bool isOffline)
    {
        Debug.Log("SetOfflineMode in GameController");
        isOfflineMode = isOffline;
    }

    public void SetHasStarted(bool hasStarted)
    {
        this.hasStarted = hasStarted;
    }

    public void StartNighttime()
    {
        // Get player's current or designated sleeping position
        Vector3 sleepPosition = player.transform.position;
        state = GameState.Sleeping;
        
        // Get current time from Timeline
        int currentHour = timeline.currentDate.Hour;
        
        if (currentHour >= 17) // After 5 PM
        {
            StartCoroutine(SleepUntilMorning(sleepPosition));
        }
        else // Before 5 PM
        {
            StartCoroutine(NapForTwoHours(sleepPosition));
        }
    }

    private IEnumerator SleepUntilMorning(Vector3 sleepPosition)
    {
        // Set player to sleep position
        player.StartSleeping(sleepPosition);
        yield return new WaitForSeconds(0.5f);
        
        // Fade to black
        BlackWindowUI.SetActive(true);
        yield return FadeBlackWindow(0f, 1f, 1f);
        
        // set the target time to 6am next day
        CHugeDateTime targetTime = new CHugeDateTime(
            timeline.currentDate.Year + (timeline.currentDate.Hour >= 19 ? 1 : 0),
            timeline.currentDate.Month,
            timeline.currentDate.Day + 1,
            6, // Hour
            0, // Minute
            0  // Second
        );
        
        // Create a completion flag
        bool fastForwardComplete = false;
        
        // Fast forward to target time over 4 seconds
        timeline.FastForwardTo(targetTime, 4f, () => {
            Debug.Log("Fast forward to morning complete");
            fastForwardComplete = true;
        });

        // Wait until fast forward completes
        yield return new WaitUntil(() => fastForwardComplete);

        // Now handle the fade and cleanup
        yield return FadeBlackWindow(1f, 0f, 0.5f);
        yield return new WaitForSeconds(0.1f); // Add small delay to ensure fade completes
        BlackWindowUI.SetActive(false);
        
        Debug.Log("returning to free roam after waking");
        // Return to free roam after waking
        yield return new WaitForSeconds(1f);
        player.WakeUp();
        SetState(GameState.FreeRoam);
    }

    private IEnumerator NapForTwoHours(Vector3 sleepPosition)
    {
        Debug.Log("NapForTwoHours in GameController");

        // Set player to sleep position
        player.StartSleeping(sleepPosition);
        yield return new WaitForSeconds(0.5f);

        Debug.Log("fading in black ui cover");
        // Fade to black
        BlackWindowUI.SetActive(true);
        yield return FadeBlackWindow(0f, 1f, 1f);
        
        // Store the target time (current time + 2 hours)
        CHugeDateTime targetTime = timeline.currentDate.AddHours(2);
        
        // Create a completion flag
        bool fastForwardComplete = false;
        
        // Fast forward to target time over 2 seconds
        timeline.FastForwardTo(targetTime, 2f, () => {
            Debug.Log("Fast forward complete");
            fastForwardComplete = true;
        });

        // Wait until fast forward completes
        yield return new WaitUntil(() => fastForwardComplete);

        yield return new WaitForSeconds(0.1f); // Add small delay to ensure fade completes
        // Now handle the fade and cleanup
        yield return FadeBlackWindow(1f, 0f, 1f);
        yield return new WaitForSeconds(0.1f); // Add small delay to ensure fade completes
        BlackWindowUI.SetActive(false);
        
        Debug.Log("returning to free roam after waking");
        // Return to free roam after waking
        yield return new WaitForSeconds(1f);
        player.WakeUp();
        SetState(GameState.FreeRoam);
    }

    public void OnPlayerSleepComplete()
    {
        // Handle time passage, stat recovery, etc.
        state = GameState.FreeRoam;
        player.WakeUp();
    }

    public void SetState(GameState newState)
    {
        if (state == newState) return;
        
        // Exit current state
        switch (state)
        {
            case GameState.Dialog:
                // Clean up any dialog-specific state
                break;
            case GameState.Battle:
                // Clean up any battle-specific state
                break;
            case GameState.Sleeping:
                // Clean up any sleep-specific state
                break;
        }

        state = newState;

        // Enter new state
        switch (newState)
        {
            case GameState.FreeRoam:
                Resume(); // Ensure time is running normally
                break;
            case GameState.Dialog:
                // Initialize dialog-specific state
                break;
            case GameState.Battle:
                // Initialize battle-specific state
                break;
            case GameState.Sleeping:
                // Initialize sleep-specific state
                break;
        }
    }

    // Add this new helper method for fading
    private IEnumerator FadeBlackWindow(float startAlpha, float endAlpha, float duration)
    {
        float elapsedTime = 0f;
        while (elapsedTime < duration)
        {
            elapsedTime += Time.deltaTime;
            float currentAlpha = Mathf.Lerp(startAlpha, endAlpha, elapsedTime / duration);
            blackWindowCanvasGroup.alpha = currentAlpha;
            yield return null;
        }
        blackWindowCanvasGroup.alpha = endAlpha;
    }
}
