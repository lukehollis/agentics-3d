using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using System.Collections;

public class SceneTransition : MonoBehaviour
{
    public float sceneWidth = 90f;  // Example width of each scene
    public float sceneHeight = 90f; // Example height of each scene
    public int currentSceneX;   // Starting X position in the grid (middle scene)
    public int currentSceneY;   // Starting Y position in the grid (middle scene)

    public GameObject modalWindow; // Reference to the modal window UI
    public Text modalText;         // Reference to the modal text UI
    public Button yesButton;       // Reference to the Yes button
    public Button noButton;        // Reference to the No button

    private int targetSceneX;
    private int targetSceneY;
    private string transitionDirection;

    void Start()
    {
        yesButton.onClick.AddListener(OnYesButtonClicked);
        noButton.onClick.AddListener(OnNoButtonClicked);
        modalWindow.SetActive(false);
    }

    void Update()
    {
        Vector3 playerPosition = transform.position;
        // Debug.Log("Player position: " + playerPosition);

        // Check for right edge transition
        if (playerPosition.x > sceneWidth)
        {
            StartCoroutine(TransitionToScene(currentSceneX + 1, currentSceneY, "east"));
        }
        // Check for left edge transition
        else if (playerPosition.x < 10)
        {
            StartCoroutine(TransitionToScene(currentSceneX - 1, currentSceneY, "west"));
        }
        // Check for top edge transition
        else if (playerPosition.y > sceneHeight)
        {
            StartCoroutine(TransitionToScene(currentSceneX, currentSceneY - 1, "north"));
        }
        // Check for bottom edge transition
        else if (playerPosition.y < 10)
        {
            StartCoroutine(TransitionToScene(currentSceneX, currentSceneY + 1, "south"));
        }
    }

    IEnumerator TransitionToScene(int x, int y, string direction)
    {
        // Ensure scene indices are within bounds (e.g., for 3x3 grid)
        x = Mathf.Clamp(x, 0, 2); 
        y = Mathf.Clamp(y, 0, 2); 

        // Create the scene name based on the grid coordinates
        string sceneName = "rome_" + x + "_" + y;

        // Load the scene asynchronously
        AsyncOperation asyncLoad = SceneManager.LoadSceneAsync(sceneName);

        // Wait until the asynchronous scene fully loads
        while (!asyncLoad.isDone)
        {
            yield return null;
        }

        // Subscribe to the sceneLoaded event
        SceneManager.sceneLoaded += OnSceneLoaded;

        // Store the target scene coordinates and direction
        targetSceneX = x;
        targetSceneY = y;
        transitionDirection = direction;
    }

    void OnSceneLoaded(Scene scene, LoadSceneMode mode)
    {
        // Adjust player position after loading the new scene
        // if (transitionDirection == "east") // Moved right
        //     transform.position = new Vector3(0, transform.position.y, 0);
        // else if (transitionDirection == "west") // Moved left
        //     transform.position = new Vector3(sceneWidth, transform.position.y, 0);
        // else if (transitionDirection == "north") // Moved up
        //     transform.position = new Vector3(transform.position.x, 0, 0);
        // else if (transitionDirection == "south") // Moved down
        //     transform.position = new Vector3(transform.position.x, sceneHeight, 0);

        currentSceneX = targetSceneX;
        currentSceneY = targetSceneY;

        // Unsubscribe from the sceneLoaded event
        SceneManager.sceneLoaded -= OnSceneLoaded;
    }

    void OnYesButtonClicked()
    {
        modalWindow.SetActive(false);
    }

    void OnNoButtonClicked()
    {
        modalWindow.SetActive(false);
    }
}
