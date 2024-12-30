using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks; // Add this line
using UnityEngine;
using NativeWebSocket;
using UnityEngine.Networking;
using Agentics;



public class NetworkingController : MonoBehaviour
{
    private WebSocket websocket;
    // keep available for local testing
    // private readonly string websocketUrl = "ws://civs.local:8000/ws/characters/";
    private readonly string websocketUrl = "wss://civs.mused.com/ws/characters/";
    public event Action OnWebSocketConnected;

    private Dictionary<int, AgenticController> characterControllers = new Dictionary<int, AgenticController>();
    private string user_uuid;
    private string game_session_uuid;

    private static NetworkingController instance;
    public static NetworkingController Instance
    {
        get
        {
            if (instance == null)
            {
                instance = FindObjectOfType<NetworkingController>();
                if (instance == null)
                {
                    var singleton = new GameObject();
                    instance = singleton.AddComponent<NetworkingController>();
                    singleton.name = typeof(NetworkingController).ToString();
                    DontDestroyOnLoad(singleton);
                }
            }
            return instance;
        }
    }

    private Dictionary<int, Action<string>> actionTasksCallbacks = new Dictionary<int, Action<string>>();
    private Dictionary<int, Action<byte[]>> tileGenerationCallbacks = new Dictionary<int, Action<byte[]>>();

    // Add this dictionary to store self-reflection callbacks
    private Dictionary<int, Action<bool>> selfReflectionCallbacks = new Dictionary<int, Action<bool>>();

    // Add this dictionary to store plan request callbacks
    private Dictionary<int, Action<bool, string>> planRequestCallbacks = new Dictionary<int, Action<bool, string>>();

    // Add this dictionary to store load game response callbacks
    private Dictionary<int, Action<string>> loadGameResponseCallbacks = new Dictionary<int, Action<string>>();

    private float reconnectInterval = 5f; // Try to reconnect every 5 seconds
    private bool shouldReconnect = true;
    private bool isInitializing = false; // Add this flag
    private bool hasInitialized = false; // Add this flag

    private async void Start()
    {
        // Initialization will proceed after receiving user_uuid
        #if UNITY_WEBGL && !UNITY_EDITOR
            // generate a default user_uuid for webgl in case the user_uuid is not received from the javascript
            user_uuid = Guid.NewGuid().ToString();
            Debug.Log("Generated user_uuid for WebGL: " + user_uuid);
        #else
            // For testing in the Unity editor or other platforms
            user_uuid = Guid.NewGuid().ToString();
            Debug.Log("Generated user_uuid for editor: " + user_uuid);
        #endif

        await InitializeWebSocket();
    }

    // This method will be called from JavaScript
    public async void OnUserUUIDReceived(string uuid)
    {
        user_uuid = uuid;
        Debug.Log("Received user_uuid from JavaScript: " + user_uuid);
        
        // Proceed with WebSocket initialization
        await InitializeWebSocket();
    }

    private async Task InitializeWebSocket()
    {
        if (isInitializing)
        {
            Debug.Log("Already initializing WebSocket - skipping");
            return;
        }
        
        isInitializing = true;
        hasInitialized = false;  // Reset this flag during initialization

        try 
        {
            if (websocket != null)
            {
                Debug.Log("Cleaning up existing WebSocket connection");
                await websocket.Close();
                websocket = null;
            }

            game_session_uuid = Guid.NewGuid().ToString();
            Debug.Log($"Creating new WebSocket connection to {websocketUrl}");
            websocket = new WebSocket(websocketUrl);

            websocket.OnOpen += () =>
            {
                Debug.Log("WebSocket connection opened successfully");
                hasInitialized = true;
                isInitializing = false;
                shouldReconnect = true;  // Reset reconnect flag
                OnWebSocketConnected?.Invoke();
            };

            websocket.OnError += (e) =>
            {
                Debug.LogError($"WebSocket error: {e}");
                hasInitialized = false;
                SimulationController.Instance.SetOfflineMode(true);

                if (!SimulationController.Instance.hasStarted)
                {
                    // var startScreen = FindObjectOfType<StartScreenController>();
                    // if (startScreen != null)
                    // {
                    //     startScreen.ShowOfflineMessage();
                    // }
                    SimulationController.Instance.HandleGameLoaded();
                }
                shouldReconnect = true;
            };

            websocket.OnClose += (e) =>
            {
                Debug.Log("WebSocket connection closed");
                hasInitialized = false;
                shouldReconnect = true;
            };

            websocket.OnMessage += (bytes) =>
            {
                var message = System.Text.Encoding.UTF8.GetString(bytes);
                OnMessageReceived(bytes);
            };

            // InvokeRepeating("SendWebSocketMessage", 0.0f, 0.3f);

            Debug.Log("Attempting to connect WebSocket...");
            await websocket.Connect();
        }
        catch (Exception ex)
        {
            Debug.LogError($"Error during WebSocket initialization: {ex.Message}");
            hasInitialized = false;
            shouldReconnect = true;
        }
        finally 
        {
            if (!hasInitialized)
            {
                isInitializing = false;  // Reset initialization flag if we failed
            }
        }
    }

    void Update()
    {
        #if !UNITY_WEBGL || UNITY_EDITOR
            if (websocket != null)
            {
                if (websocket.State == WebSocketState.Open)
                {
                    websocket.DispatchMessageQueue();
                }
                else if (shouldReconnect && websocket.State == WebSocketState.Closed)
                {
                    Debug.Log("WebSocket closed, attempting to reconnect...");
                    StartCoroutine(ReconnectWebSocket());
                }
            }
        #endif

        if (websocket == null)
        {
            Debug.Log($"[{Time.time}] WebSocket is null in Update()");
        }
        else
        {
            // Debug.Log($"[{Time.time}] WebSocket state: {websocket.State}");
        }
    }

    private IEnumerator ReconnectWebSocket()
    {
        if (isInitializing) yield break; // Don't attempt reconnect if already initializing
        
        shouldReconnect = false;  // Prevent multiple reconnection attempts
        yield return new WaitForSeconds(reconnectInterval);
        
        Debug.Log("Attempting to reconnect WebSocket...");
        yield return new WaitForSeconds(0);
        
        // Convert the async call to a coroutine-friendly version
        var initTask = InitializeWebSocket();
        while (!initTask.IsCompleted)
        {
            yield return null;
        }
        
        if (initTask.Exception != null)
        {
            Debug.LogError($"Error during reconnection: {initTask.Exception.Message}");
        }
        
        shouldReconnect = true;
    }

    private async void SendWebSocketMessage(string jsonString)
    {
        if (websocket == null)
        {
            Debug.LogError("WebSocket is not initialized.");
            return;
        }

        var bytesToSend = System.Text.Encoding.UTF8.GetBytes(jsonString);
        try
        {
            await websocket.Send(bytesToSend);
        }
        catch (Exception ex)
        {
            Debug.LogError("WebSocket send error:");
            if (ex != null && !string.IsNullOrEmpty(ex.Message))
            {
                Debug.LogError(" -- Error message: " + ex.Message);
            }
        }
    }

    private void OnMessageReceived(byte[] bytes)
    {
        var message = System.Text.Encoding.UTF8.GetString(bytes);
        Debug.Log("Raw message received: " + message);

        try
        {
            if (string.IsNullOrEmpty(message))
            {
                Debug.LogError("Received empty message");
                return;
            }

            if (!message.Contains("type"))
            {
                Debug.LogError("Message missing type field. Message: " + message);
                return;
            }

            var receivedData = JsonUtility.FromJson<ReceivedMessageData>(message);
            
            if (receivedData == null)
            {
                Debug.LogError("Failed to parse message into ReceivedMessageData. Message: " + message);
                return;
            }

            var messageType = receivedData.type;
            Debug.Log("Received message type: " + messageType);

            var characterId = receivedData.character_id;
            
            if (messageType == "action_tasks_response")
            {
                Debug.Log("Received action tasks response: " + receivedData.message);

                if (actionTasksCallbacks.TryGetValue(characterId, out var callback))
                {
                    callback(receivedData.message);
                    actionTasksCallbacks.Remove(characterId);
                }
                else
                {
                    Debug.LogWarning($"No callback found for character ID: {characterId}");
                }
            }
            else if (messageType == "plan_response")
            {
                Debug.Log("Received plan response: " + receivedData.message);

                // Call the callback if it exists
                if (planRequestCallbacks.TryGetValue(characterId, out var callback))
                {
                    callback(true, receivedData.message);
                    planRequestCallbacks.Remove(characterId);
                }

                // Update the character's plan in the game
                if (characterControllers.ContainsKey(characterId))
                {
                    var controller = characterControllers[characterId];
                    controller.UpdatePlan(receivedData.message);
                }
                else
                {
                    Debug.LogWarning($"No NPCController found for character ID: {characterId}");
                }
            }
            else if (messageType == "plan_error")
            {
                Debug.LogError($"Plan request failed for character {characterId}: {receivedData.message}");
                if (planRequestCallbacks.TryGetValue(characterId, out var callback))
                {
                    callback(false, null);
                    planRequestCallbacks.Remove(characterId);
                }
            }
            else if (messageType == "conversation_response")
            {
                // Handle the streamed conversation response
                Debug.Log("Received conversation response: " + receivedData.message);

                // Display the character's response in the conversation UI
                // Pass the conversation response to the DialogManager for display
                DialogManager.Instance.DisplayResponse(receivedData.message);
            }
            else if (messageType == "character_self_reflection_response")
            {
                // Handle the self-reflection response
                Debug.Log("Received self-reflection response: " + receivedData.message);

                bool tasksCompleted = false;

                // Parse the backend message to determine if tasks are completed
                // For example, if the message is "tasks_completed", set tasksCompleted to true
                // if (receivedData.message == "tasks_completed")
                // {
                tasksCompleted = true;
                // }
                // else
                // {
                //     tasksCompleted = false;
                // }

                // Invoke the callback for the character
                if (selfReflectionCallbacks.TryGetValue(characterId, out var callback))
                {
                    
                    callback?.Invoke(tasksCompleted);
                    selfReflectionCallbacks.Remove(characterId);
                }
                else
                {
                    Debug.LogWarning($"No callback found for character ID: {characterId}");
                }
            }
            else if (messageType == "tile_generated")
            {
                Debug.Log("Received tile generated response: " + receivedData.response);

                // Handle tile_generated message
                if (receivedData.response != null && !string.IsNullOrEmpty(receivedData.response.b64_json))
                {
                    string base64ImageData = receivedData.response.b64_json;
                    byte[] imageData = Convert.FromBase64String(base64ImageData);

                    if (tileGenerationCallbacks.TryGetValue(characterId, out var callback))
                    {
                        callback(imageData);
                        tileGenerationCallbacks.Remove(characterId);
                    }
                    else
                    {
                        Debug.LogWarning($"No callback found for character ID: {characterId}");
                    }
                }
                else
                {
                    Debug.LogError("Tile generation response is invalid.");
                }
            }
            else if (messageType == "load_game_response")
            {
                Debug.Log("Received saved game data");
                // Parse as LoadGameResponseData instead of ReceivedMessageData
                SimulationController.Instance.HandleGameLoaded();
            }
        }
        catch (Exception ex)
        {
            Debug.LogError($"Error deserializing WebSocket message: {ex.Message}");
            Debug.LogError($"Message content: {message}");
            Debug.LogError($"Stack trace: {ex.StackTrace}");
        }
    }

    public void RequestActionTasks(int characterId, DayPlanAction currentAction, Action<string> callback)
    {
        // Store the callback using characterId
        if (!actionTasksCallbacks.ContainsKey(characterId))
        {
            actionTasksCallbacks.Add(characterId, callback);
        }
        else
        {
            actionTasksCallbacks[characterId] = callback;
        }

        // Create the request message
        var message = new ActionTasksRequest
        {
            type = "character_task_planner",
            character_id = characterId,
            game_session_uuid = game_session_uuid, // Add this line
            action = currentAction
        };

        var jsonString = JsonUtility.ToJson(message);
        SendWebSocketMessage(jsonString);
    }

    public void RequestTileGeneration(int npcId, string tileDescription, Action<byte[]> callback)
    {
        Debug.Log("Requesting tile generation for NPC ID: " + npcId + " with description: " + tileDescription);
        // Store the callback using npcId as the key
        tileGenerationCallbacks[npcId] = callback;

        // Create a message to send to the backend
        var message = new TileGenerationRequest
        {
            type = "generate_tile",
            character_id = npcId,
            tile_description = tileDescription
        };

        var jsonString = JsonUtility.ToJson(message);
        SendWebSocketMessage(jsonString);
    }

    public void RegisterAgenticController(int characterId, AgenticController controller)
    {
        characterControllers[characterId] = controller;
    }

    public void UnregisterAgenticController(int characterId)
    {
        characterControllers.Remove(characterId);
    }

    public void RequestAgenticPlan(int characterId, Action<bool, string> callback)
    {
        // Store the callback
        planRequestCallbacks[characterId] = callback;

        string gameWorldDate = Timeline.Instance.GetFormattedDate();
        string gameWorldPlaceSetting = Timeline.Instance.place;
        List<string> waypointNames = TaskWaypoints.Instance.GetNames();
        
        // Get all NPCs and Players by tag
        List<string> npcCharacterNames = new List<string>();
        List<string> playerCharacterNames = new List<string>();

        // Find all NPCs
        GameObject[] npcs = GameObject.FindGameObjectsWithTag("NPC");
        foreach (GameObject npc in npcs)
        {
            var character = npc.GetComponent<AgenticCharacter>();
            if (character != null)
            {
                npcCharacterNames.Add(character.CharacterName);
            }
        }

        // Find all Players
        GameObject[] players = GameObject.FindGameObjectsWithTag("Player");
        foreach (GameObject player in players)
        {
            var character = player.GetComponent<AgenticCharacter>();
            if (character != null)
            {
                playerCharacterNames.Add(character.CharacterName);
            }
        }

        var jsonMessage = new CharacterPlanRequest
        {
            type = "character_planner",
            character_id = characterId,
            game_world_date = gameWorldDate,
            game_world_place_setting = gameWorldPlaceSetting,
            waypoint_names = waypointNames,
            npc_names = npcCharacterNames,
            player_names = playerCharacterNames,
        };
        var jsonString = JsonUtility.ToJson(jsonMessage);
        Debug.Log("Sending JSON: " + jsonString);
        SendWebSocketMessage(jsonString);
    }

    public void LogCharacterEvent(int characterId, string eventDescription, string eventLocation)
    {
        var jsonMessage = new
        {
            type = "character_memory_event",
            character_id = characterId,
            description = eventDescription,
            location = eventLocation
        };
        var jsonString = JsonUtility.ToJson(jsonMessage);
        SendWebSocketMessage(jsonString);
    }

    public void SendCharacterConversation(int characterId, string message)
    {
        string gameWorldDateTime = Timeline.Instance.GetFormattedDateTime();
        string gameWorldPlaceSetting = Timeline.Instance.place;
        var agenticController = characterControllers[characterId];
        
        // Get the current day plan and action from AgenticController
        string currentDayPlanOverview = JsonUtility.ToJson(agenticController.currentDayPlan);
        string currentAction = agenticController.currentDayPlanAction.action;
        string currentLocation = agenticController.currentDayPlanAction.location;

        var jsonMessage = new CharacterConversationMessage
        {
            type = "character_conversation",
            user_uuid = user_uuid,
            game_session_uuid = game_session_uuid,
            character_id = characterId,
            message = message,
            game_world_date_time = gameWorldDateTime,
            game_world_place_setting = gameWorldPlaceSetting,
            current_day_plan = currentDayPlanOverview,
            current_action = currentAction,
            current_location = currentLocation,
        };
        var jsonString = JsonUtility.ToJson(jsonMessage);
        SendWebSocketMessage(jsonString);
    }

    public void SaveCharacterMemory(int characterId, string description, string location, int priority, string userId)
    {
        var jsonMessage = new MemoryEvent
        {
            type = "memory_event",
            character_id = characterId,
            description = description,
            location = location,
            priority = priority,
            user_id = userId
        };
        var jsonString = JsonUtility.ToJson(jsonMessage);
        SendWebSocketMessage(jsonString);
    }

    public void SendImageToBackend(
        int characterId,
        byte[] imageBytes,
        string dayPlanJson,
        string dayPlanActionJson,
        string lastCompletedTaskJson,
        Action<bool> callback
    )
    {
        // Store the callback
        selfReflectionCallbacks[characterId] = callback;

        // Encode the image as Base64
        var base64Image = Convert.ToBase64String(imageBytes);

        // Create the message object with the additional lastCompletedTask
        var jsonMessage = new CharacterSelfReflectionMessage
        {
            type = "character_self_reflection",
            character_id = characterId,
            image_data = base64Image,
            day_plan = dayPlanJson,
            current_action = dayPlanActionJson,
            last_completed_task = lastCompletedTaskJson
        };

        var jsonString = JsonUtility.ToJson(jsonMessage);
        Debug.Log("Sending image JSON: " + jsonString);

        // Send the message via WebSocket
        SendWebSocketMessage(jsonString);
    }

    public void FetchAndPlaceSprite(string imageUrl, Vector3 position)
    {
        StartCoroutine(FetchAndPlaceSpriteCoroutine(imageUrl, position));
    }

    private IEnumerator FetchAndPlaceSpriteCoroutine(string imageUrl, Vector3 position)
    {
        using (UnityWebRequest webRequest = UnityWebRequest.Get(imageUrl))
        {
            // Request and wait for the desired page.
            yield return webRequest.SendWebRequest();

            // Check for network errors
            #if UNITY_2020_1_OR_NEWER
            if (webRequest.result == UnityWebRequest.Result.ConnectionError || webRequest.result == UnityWebRequest.Result.ProtocolError)
            #else
            if (webRequest.isNetworkError || webRequest.isHttpError)
            #endif
            {
                Debug.LogError("Error fetching image: " + webRequest.error);
            }
            else
            {
                // Get the base64 string from the response
                string base64String = webRequest.downloadHandler.text;

                // Convert base64 string to Texture2D
                Texture2D texture = Base64ToTexture2D(base64String);

                // Create a sprite from the Texture2D
                Sprite sprite = TextureToSprite(texture);

                // Place the sprite in the world
                PlaceSpriteInWorld(sprite, position);
            }
        }
    }

    private Texture2D Base64ToTexture2D(string base64String)
    {
        byte[] imageBytes = System.Convert.FromBase64String(base64String);
        Texture2D texture = new Texture2D(2, 2);
        texture.LoadImage(imageBytes);
        return texture;
    }

    private Sprite TextureToSprite(Texture2D texture)
    {
        return Sprite.Create(
            texture,
            new Rect(0, 0, texture.width, texture.height),
            new Vector2(0.5f, 0.5f));
    }

    public void PlaceSpriteInWorld(Sprite sprite, Vector3 position)
    {
        GameObject spriteObject = new GameObject("PlacedSprite");
        SpriteRenderer spriteRenderer = spriteObject.AddComponent<SpriteRenderer>();
        spriteRenderer.sprite = sprite;
        spriteObject.transform.position = position;
    }

    private void OnApplicationQuit()
    {
        if (websocket != null)
        {
            websocket.Close();
        }
    }

    public bool IsWebSocketReady()
    {
        if (websocket == null)
        {
            Debug.LogWarning("WebSocket is null - attempting to reinitialize");
            StartCoroutine(HandleNullWebSocket());
            return false;
        }
        
        var state = websocket.State;
        Debug.Log($"Current WebSocket state: {state}");
        
        if (state != WebSocketState.Open)
        {
            Debug.LogWarning($"WebSocket not ready (State: {state})");
            // Only trigger reconnect if we're closed and not already initializing
            if (state == WebSocketState.Closed && !isInitializing)
            {
                shouldReconnect = true;
                StartCoroutine(ReconnectWebSocket());
            }
            return false;
        }
        
        return true;
    }

    private IEnumerator HandleNullWebSocket()
    {
        if (!isInitializing)
        {
            Debug.Log("Attempting to reinitialize null WebSocket connection");
            var initTask = InitializeWebSocket();
            while (!initTask.IsCompleted)
            {
                yield return null;
            }
            
            if (initTask.Exception != null)
            {
                Debug.LogError($"Error during WebSocket reinitialization: {initTask.Exception.Message}");
            }
        }
        else
        {
            Debug.Log("WebSocket initialization already in progress");
        }
    }

    // Add this method to check connection status
    public bool IsConnected()
    {
        return websocket != null && websocket.State == WebSocketState.Open;
    }
}