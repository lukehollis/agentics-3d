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
    private readonly string websocketUrl = "ws://civs.local:8000/ws/characters/";
    // private readonly string websocketUrl = "wss://civs.mused.com/ws/characters/";
    public event Action OnWebSocketConnected;

    private Dictionary<int, AgenticController> characterControllers = new Dictionary<int, AgenticController>();
    private string user_uuid;
    private string game_session_uuid; // Add this line

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
        game_session_uuid = Guid.NewGuid().ToString();

        websocket = new WebSocket(websocketUrl);

        websocket.OnOpen += () =>
        {
            Debug.Log("WebSocket connection opened");
            OnWebSocketConnected?.Invoke();
        };

        websocket.OnError += (e) =>
        {
            Debug.Log("WebSocket error: " + e);
        };

        websocket.OnClose += (e) =>
        {
            Debug.Log("WebSocket connection closed");
        };

        websocket.OnMessage += (bytes) =>
        {
            var message = System.Text.Encoding.UTF8.GetString(bytes);
            OnMessageReceived(bytes);
        };

        // InvokeRepeating("SendWebSocketMessage", 0.0f, 0.3f);

        await websocket.Connect();
    }

    void Update()
    {
        #if !UNITY_WEBGL || UNITY_EDITOR
            if (websocket != null && websocket.State == WebSocketState.Open)
            {
                websocket.DispatchMessageQueue();
            }
        #endif
    }

    async void SendWebSocketMessage()
    {
    if (websocket.State == WebSocketState.Open)
    {
      // Sending bytes
      await websocket.Send(new byte[] { 10, 20, 30 });

      // Sending plain text
      await websocket.SendText("plain text message");
    }
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
        try
        {
            var receivedData = JsonUtility.FromJson<ReceivedMessageData>(message);
            var messageType = receivedData.type;
            var characterId = receivedData.character_id;
            Debug.Log("Received message type: " + messageType + " for character ID: " + characterId);

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
                // Handle the streamed plan response
                Debug.Log("Received plan response: " + receivedData.message);

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
        }
        catch (Exception ex)
        {
            Debug.LogError("Error deserializing WebSocket message: " + ex.Message);
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

    public void RegisterCharacterController(int characterId, AgenticController controller)
    {
        characterControllers[characterId] = controller;
    }

    public void UnregisterCharacterController(int characterId)
    {
        characterControllers.Remove(characterId);
    }

    public void RequestCharacterPlan(int characterId)
    {
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
        websocket.Close();
    }
}