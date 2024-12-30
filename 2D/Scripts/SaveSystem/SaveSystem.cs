using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.SceneManagement;

public class SaveSystem : MonoBehaviour
{
    private static SaveSystem instance;
    public static SaveSystem Instance
    {
        get
        {
            if (instance == null)
            {
                instance = FindObjectOfType<SaveSystem>();
                if (instance == null)
                {
                    var singleton = new GameObject();
                    instance = singleton.AddComponent<SaveSystem>();
                    singleton.name = typeof(SaveSystem).ToString();
                    DontDestroyOnLoad(singleton);
                }
            }
            return instance;
        }
    }

    private SaveData currentData = new SaveData();
    private SaveData? pendingSaveData = null;
    public bool HasPendingSave => pendingSaveData.HasValue;
    
    [System.Serializable]
    public struct SaveData
    {
        public PlayerData PlayerData;
        public DayCycleHandlerSaveData TimeSaveData;
        public EnvironmentData EnvironmentData;
        public string SceneName;
    }

    public void Save()
    {
        // Create player save data
        currentData.PlayerData = new PlayerData
        {
            Position = GameController.Instance.player.transform.position,
            Money = GameController.Instance.player.Money,
            Health = GameController.Instance.player.Health,
            Stamina = GameController.Instance.player.Stamina
        };

        // Save time data through GameController
        currentData.TimeSaveData = GameController.Instance.dayCycleHandler.Serialize();

        currentData.SceneName = SceneManager.GetActiveScene().name;
        currentData.EnvironmentData = GameController.Instance.EnvironmentManager.Serialize();

        // Convert save data to JSON
        string jsonData = JsonUtility.ToJson(currentData);

        // Send save data through NetworkingController
        NetworkingController.Instance.SaveGameData(jsonData);
    }

    public void Load(string jsonData)
    {
        currentData = JsonUtility.FromJson<SaveData>(jsonData);
        pendingSaveData = currentData;
    }

    public void ApplyPendingSave()
    {
        if (!pendingSaveData.HasValue)
        {
            Debug.LogWarning("No pending save data to apply");
            return;
        }

        // Apply player data
        GameController.Instance.player.transform.position = pendingSaveData.Value.PlayerData.Position;
        GameController.Instance.player.Money = pendingSaveData.Value.PlayerData.Money;
        GameController.Instance.player.Health = pendingSaveData.Value.PlayerData.Health;
        GameController.Instance.player.Stamina = pendingSaveData.Value.PlayerData.Stamina;

        // Load environment data
        GameController.Instance.EnvironmentManager.Load(pendingSaveData.Value.EnvironmentData);

        // Clear pending save after applying
        pendingSaveData = null;
    }
}