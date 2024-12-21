using UnityEngine;
using UnityEngine.Networking;
using System;
using System.Collections;
using System.Threading.Tasks;

public class AgenticsEditorService
{
    private string baseUrl;

    public AgenticsEditorService(string url)
    {
        baseUrl = url.TrimEnd('/');
    }

    public class SpawnBounds
    {
        public float minX;
        public float maxX;
        public float minZ;
        public float maxZ;
        public float y;
    }

    [Serializable]
    public class WorldData
    {
        public string name;
        public SpawnBounds bounds;
        // Add other world properties as needed
    }

    public async Task<bool> TestConnection()
    {
        try
        {
            using (UnityWebRequest request = UnityWebRequest.Get($"{baseUrl}/api/health/"))
            {
                var operation = request.SendWebRequest();
                while (!operation.isDone)
                    await Task.Yield();

                if (request.result == UnityWebRequest.Result.Success)
                {
                    return true;
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Connection test failed: {e.Message}");
        }
        return false;
    }

    public async Task<WorldData> GetWorldData(int worldId)
    {
        try
        {
            using (UnityWebRequest request = UnityWebRequest.Get($"{baseUrl}/api/worlds/{worldId}/"))
            {
                var operation = request.SendWebRequest();
                while (!operation.isDone)
                    await Task.Yield();

                if (request.result == UnityWebRequest.Result.Success)
                {
                    string jsonResponse = request.downloadHandler.text;
                    return JsonUtility.FromJson<WorldData>(jsonResponse);
                }
                else
                {
                    Debug.LogError($"Failed to get world data: {request.error}");
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Error getting world data: {e.Message}");
        }
        return null;
    }
}