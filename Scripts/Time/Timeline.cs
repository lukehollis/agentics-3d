using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEditor;
using UnityEngine.SceneManagement;
using CbAutorenTool.Tools; // For CHugeDateTime

[System.Serializable]
public struct HistoricalEvent
{
    public string name;
    public string description;
    public CHugeDateTime date;
    public string place;
    public Action eventAction;
}

public class Timeline : MonoBehaviour
{
    public string place;

    // egypt
    // public CHugeDateTime currentDate = new CHugeDateTime(-1394, 10, 4, 9, 0, 0);

    // greece
    // public CHugeDateTime currentDate = new CHugeDateTime(-229, 7, 21, 9, 0, 0);

    // rome 
    // public CHugeDateTime currentDate = new CHugeDateTime(-44, 6, 19, 9, 0, 0);

    // maya
    // public CHugeDateTime currentDate = new CHugeDateTime(628, 4, 14, 9, 0, 0);

    // hackathon 
    // public CHugeDateTime currentDate = new CHugeDateTime(2024, 9, 7, 11, 0, 0);

    // darwin
    // public CHugeDateTime currentDate = new CHugeDateTime(1835, 9, 16, 9, 0, 0);

    public CHugeDateTime currentDate;

    public float latitude = 36.8912578f;
    public float longitude = 27.2533406f;


    private float timeSinceLastUpdate = 0f;
    public float gameDayDuration = 600f; // 10 mins represents 24 hours in-game
    

    public Text dateText;

    private float originalTimeScale = 1f;
    private bool isFastForwarding = false;

    private static Timeline instance;
    public static Timeline Instance
    {
        get
        {
            if (instance == null)
            {
                instance = FindObjectOfType<Timeline>();
                if (instance == null)
                {
                    var singleton = new GameObject();
                    instance = singleton.AddComponent<Timeline>();
                    singleton.name = typeof(Timeline).ToString();
                    DontDestroyOnLoad(singleton);
                }
            }
            return instance;
        }
    }

    void Start()
    {
        SetDateBasedOnScene();
    }

    void SetDateBasedOnScene()
    {
        string sceneName = SceneManager.GetActiveScene().name;
        DateTime now = DateTime.Now;

        switch (sceneName)
        {
            case "FarmKheti":
                currentDate = new CHugeDateTime(-1394, 10, 4, 9, 0, 0);
                break;
            case "FarmLycidas":
                currentDate = new CHugeDateTime(-229, 7, 21, 9, 0, 0);
                break;
            case "FarmAurelia":
                currentDate = new CHugeDateTime(-44, 6, 19, 9, 0, 0);
                break;
            case "FarmHunahpuXbalanque":
                currentDate = new CHugeDateTime(628, 4, 14, 9, 0, 0);
                break;
            case "RomeCity":
                currentDate = new CHugeDateTime(-44, 6, 19, 9, 0, 0);
                break;
            case "DarwinIsland":
                currentDate = new CHugeDateTime(1835, 9, 16, 9, 0, 0);
                break;
            case "blank_starter":
                currentDate = new CHugeDateTime(now.Year, now.Month, now.Day, 11, now.Minute, now.Second);
                break;
            case "tree_game":
                currentDate = new CHugeDateTime(-44, now.Month, now.Day, now.Hour, now.Minute, now.Second);
                break;
            case "hackathon":
                currentDate = new CHugeDateTime(now.Year, now.Month, now.Day, 11, 0, now.Second);
                break;
            default:
                currentDate = new CHugeDateTime(now.Year, now.Month, now.Day, now.Hour, now.Minute, now.Second);
                // Debug.Log("Scene name not recognized. Date is current time.");
                break;
        }

        // if sceneName starts with "rome_" then set currentDate to the date of the first event in the rome_events list
        if (sceneName.StartsWith("rome_"))
        {
            currentDate = new CHugeDateTime(-44, 6, 19, 9, 0, 0);
        }
    }

    void Update()
    {
        timeSinceLastUpdate += Time.deltaTime;
        
        if (timeSinceLastUpdate >= 1f) 
        {
            UpdateTime(timeSinceLastUpdate);
            timeSinceLastUpdate = 0f;
        }
    }

    void UpdateTime(float secondsElapsed)
    {
        // Convert elapsed real-time seconds to in-game time, where 600 real seconds (10 minutes) = 24 in-game hours (1 day)
        float secondsPerInGameDay = gameDayDuration; // 600 real seconds for one in-game day
        float inGameSecondsPerRealSecond = 24 * 60 * 60 / secondsPerInGameDay; // Total in-game seconds in a day divided by real seconds per in-game day

        // Calculate how many in-game seconds to add based on the real-time seconds elapsed
        float inGameSecondsToAdd = secondsElapsed * inGameSecondsPerRealSecond;
        
        // Calculate in-game hours and minutes to add
        int hoursToAdd = (int)inGameSecondsToAdd / 3600;
        int minutesToAdd = ((int)inGameSecondsToAdd % 3600) / 60;

        // Update the currentDate with hours and minutes
        currentDate = currentDate.AddHours(hoursToAdd).AddMinutes(minutesToAdd);

        // CheckForEvents();
        dateText.text = GetFormattedDateTime();
    }

    // void CheckForEvents()
    // {
    //     foreach (var historicalEvent in events)
    //     {
    //         if (currentDate.Equals(historicalEvent.date)) // Assumes CHugeDateTime supports Equals with time comparison
    //         {
    //             historicalEvent.eventAction?.Invoke();
    //             Debug.Log($"Historical Event: {historicalEvent.name} - {historicalEvent.description}");
    //         }
    //     }
    // }

    public string GetFormattedDate()
    {
        string monthName = GetAbbreviatedMonthName(currentDate.Month);
        string yearSuffix = currentDate.Year < 0 ? "BCE" : "CE";
        int yearValue = (int)Math.Abs(currentDate.Year);
        return $"{currentDate.Day} {monthName} {yearValue} {yearSuffix}";
    }

    public string GetFormattedDateTime()
    {
        string monthName = GetAbbreviatedMonthName(currentDate.Month);
        string yearSuffix = currentDate.Year < 0 ? "BCE" : "CE";
        int yearValue = (int)Math.Abs(currentDate.Year);
        // Include time in the format
        return $"{currentDate.Day} {monthName} {yearValue} {yearSuffix} {currentDate.Hour:0}:{currentDate.Minute:00}";
    }

    private string GetMonthName(int monthNumber)
    {
        string[] monthNames = {
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        };

        if (monthNumber >= 1 && monthNumber <= 12)
        {
            return monthNames[monthNumber - 1];
        }
        else
        {
            Debug.LogError("Invalid month number: " + monthNumber);
            return string.Empty;
        }
    }
    private string GetAbbreviatedMonthName(int monthNumber)
    {
        string[] monthNames = {
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"
        };

        if (monthNumber >= 1 && monthNumber <= 12)
        {
            return monthNames[monthNumber - 1];
        }
        else
        {
            Debug.LogError("Invalid month number: " + monthNumber);
            return string.Empty;
        }
    }
#if UNITY_EDITOR
    // private CHugeDateTime CHugeDateTimeField(string label, CHugeDateTime dateTime)
    // {
    //     // EditorGUILayout.BeginHorizontal();
    //     // EditorGUILayout.LabelField(label, GUILayout.Width(40));

    //     // long year = EditorGUILayout.LongField(dateTime.Year, GUILayout.Width(60));
    //     // int month = EditorGUILayout.IntField(dateTime.Month, GUILayout.Width(30));
    //     // int day = EditorGUILayout.IntField(dateTime.Day, GUILayout.Width(30));
    //     // int hour = EditorGUILayout.IntField(dateTime.Hour, GUILayout.Width(30));
    //     // int minute = EditorGUILayout.IntField(dateTime.Minute, GUILayout.Width(30));

    //     // EditorGUILayout.EndHorizontal();

    //     // // Create a new CHugeDateTime instance with the modified values
    //     // return new CHugeDateTime(year, month, day, hour, minute, 0);
    // }
#endif

   public void StartFastForwarding()
    {
        originalTimeScale = Time.timeScale;
        Time.timeScale = 30f; // Fast-forward time, adjust this value as needed
        isFastForwarding = true;
        StartCoroutine(WaitUntilMorning());
    }

    public void StopFastForwarding()
    {
        Time.timeScale = 1f;
        isFastForwarding = false;
    }


    private IEnumerator WaitUntilMorning()
    {
        Debug.Log(currentDate.Hour);
        while (currentDate.Hour >= 19 || currentDate.Hour < 6)
        {
            yield return null; // Wait until the next frame
        }

        StopFastForwarding();
    }

    private void OnDestroy()
    {
        if (isFastForwarding)
        {
            StopFastForwarding();
        }
    }
}
