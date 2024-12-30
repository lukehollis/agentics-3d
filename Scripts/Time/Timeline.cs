using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEditor;
using UnityEngine.SceneManagement;
using CbAutorenTool.Tools; // For CHugeDateTime
using TMPro;

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
    public CHugeDateTime currentDate;

    [Header("Simulation Settings")]
    public string place;
    public float gameDayDuration = 600f; // 10 mins represents 24 hours in-game

    [Header("UI")]
    public TMP_Text dateText;

    private float timeSinceLastUpdate = 0f;
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
        currentDate = new CHugeDateTime(DateTime.Now.Year, DateTime.Now.Month, DateTime.Now.Day, 
            DateTime.Now.Hour, DateTime.Now.Minute, DateTime.Now.Second);
        SetDateBasedOnScene();
    }

    void SetDateBasedOnScene()
    {
        string sceneName = SceneManager.GetActiveScene().name;
        DateTime now = DateTime.Now;

        switch (sceneName)
        {
            case "dev":
                currentDate = new CHugeDateTime(now.Year, now.Month, now.Day, 11, now.Minute, now.Second);
                break;
            case "BART":
                currentDate = new CHugeDateTime(2024, 12, 11, 11, 0, 0);
                break;
            default:
                currentDate = new CHugeDateTime(now.Year, now.Month, now.Day, now.Hour, now.Minute, now.Second);
                break;
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
