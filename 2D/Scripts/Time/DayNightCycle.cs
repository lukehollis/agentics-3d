using UnityEngine;
 // For Light2D
using TMPro;

public class DayNightCycle : MonoBehaviour
{
    public UnityEngine.Rendering.Universal.Light2D globalLight;
    public DayNightAudioManager audioManager;
    public bool lightsActive;

    [Header("Lights that are on during the day")]
    [SerializeField]
    public UnityEngine.Rendering.Universal.Light2D[] dayLights;

    [Header("Lights that are on during the night")]
    [SerializeField]
    public GameObject[] nightLights;


    // Define the color and intensity settings for different times of day
    private float nightIntensity = 0.1f;
    private Color nightColor = new Color(0x56 / 255f, 0x62 / 255f, 1.0f);

    private float dayIntensity = 1.2f;
    private Color dayColor = new Color(0xFD / 255f, 0xF2 / 255f, 0xD8 / 255f);

    private Color setRiseColor = new Color(0xFF / 255f, 0xC7 / 255f, 0x34 / 255f);

    private bool isDayTime = true;
    private bool npcsAsleep = false;

    private Timeline timeline => GameController.Instance.timeline;

    private void Start()
    {
        if (globalLight == null)
        {
            globalLight = GetComponent<UnityEngine.Rendering.Universal.Light2D>();
        }
    }

    private void Update()
    {
        // Update the light settings based on the time of day
        ControlLight();
        CheckDailyRoutines();
    }

    public void CheckDailyRoutines()
    {
        int hours = timeline.currentDate.Hour;
        int mins = timeline.currentDate.Minute;

        if (hours == 21 && !npcsAsleep)
        {
            // NPCCharacters.Instance.MakeNPCsSleep();
            // npcsAsleep = true;
        }

        if (hours == 6 && npcsAsleep) // 6:00 AM
        {
            // NPCCharacters.Instance.WakeUpNPCs();
            // npcsAsleep = false;

            // // if player is asleep also, wake up
            // PlayerCharacters.Instance.WakeUpPlayers();
        }

    }

    public void ControlLight()
    {
        int hours = timeline.currentDate.Hour;
        int mins = timeline.currentDate.Minute;

        float t = 0f;

        if (hours >= 19 && hours < 22)
        {
            t = (hours - 19) / 3f + mins / 180f;
            globalLight.color = Color.Lerp(dayColor, nightColor, t);
            globalLight.intensity = Mathf.Lerp(dayIntensity, nightIntensity, t);
            foreach (var light in dayLights) // Add this block
            {
                light.intensity = Mathf.Lerp(dayIntensity, nightIntensity, t);
            }
            if (hours == 19 && isDayTime)
            {
                isDayTime = false;
                audioManager.SetDayTime(false); // Trigger night music
            }
        }
        else if (hours >= 6 && hours < 9)
        {
            t = (hours - 6) / 3f + mins / 180f;
            globalLight.color = Color.Lerp(nightColor, dayColor, t);
            globalLight.intensity = Mathf.Lerp(nightIntensity, dayIntensity, t);
            foreach (var light in dayLights) // Add this block
            {
                light.intensity = Mathf.Lerp(nightIntensity, dayIntensity, t);
            }
            if (hours == 6 && !isDayTime)
            {
                isDayTime = true;
                audioManager.SetDayTime(true); // Trigger day music
            }
        }
        else if (hours >= 9 && hours < 19)
        {
            globalLight.color = dayColor;
            globalLight.intensity = dayIntensity;
            foreach (var light in dayLights) // Add this block
            {
                light.intensity = dayIntensity;
            }
            if (!isDayTime)
            {
                isDayTime = true;
                audioManager.SetDayTime(true); // Ensure day music is playing
            }
        }
        else
        {
            globalLight.color = nightColor;
            globalLight.intensity = nightIntensity;
            foreach (var light in dayLights) // Add this block
            {
                light.intensity = nightIntensity;
            }
            if (isDayTime)
            {
                isDayTime = false;
                audioManager.SetDayTime(false); // Ensure night music is playing
            }
        }

        if (hours >= 19 || hours < 9)
        {
            if (!lightsActive)
            {
                for (int i = 0; i < nightLights.Length; i++)
                {
                    nightLights[i].SetActive(true);
                }
                lightsActive = true;
            }
        }
        else
        {
            if (lightsActive)
            {
                for (int i = 0; i < nightLights.Length; i++)
                {
                    nightLights[i].SetActive(false);
                }
                lightsActive = false;
            }
        }
    }

    public DayCycleHandlerSaveData Serialize()
    {
        return new DayCycleHandlerSaveData
        {
            CurrentDate = timeline.currentDate,
            IsDayTime = isDayTime,
            LightsActive = lightsActive
        };
    }

    public void Load(DayCycleHandlerSaveData data)
    {
        timeline.currentDate = data.CurrentDate;
        isDayTime = data.IsDayTime;
        lightsActive = data.LightsActive;
        
        // Force immediate visual update
        ControlLight();
    }
}