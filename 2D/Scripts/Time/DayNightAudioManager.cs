using System.Collections;
using UnityEngine;

public class DayNightAudioManager : MonoBehaviour
{
    public AudioSource dayMusic;
    public AudioSource nightMusic;
    public AudioSource dayAmbience; // New
    public AudioSource nightAmbience; // New
    public float crossfadeDuration = 2.0f; // Duration of the crossfade
    [Range(0f, 1f)]
    public float maxBackgroundVolume = 0.66f; // New configurable max volume

    private bool isDayTime = true;

    void Start()
    {
        // Ensure both audio sources are set up correctly
        dayMusic.loop = true;
        nightMusic.loop = true;
        dayAmbience.loop = true; 
        nightAmbience.loop = true; 
        
        // Start both day and night audio sources playing
        dayMusic.Play();
        dayAmbience.Play();
        nightMusic.Play();
        nightAmbience.Play();
        
        // Get current time from Timeline singleton
        int currentHour = 9; // This should be replaced with actual time
        bool isDay = currentHour >= 6 && currentHour < 19;
        
        // Set initial volumes based on time of day
        dayMusic.volume = isDay ? maxBackgroundVolume : 0f;
        dayAmbience.volume = isDay ? maxBackgroundVolume : 0f;
        nightMusic.volume = isDay ? 0f : maxBackgroundVolume;
        nightAmbience.volume = isDay ? 0f : maxBackgroundVolume;
        
        // Set the initial state
        isDayTime = isDay;
    }

    void Update()
    {
    }

    public void SetDayTime(bool isDay)
    {
        // don't start a crossfade if already crossfading
        if (isDay && !isDayTime)
        {
            StartCoroutine(Crossfade(dayMusic, nightMusic));
            StartCoroutine(Crossfade(dayAmbience, nightAmbience)); // New
        }
        else
        {
            StartCoroutine(Crossfade(nightMusic, dayMusic));
            StartCoroutine(Crossfade(nightAmbience, dayAmbience)); // New
        }

        isDayTime = isDay;
    }

    private IEnumerator Crossfade(AudioSource fadeIn, AudioSource fadeOut)
    {
        float timeElapsed = 0f;

        while (timeElapsed < crossfadeDuration)
        {
            float t = timeElapsed / crossfadeDuration;
            fadeIn.volume = Mathf.Lerp(0f, maxBackgroundVolume, t);
            fadeOut.volume = Mathf.Lerp(maxBackgroundVolume, 0f, t);
            timeElapsed += Time.deltaTime;
            yield return null;
        }

        fadeIn.volume = maxBackgroundVolume;
        fadeOut.volume = 0f;
    }
}