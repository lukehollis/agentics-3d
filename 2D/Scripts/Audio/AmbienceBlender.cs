using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Handle blending between a Day and Night ambience audio clip
/// </summary>
public class AmbienceBlender : MonoBehaviour
{
    enum State
    {
        BlendToNight,
        BlendToDay,
        Playing
    }
    
    public AudioSource DayAmbienceSource;
    public AudioSource NightAmbienceSource;

    private State m_CurrentState;
    private float m_CurrentBlendRatio = 0.0f;
    
    private void Start()
    {
        DayAmbienceSource.volume = 0.0f;
        NightAmbienceSource.volume = 0.0f;

        m_CurrentState = State.Playing;
    }

    private void Update()
    {
        if (m_CurrentState != State.Playing)
        {
            bool isFinished = AdvanceBlending();
            switch (m_CurrentState)
            {
                case State.BlendToDay :
                    DayAmbienceSource.volume = m_CurrentBlendRatio;
                    NightAmbienceSource.volume = 1.0f - m_CurrentBlendRatio;
                    break;
                case State.BlendToNight :
                    NightAmbienceSource.volume = m_CurrentBlendRatio;
                    DayAmbienceSource.volume = 1.0f - m_CurrentBlendRatio;
                    break;
            }

            if (isFinished)
            {
                m_CurrentState = State.Playing;
            }
        }
    }

    bool AdvanceBlending()
    {
        m_CurrentBlendRatio = Mathf.Clamp01(m_CurrentBlendRatio + Time.deltaTime);
        return Mathf.Approximately(m_CurrentBlendRatio, 1.0f);
    }

    //Call this from an event in the editor or another script to start blending to the day ambience
    public void BlendToDay()
    {
        m_CurrentState = State.BlendToDay;
        m_CurrentBlendRatio = 0.0f;
    }

    //Call this from an event in the editor or another script to start blending to the night ambience
    public void BlendToNight()
    {
        m_CurrentState = State.BlendToNight;
        m_CurrentBlendRatio = 0.0f;
    }
}