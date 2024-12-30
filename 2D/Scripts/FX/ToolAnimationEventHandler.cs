using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.VFX;

/// <summary>
/// The tool animation can call the different events at the right frame to trigger the set VFX
/// This script need to be added on the same GameObject wit the Animator on the tool to be able to receive the
/// animation events.
/// </summary>
public class ToolAnimationEventHandler : MonoBehaviour
{
    [Header("Front")]
    public VisualEffect FrontEffect;
    public string FrontEffectId;

    [Header("Up")]
    public VisualEffect UpEffect;
    public string UpEffectId;

    [Header("Side")]
    public VisualEffect SideEffect;
    public string SideEffectId;

    public void TriggerFrontVFX()
    {
        SideEffect.gameObject.SetActive(false);
        UpEffect.gameObject.SetActive(false);
        FrontEffect.gameObject.SetActive(true);
    
        FrontEffect.SendEvent(FrontEffectId);
    }

    public void TriggerSideVFX()
    {
        SideEffect.gameObject.SetActive(true);
        UpEffect.gameObject.SetActive(false);
        FrontEffect.gameObject.SetActive(false);
    
        SideEffect.SendEvent(SideEffectId);
    }

    public void TriggerUpVFX()
    {
        SideEffect.gameObject.SetActive(false);
        UpEffect.gameObject.SetActive(true);
        FrontEffect.gameObject.SetActive(false);
    
        UpEffect.SendEvent(UpEffectId);
    }
}