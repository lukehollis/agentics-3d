using UnityEngine;
using TMPro;
using UnityEngine.UI;
using System.Collections;
using Agentics;


public class SleepingSpot : MonoBehaviour, Interactable
{
    [Header("UI References")]
    public GameObject sleepPromptPanel;
    public Button confirmButton;
    public Button cancelButton;
    
    private void Start()
    {
        // Ensure the prompt is hidden at start
        if (sleepPromptPanel != null)
            sleepPromptPanel.SetActive(false);
            
        // Setup button listeners
        if (confirmButton != null)
            confirmButton.onClick.AddListener(ConfirmSleep);
        if (cancelButton != null)
            cancelButton.onClick.AddListener(CancelSleep);
    }

    public void Interact()
    {
        ShowSleepPrompt();
    }

    private void ShowSleepPrompt()
    {
        if (sleepPromptPanel != null)
        {
            sleepPromptPanel.SetActive(true);
        }
    }

    private void ConfirmSleep()
    {
        // Hide the prompt
        sleepPromptPanel.SetActive(false);
        
        // Start the sleep sequence
        GameController.Instance.StartNighttime();
    }

    private void CancelSleep()
    {
        // Hide the prompt and return to free roam
        sleepPromptPanel.SetActive(false);
        GameController.Instance.SetState(GameState.FreeRoam);
    }

    private void OnDestroy()
    {
        // Clean up button listeners
        if (confirmButton != null)
            confirmButton.onClick.RemoveListener(ConfirmSleep);
        if (cancelButton != null)
            cancelButton.onClick.RemoveListener(CancelSleep);
    }
}