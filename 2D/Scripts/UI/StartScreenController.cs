using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;
using TMPro;

public class StartScreenController : MonoBehaviour
{
    public Image blackBackgroundImage;
    public TMP_Text loadingMessageText;
    public GameObject startScreen;

    private bool showingOfflineMessage = false;

    private string[] loadingMessages = new string[]
    {
        "...Consulting Cato...",
        "...Reading the Georgics...",
        "...Checking the Fasti...",
        "...Preparing for the Ambarvalia...",
        "...Consulting the calendar...",
        "...Awaiting favorable omens...",
        "...Making offerings to Ceres...",
        "...Observing the sacred birds...",
        "...Calibrating the gnomon...",
        "...Consulting the haruspex...",
        "...Preparing the sacrificial grain...",
        "...Checking the Pleiades...",
        "...Listening to the shepherds' songs...",
        "...Consulting with the Pontifex...",
        "...Examining the entrails...",
        "...Preparing the ritual flour...",
        "...Aligning the sundial...",
        "...Consulting Varro...",
        "...Reading Columella...",
        "...Preparing libations...",
        "...Consulting the vilicus...",
        "...Reviewing Pliny...",
        "...Preparing the mola...",
        "...Propitiating the Lares...",
        "...Interpreting lunar phases..."
    };

    private Coroutine loadingMessagesCoroutine;

    private void Start()
    {
        startScreen.SetActive(true);
        StartCoroutine(FadeAndMoveImages());
        loadingMessagesCoroutine = StartCoroutine(DisplayLoadingMessages());
    }

    private IEnumerator FadeAndMoveImages()
    {
        blackBackgroundImage.gameObject.SetActive(true);
        float duration = 0.6f;
        for (float t = 0; t < duration; t += Time.deltaTime)
        {
            float normalizedTime = t / duration;
            blackBackgroundImage.color = new Color(blackBackgroundImage.color.r, blackBackgroundImage.color.g, blackBackgroundImage.color.b, 1.0f - normalizedTime);
            yield return null;
        }
        blackBackgroundImage.color = new Color(blackBackgroundImage.color.r, blackBackgroundImage.color.g, blackBackgroundImage.color.b, 0);
        blackBackgroundImage.gameObject.SetActive(false);
    }


    private IEnumerator DisplayLoadingMessages()
    {
        float charactersPerSecond = 12f;
        float delayBetweenCharacters = 1f / charactersPerSecond;
        List<string> remainingMessages = new List<string>(loadingMessages);
        
        while (remainingMessages.Count > 0 && !showingOfflineMessage)
        {
            // Get random message
            int randomIndex = UnityEngine.Random.Range(0, remainingMessages.Count);
            string message = remainingMessages[randomIndex];
            remainingMessages.RemoveAt(randomIndex);
            
            loadingMessageText.text = "";
            // Type out each character
            foreach (char c in message)
            {
                if (showingOfflineMessage) yield break;
                loadingMessageText.text += c;
                yield return new WaitForSeconds(delayBetweenCharacters);
            }

            // Wait a bit before starting the next message
            yield return new WaitForSeconds(1f);
        }
    }


    public void StartGame()
    {
        // stop the loading messages and start fade out
        if (loadingMessagesCoroutine != null)
        {
            StopCoroutine(loadingMessagesCoroutine);
        }
        GameController.Instance.SetHasStarted(true);
        StartCoroutine(FadeOutStartScreen());
    }

    public void ShowOfflineMessage()
    {
        showingOfflineMessage = true;

        if (loadingMessageText != null)
        {
            if (loadingMessagesCoroutine != null)
            {
                StopCoroutine(loadingMessagesCoroutine);
            }
            loadingMessageText.text = "Unable to connect to the ancient digital worlds, playing offline...";
            StartCoroutine(StartGameAfterDelay());
        }
    }

    public IEnumerator FadeOutStartScreen()
    {
        // Get the Canvas Group or add one if it doesn't exist
        CanvasGroup canvasGroup = startScreen.GetComponent<CanvasGroup>();
        if (canvasGroup == null)
        {
            canvasGroup = startScreen.AddComponent<CanvasGroup>();
        }

        // Stop the loading messages coroutine if it's running
        if (loadingMessagesCoroutine != null)
        {
            StopCoroutine(loadingMessagesCoroutine);
        }

        float duration = 1f;
        float elapsedTime = 0f;

        // Fade out the start screen
        while (elapsedTime < duration)
        {
            elapsedTime += Time.deltaTime;
            canvasGroup.alpha = Mathf.Lerp(1f, 0f, elapsedTime / duration);
            yield return null;
        }

        // After fully faded out, deactivate the screen and clean up
        startScreen.SetActive(false);
        loadingMessageText.text = "";
        showingOfflineMessage = false;
    }

    private IEnumerator StartGameAfterDelay()
    {
        yield return new WaitForSeconds(3f); // Give players time to read the message
        StartGame();
    }
}