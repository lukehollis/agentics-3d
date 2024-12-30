using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro; // Import TextMeshPro namespace
using System;
using Agentics;


public enum DialogState
{
    NPCSpeaking,
    PlayerInput,
    AIGeneratingResponse
}

public class DialogManager : MonoBehaviour
{
    [SerializeField] public GameObject dialogBox;
    [SerializeField] public GameObject dialogResponseBox;
    [SerializeField] public GameObject dialogAdvanceArrow;
    [SerializeField] public TMP_Text npcNameLabel;
    [SerializeField] public Image npcAvatarImage;
    [SerializeField] TMP_Text dialogText;

    [SerializeField] int lettersPerSecond;

    public event Action OnShowDialog;
    public event Action OnHideDialog;

    public static DialogManager Instance { get; private set; }

    public DialogState dialogState { get; private set; }

    [SerializeField] TMP_InputField playerInputField; 

    private bool isTyping = false;
    private int npcId;
    private AgenticController currentNPCController;
    private Coroutine currentTypingCoroutine;

    private void Awake()
    {
        Instance = this;
        playerInputField.onEndEdit.AddListener(HandleEndEdit);
    }

    Dialog dialog;
    int currentLine = 0;

    public void HandleUpdate() 
    {
        if (Input.GetMouseButtonDown(0))
        {
            if (dialogState == DialogState.NPCSpeaking) {
                if (!isTyping) {
                    AdvanceDialog();
                }
            } else if (dialogState == DialogState.PlayerInput){

                // click should not submit because user has to click the text field
                // HandlePlayerInput();
            }
        }
        if (Input.GetKeyDown(KeyCode.Return))
        {

            // If the dialog state is PlayerInput, handle the player's input
            if (dialogState == DialogState.PlayerInput)
            {

                // Handle the player's input
                DialogManager.Instance.HandlePlayerInput();
            }
        }
    }

    private void HandleEndEdit(string text)
    {
        if (dialogState == DialogState.PlayerInput)
        {
            // Check if the input field is focused to ensure it only submits when the user is done typing
            if (!playerInputField.isFocused)
            {
                HandlePlayerInput();
            }
        }
    }

    public void AdvanceDialog() 
    {
        if (!isTyping && dialogState == DialogState.NPCSpeaking)
        {
            // advance to the next line in the dialog
            currentLine++;

            // if there are more lines in the dialog, type the next line
            if (currentLine < dialog.Lines.Count)
            {
                dialogText.text = "";
                StartCoroutine(TypeDialog(dialog.Lines[currentLine]));
            }
            else
            {
                currentLine = 0;
                dialogText.text = "";
                
                // Check if game is offline
                if (GameController.Instance.isOfflineMode)
                {
                    // In offline mode, just hide the dialog
                    HideDialog();
                }
                else
                {
                    // Online mode - show player input as before
                    dialogState = DialogState.PlayerInput;
                    dialogResponseBox.SetActive(true);
                    dialogBox.SetActive(false);
                    playerInputField.Select();
                    playerInputField.ActivateInputField();
                    dialogAdvanceArrow.SetActive(false);
                }
            }
        }
    }

    public IEnumerator ShowDialog(Dialog dialog, int npcId, string npcName, Sprite npcAvatar=null, bool npcSpeakingFirst=false) 
    {
        this.npcId = npcId;

        // set name and avatar for the npc in dialog box
        npcNameLabel.text = npcName;
        npcAvatarImage.sprite = npcAvatar;

        yield return new WaitForEndOfFrame();

        OnShowDialog?.Invoke();

        this.dialog = dialog;
        dialogBox.SetActive(true);
        currentLine = 0;
        dialogText.text = "";

        // Find the AgenticController based on npcId
        currentNPCController = FindNPCControllerById(npcId);
        if (currentNPCController != null)
        {
            currentNPCController.SetDialogState(true);
        }

        // if there is dialog to display, display it, otherwise, prompt for user to start
        // the conversation
        if (dialog.Lines.Count > 0) {
            dialogState = DialogState.NPCSpeaking;
            currentTypingCoroutine = StartCoroutine(TypeDialog(dialog.Lines[0]));
            dialogAdvanceArrow.SetActive(true);
        } else if (npcSpeakingFirst) {
            dialogState = DialogState.AIGeneratingResponse;
            dialogResponseBox.SetActive(false);
            dialogAdvanceArrow.SetActive(true);
        } else {
            dialogState = DialogState.PlayerInput;
            dialogBox.SetActive(false);
            dialogResponseBox.SetActive(true);  // Show the player input field
            playerInputField.Select(); // Focus the input field
            playerInputField.ActivateInputField(); // Activate the input field to allow typing
            dialogAdvanceArrow.SetActive(false);
        }
    }

    // New method for NPC-initiated conversation
    public IEnumerator ShowDialogFromNPC(string initialText, int npcId)
    {
        this.npcId = npcId;
        yield return new WaitForEndOfFrame();

        OnShowDialog?.Invoke();

        // Find the AgenticController based on npcId
        currentNPCController = FindNPCControllerById(npcId);
        if (currentNPCController != null)
        {
            currentNPCController.SetDialogState(true);
        }

        dialogBox.SetActive(true);
        currentLine = 0;
        dialogText.text = "";

        dialogState = DialogState.NPCSpeaking;
        currentTypingCoroutine = StartCoroutine(TypeDialog(initialText));
        dialogAdvanceArrow.SetActive(true);
    }

    public void HideDialog() 
    {
        dialogBox.SetActive(false);
        dialogState = DialogState.NPCSpeaking;
        currentLine = 0;
        dialogText.text = "";
        dialog.Lines.Clear();
        OnHideDialog?.Invoke();
        dialogResponseBox.SetActive(false);

        // Reset the dialog state of the NPC
        if (currentNPCController != null)
        {
            currentNPCController.SetDialogState(false);
        }

        // Reset the player's dialog state
        GameController.Instance.player.SetDialogState(false);
    }

    public IEnumerator TypeDialog(string line)
    {
        // Stop any existing typing coroutine
        if (currentTypingCoroutine != null)
        {
            StopCoroutine(currentTypingCoroutine);
            isTyping = false;
        }

        isTyping = true;
        dialogText.text = "";
        foreach (var letter in line.ToCharArray()) 
        {
            dialogText.text += letter;
            yield return new WaitForSeconds(1f/lettersPerSecond);
        }
        isTyping = false;
        currentTypingCoroutine = null;
    }

    public void HandlePlayerInput()
    {
        string input = playerInputField.text;

        if (input.Length > 0)
        {
            dialogResponseBox.SetActive(false);
            playerInputField.text = "";

            // Set the dialog state to AIGeneratingResponse
            dialogState = DialogState.AIGeneratingResponse;

            // Start the loading animation
            StartCoroutine(ShowLoadingAnimation());

            // Add more detailed logging
            if (NetworkingController.Instance == null)
            {
                Debug.LogError("NetworkingController.Instance is null");
                return;
            }

            var websocketState = NetworkingController.Instance.IsWebSocketReady();
            Debug.Log($"WebSocket State: {websocketState}");

            if (websocketState)
            {
                // Send the player's input and get the response
                NetworkingController.Instance.SendCharacterConversation(npcId, input);

                // clear the previous dialog.
                dialog.Lines.Clear();
                dialogText.text = "";
                currentLine = 0;
                dialogBox.SetActive(true);
            }
            else
            {
                // Handle the case where WebSocket is not ready
                Debug.LogWarning($"Cannot send message: WebSocket is not ready. Current state: {websocketState}");
                dialog.Lines.Clear();
                dialogText.text = "";
                currentLine = 0;
                dialogBox.SetActive(true);

                DisplayResponse("Sorry, I'm not feeling myself right now...");
            }
        }
    }


    public void DisplayResponse(string response)
    {
        var chunkLength = 140;

        if (dialog.Lines.Count > 0) {
            if (dialog.Lines[dialog.Lines.Count - 1].Length < chunkLength) {
                dialog.Lines[dialog.Lines.Count - 1] += response;

                // update the display also 
                if ((dialog.Lines.Count - 1) == currentLine) {
                    dialogText.text = dialog.Lines[dialog.Lines.Count - 1];
                }
            }
            else 
            {
                dialog.Lines.Add(response);
            }

            dialogAdvanceArrow.SetActive(true);
        } 
        else 
        {
            // first response back from api
            // Set the dialog state to NPCSpeaking regardless of the response
            dialogState = DialogState.NPCSpeaking;

            // Stop the loading animation
            StopCoroutine(ShowLoadingAnimation());

            // start new line in dialog
            dialog.Lines.Add(response);

            // set the visible text in the dialog window
            dialogText.text = response;
        }

    }

    private IEnumerator ShowLoadingAnimation()
    {
        while (dialogState == DialogState.AIGeneratingResponse)
        {
            // these extra checks are needed because of the yield return waitforseconds
            // (it could replace dialog text)
            if (dialogState == DialogState.AIGeneratingResponse)
            {
                dialogText.text = ".";
            }
            yield return new WaitForSeconds(0.5f);
            if (dialogState == DialogState.AIGeneratingResponse)
            {
                dialogText.text = "..";
            }
            yield return new WaitForSeconds(0.5f);
            if (dialogState == DialogState.AIGeneratingResponse)
            {
                dialogText.text = "...";
            }
            yield return new WaitForSeconds(0.5f);
        }
    }

    private AgenticController FindNPCControllerById(int id)
    {
        // Find the AgenticController instance by NPC ID
        foreach (AgenticController npc in FindObjectsOfType<AgenticController>())
        {
            if (npc.character.ID == id)
            {
                return npc;
            }
        }
        return null;
    }
}