using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UIElements;

public class MainMenuController: MonoBehaviour
{
    private UIDocument m_Document;
    private Button m_StartButton;

    private VisualElement m_Blocker;

    private void Start()
    {
        m_Document = GetComponent<UIDocument>();
        m_StartButton = m_Document.rootVisualElement.Q<Button>("StartButton");

        m_StartButton.clicked += () => { m_Blocker.style.opacity = 1.0f; };

        m_Blocker = m_Document.rootVisualElement.Q<VisualElement>("Blocker");
        m_Blocker.RegisterCallback<TransitionEndEvent>(evt =>
        {
            SceneManager.LoadScene(1, LoadSceneMode.Single);
        });
    }
}