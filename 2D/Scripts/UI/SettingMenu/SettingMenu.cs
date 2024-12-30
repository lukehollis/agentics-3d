using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UIElements;

namespace Template2DCommon
{
    public class SettingMenu
    {
        public System.Action OnClose;
        public System.Action OnOpen;

        private VisualElement m_Root;
        private Button m_OpenMenu;

        private Button m_CloseButton;
        private Button m_QuitButton;

        private DropdownField m_ResolutionDropdown;
        private Toggle m_FullscreenToggle;

        private Slider m_MainVolumeSlider;
        private Slider m_BGMVolumeSlider;
        private Slider m_SFXVolumeSlider;

        private List<Resolution> m_AvailableResolutions;
    
        public SettingMenu(VisualElement root)
        {
            m_Root = root.Q<VisualElement>("SettingMenu");
            m_OpenMenu = root.Q<Button>("OpenSettingMenuButton");

            m_CloseButton = m_Root.Q<Button>("CloseButton");
            m_QuitButton = m_Root.Q<Button>("QuitButton");

            m_ResolutionDropdown = m_Root.Q<DropdownField>("ResolutionDropdown");
            m_FullscreenToggle = m_Root.Q<Toggle>("FullscreenToggle");

            m_MainVolumeSlider = m_Root.Q<Slider>("MainVolume");
            m_BGMVolumeSlider = m_Root.Q<Slider>("MusicVolume");
            m_SFXVolumeSlider = m_Root.Q<Slider>("SFXVolume");

            m_MainVolumeSlider.RegisterValueChangedCallback(evt =>
            {
                SoundManager.Instance.Sound.MainVolume = evt.newValue;
                SoundManager.Instance.UpdateVolume();
            });
            m_BGMVolumeSlider.RegisterValueChangedCallback(evt =>
            {
                SoundManager.Instance.Sound.BGMVolume = evt.newValue;
                SoundManager.Instance.UpdateVolume();
            });
            m_SFXVolumeSlider.RegisterValueChangedCallback(evt =>
            {
                SoundManager.Instance.Sound.SFXVolume = evt.newValue;
                SoundManager.Instance.UpdateVolume();
            });

            m_Root.visible = false;

            m_OpenMenu.clicked += () =>
            {
                if (m_Root.visible)
                {
                    Close();
                }
                else
                {
                    Open();
                }
            };

            m_CloseButton.clicked += Close;
            m_QuitButton.clicked += Application.Quit;
        
            //fill resolution dropdown
            m_AvailableResolutions = new List<Resolution>();
        
            List<string> resEntries = new List<string>();
            foreach (var resolution in Screen.resolutions)
            {
                //if we already have a resolution with same width & height, we skip.
                if(m_AvailableResolutions.FindIndex(r => r.width == resolution.width && r.height == resolution.height) != -1)
                    continue;
            
                var resName = resolution.width+"x"+resolution.height;
                resEntries.Add(resName);
                m_AvailableResolutions.Add(resolution);
            
            }

            m_ResolutionDropdown.choices = resEntries;

            m_ResolutionDropdown.RegisterValueChangedCallback(evt =>
            {
                if (m_ResolutionDropdown.index == -1)
                    return;
            
                var res = m_AvailableResolutions[m_ResolutionDropdown.index];
                Screen.SetResolution(res.width, res.height, m_FullscreenToggle.value);
            });

            m_FullscreenToggle.value = Screen.fullScreen;
            m_FullscreenToggle.RegisterValueChangedCallback(evt =>
            {
                Screen.fullScreen = evt.newValue;
            });
        }

        bool CompareResolution(Resolution a, Resolution b)
        {
            return a.width == b.width && a.height == b.height && a.refreshRateRatio.CompareTo(b.refreshRateRatio) == 0;
        }

        void Open()
        {
            m_MainVolumeSlider.SetValueWithoutNotify(SoundManager.Instance.Sound.MainVolume);
            m_BGMVolumeSlider.SetValueWithoutNotify(SoundManager.Instance.Sound.BGMVolume);
            m_SFXVolumeSlider.SetValueWithoutNotify(SoundManager.Instance.Sound.SFXVolume);
        
            string currentRes = Screen.width + "x" + Screen.height;
            m_ResolutionDropdown.label = currentRes;
            m_ResolutionDropdown.SetValueWithoutNotify(currentRes);
        
            m_Root.visible = true;
            OnOpen.Invoke();   
        }

        void Close()
        {
            SoundManager.Instance.PlayUISound();
            SoundManager.Instance.Save();
            m_Root.visible = false;
            OnClose.Invoke();
        }
    }
}