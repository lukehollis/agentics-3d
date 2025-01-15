using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;
using Agentics;

namespace Agentics
{
    public class AgenticCharacter : MonoBehaviour
    {
        [Header("Character Stats")]
        public int Money = 10;
        public int Health = 10;
        public int Stamina = 10;

        [Header("Identity")]
        public int ID;
        public string CharacterName;
        public Sprite Avatar;

        [Header("Health Conditions")]
        public List<string> healthConditions = new List<string>();
        public float immunityStrength = 1f; // Base immunity multiplier
        public bool isQuarantined = false;

        [Header("Inventory")]
        public Inventory inventory;

        private AgenticNeuralState neuralState;
        private AgenticController controller;

        protected virtual void Awake()
        {
            neuralState = GetComponent<AgenticNeuralState>();
            controller = GetComponent<AgenticController>();
            inventory = new Inventory($"{ID}_{CharacterName}", 24);
            
            // Register with disease controller
            DiseaseController.Instance.RegisterCharacter(this);
        }

        protected virtual void OnDestroy()
        {
            // Unregister from disease controller
            if (DiseaseController.Instance != null)
            {
                DiseaseController.Instance.UnregisterCharacter(this);
            }
        }

        // Add methods to interface with neural state
        public float GetMood() => neuralState.needs.mood;
        public float GetEnergy() => neuralState.needs.energy;

        // Disease-related methods
        public void AddHealthCondition(string condition)
        {
            if (!healthConditions.Contains(condition))
            {
                healthConditions.Add(condition);
                DiseaseController.Instance.OnCharacterInfected(this, condition);
                
                // Update indicator color
                if (controller != null)
                {
                    controller.UpdateIndicatorColor();
                }
            }
        }

        public void RemoveHealthCondition(string condition)
        {
            if (healthConditions.Contains(condition))
            {
                healthConditions.Remove(condition);
                DiseaseController.Instance.OnCharacterRecovered(this, condition);
                
                // Update indicator color
                if (controller != null)
                {
                    controller.UpdateIndicatorColor();
                }
            }
        }

        public bool HasCondition(string condition)
        {
            return healthConditions.Contains(condition);
        }
    }
}