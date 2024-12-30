using UnityEngine;
using System;
using System.Collections;
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

        [Header("Inventory")]
        public Inventory inventory;

        private AgenticNeuralState neuralState;

        protected virtual void Awake()
        {
            neuralState = GetComponent<AgenticNeuralState>();
            inventory = new Inventory($"{ID}_{CharacterName}", 24);
        }

        // Add methods to interface with neural state
        public float GetMood() => neuralState.needs.mood;
        public float GetEnergy() => neuralState.needs.energy;
    }
}