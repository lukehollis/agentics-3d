using UnityEngine;
using System;
using System.Collections;
using Agentics;

namespace Agentics
{
    public abstract class AgenticCharacter : MonoBehaviour
    {
        [Header("Core Components")]
        public Animator Animator;
        
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

        protected MoveType moveType = MoveType.Idle;
        private AgenticNeuralState neuralState;
        protected MovementAnimationControl movementControl;

        protected virtual void Awake()
        {
            neuralState = GetComponent<AgenticNeuralState>();
            movementControl = GetComponent<MovementAnimationControl>();
            inventory = new Inventory($"{ID}_{CharacterName}", 24);
        }

        public abstract void UpdateAnimationState(Vector2 movement, MoveType newMoveType);

        // Add methods to interface with neural state
        public float GetMood() => neuralState.needs.mood;
        public float GetEnergy() => neuralState.needs.energy;
    }
}