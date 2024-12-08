using UnityEngine;
using System;
using System.Collections;

namespace Agentics
{
    public class AgenticCharacter : MonoBehaviour
    {
        [Header("Core Components")]
        public Animator Animator;
        public SpriteRenderer Body;
        
        [Header("Character Stats")]
        public int Money = 10;
        public int Health = 10;
        public int Stamina = 10;

        [Header("Identity")]
        public int ID;
        public string CharacterName;
        public Sprite Avatar;

        private Direction characterDirection = Direction.none;
        private MoveType moveType = MoveType.idle;
        private AgenticNeuralState neuralState;
        private MovementAnimationControl movementControl;

        protected virtual void Awake()
        {
            neuralState = GetComponent<AgenticNeuralState>();
            movementControl = GetComponent<MovementAnimationControl>();
            SetupMSCAnimations();
        }

        public void UpdateAnimationState(Vector2 movement, MoveType newMoveType)
        {
            moveType = newMoveType;
            
            if (movement.magnitude > 0)
            {
                if (movement.x < 0) characterDirection = Direction.left;
                else if (movement.x > 0) characterDirection = Direction.right;
                else if (movement.y < 0) characterDirection = Direction.down;
                else characterDirection = Direction.up;
            }

            movementControl.SetAnimationParameters(
                movement.x, 
                movement.y, 
                moveType, 
                characterDirection
            );
        }

        private void SetTexture(Texture2D texture, string layer)
        {
            string path = textureBasePath.Replace("Assets/Resources/", "");
            // Implementation from Player.cs SetTexture method
            // Reference to Player.cs SetTexture implementation:
        }

        // Add methods to interface with neural state
        public float GetMood() => neuralState.needs.mood;
        public float GetEnergy() => neuralState.needs.energy;
    }
}