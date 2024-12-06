using UnityEngine;
using System;
using System.Collections;
using MSCEventHandler = ManaSeedTools.CharacterAnimator.EventHandler;
using ManaSeedTools.CharacterAnimator;

namespace Agentics.Core
{
    public class AgenticCharacter : MonoBehaviour
    {
        [Header("Core Components")]
        public Animator Animator;
        public SpriteRenderer Body;
        
        [Header("Character Stats")]
        public int Money = 9;
        public int Health = 10;
        public int Stamina = 10;

        [Header("Identity")]
        public int ID;
        public string CharacterName;
        public Sprite Avatar;

        [Header("MSC Animation Settings")]
        public string textureBasePath;
        public string spriteSetPath;
        
        [Header("Character Textures")]
        public Texture2D bodyT;
        public Texture2D outfitT;
        public Texture2D cloakT;
        public Texture2D faceitemsT;
        public Texture2D hairT;
        public Texture2D hatT;
        
        [Header("Equipment")]
        public GameObject hat;
        public GameObject hair;

        private Direction characterDirection = Direction.none;
        private MoveType moveType = MoveType.idle;
        private AgenticNeuralState neuralState;
        private AgenticMovementAnimationControl movementControl;

        protected virtual void Awake()
        {
            neuralState = GetComponent<AgenticNeuralState>();
            movementControl = GetComponent<AgenticMovementAnimationControl>();
            SetupMSCAnimations();
        }

        private void SetupMSCAnimations()
        {
            if (bodyT != null) SetTexture(bodyT, "body");
            if (outfitT != null) SetTexture(outfitT, "outfit");
            if (cloakT != null) SetTexture(cloakT, "cloak");
            if (faceitemsT != null) SetTexture(faceitemsT, "faceitems");
            if (hairT != null) SetTexture(hairT, "hair");
            if (hatT != null)
            {
                hair.SetActive(false);
                SetTexture(hatT, "hat");
            }
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