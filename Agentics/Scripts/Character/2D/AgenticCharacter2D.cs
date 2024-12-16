using UnityEngine;
using System;
using System.Collections;
using Agentics;

namespace Agentics
{
    public class AgenticCharacter2D : AgenticCharacter
    {
        [Header("2D Specific Components")]
        public SpriteRenderer Body;

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

        private Direction characterDirection = Direction.None;

        protected override void Awake()
        {
            base.Awake();
            SetupAnimations();
        }

        private void SetupAnimations()
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

        public override void UpdateAnimationState(Vector2 movement, MoveType newMoveType)
        {
            moveType = newMoveType;
            
            // Update character direction based on movement
            if (movement.magnitude > 0)
            {
                if (Mathf.Abs(movement.x) > Mathf.Abs(movement.y))
                {
                    // Horizontal movement takes priority
                    characterDirection = movement.x > 0 ? Direction.Right : Direction.Left;
                }
                else
                {
                    // Vertical movement
                    characterDirection = movement.y > 0 ? Direction.Up : Direction.Down;
                }
            }

            // Update animator parameters
            if (Animator != null)
            {
                Animator.SetFloat("xInput", movement.x);
                Animator.SetFloat("yInput", movement.y);
                Animator.SetInteger("direction", (int)characterDirection);
                Animator.SetBool("isWalking", moveType == MoveType.Walking);
                Animator.SetBool("isRunning", moveType == MoveType.Running);
            }
        }

        protected virtual void SetTexture(Texture2D texture, string layer)
        {
            string path = textureBasePath.Replace("Assets/Resources/", "");
            // Implementation from Player.cs SetTexture method
            // Reference to Player.cs SetTexture implementation:
        }
    }
}