using UnityEngine;

namespace Agentics
{
    public class AgenticCharacter3D : AgenticCharacter
    {
        [Header("3D Specific Components")]
        public SkinnedMeshRenderer MeshRenderer;

        protected override void Awake()
        {
            base.Awake();
            // Any 3D-specific initialization can go here
        }

        public override void UpdateAnimationState(Vector2 movement, MoveType newMoveType)
        {
            moveType = newMoveType;
            
            // 3D specific movement handling
            Vector3 movement3D = new Vector3(movement.x, 0, movement.y);
            if (movement3D.magnitude > 0)
            {
                transform.rotation = Quaternion.LookRotation(movement3D);
            }

            // Use MovementAnimationControl to update animator
            if (movementControl != null)
            {
                movementControl.SetAnimationParameters(
                    movement.x,
                    movement.y,
                    moveType,
                    Direction.None  // 3D doesn't use discrete directions
                );
            }
        }
    }
} 