using UnityEngine;

namespace Agentics
{
    public abstract class MovementAnimationControl : MonoBehaviour
    {
        protected Animator animator;

        protected virtual void Awake()
        {
            animator = GetComponent<Animator>();
        }

        public abstract void SetAnimationParameters(float inputX, float inputY, MoveType moveType, Direction direction);
    }

    public class MovementAnimationControl2D : MovementAnimationControl
    {
        public override void SetAnimationParameters(float inputX, float inputY, MoveType moveType, Direction direction)
        {
            animator.SetFloat("xInput", inputX);
            animator.SetFloat("yInput", inputY);
            animator.SetInteger("direction", (int)direction);

            animator.SetBool("isWalking", moveType == MoveType.Walking);
            animator.SetBool("isRunning", moveType == MoveType.Running);
        }
    }

    public class MovementAnimationControl3D : MovementAnimationControl
    {
        public override void SetAnimationParameters(float inputX, float inputY, MoveType moveType, Direction direction)
        {
            float speed = new Vector2(inputX, inputY).magnitude;
            animator.SetFloat("Speed", speed);
            animator.SetBool("IsWalking", moveType == MoveType.Walking);
            animator.SetBool("IsRunning", moveType == MoveType.Running);
        }
    }
}