using UnityEngine;
using ManaSeedTools.CharacterAnimator;

namespace Agentics.Core
{
    public class AgenticMovementAnimationControl : MonoBehaviour
    {
        private Animator animator;

        private void Awake()
        {
            animator = GetComponent<Animator>();
        }

        // Remove event subscription/unsubscription since we'll call this directly
        public void SetAnimationParameters(float inputX, float inputY,
            MoveType moveType, Direction direction)
        {
            animator.SetFloat("xInput", inputX);
            animator.SetFloat("yInput", inputY);
            animator.SetBool("isRunning", true);
            animator.SetInteger("direction", (int)direction);

            switch (moveType)
            {
                case MoveType.walking:
                    animator.SetBool("isWalking", true);
                    animator.SetBool("isRunning", false);
                    break;

                case MoveType.running:
                    animator.SetBool("isWalking", false);
                    animator.SetBool("isRunning", true);
                    break;

                case MoveType.idle:
                    animator.SetTrigger("idle");
                    break;
            }
        }
    }
}