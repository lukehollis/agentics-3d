using UnityEngine;

public class MovementAnimationController : MonoBehaviour
{
    private Animator animator;
    
    // Animation parameter names - can be customized per project
    [Header("Animation Parameters")]
    [SerializeField] private string horizontalInputParam = "xInput";
    [SerializeField] private string verticalInputParam = "yInput";
    [SerializeField] private string isWalkingParam = "isWalking";
    [SerializeField] private string isRunningParam = "isRunning";
    [SerializeField] private string directionParam = "direction";
    [SerializeField] private string idleParam = "idle";

    private void Awake()
    {
        animator = GetComponent<Animator>();
        ValidateParameters();
    }

    private void ValidateParameters()
    {
        if (animator == null)
        {
            Debug.LogError($"No Animator component found on {gameObject.name}");
            enabled = false;
            return;
        }
    }

    /// <summary>
    /// Updates the animation state based on movement input and type
    /// </summary>
    /// <param name="inputX">Horizontal input (-1 to 1)</param>
    /// <param name="inputY">Vertical input (-1 to 1)</param>
    /// <param name="moveType">Type of movement (idle, walking, running)</param>
    /// <param name="direction">Direction the character is facing</param>
    public void UpdateAnimationState(float inputX, float inputY, MovementType moveType, Direction direction)
    {
        if (animator == null) return;

        // Update movement inputs
        animator.SetFloat(horizontalInputParam, inputX);
        animator.SetFloat(verticalInputParam, inputY);

        // Update movement state
        switch (moveType)
        {
            case MovementType.Walking:
                animator.SetBool(isWalkingParam, true);
                animator.SetBool(isRunningParam, false);
                break;

            case MovementType.Running:
                animator.SetBool(isWalkingParam, false);
                animator.SetBool(isRunningParam, true);
                break;

            case MovementType.Idle:
                animator.SetBool(isWalkingParam, false);
                animator.SetBool(isRunningParam, false);
                animator.SetTrigger(idleParam);
                break;
        }

        // Update direction
        animator.SetInteger(directionParam, (int)direction);
    }
}
