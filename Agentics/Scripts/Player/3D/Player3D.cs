using UnityEngine;
using UnityEngine.AI;
using Agentics;


namespace Agentics
{
    public class Player3D : Player
    {
        protected override void ConfigureNavMeshAgent()
        {
            agent.updateRotation = true;
            agent.speed = runningSpeed;
            agent.angularSpeed = rotationSpeed;
            agent.stoppingDistance = 0.1f;
        }

        protected override void HandleMovementInput()
        {
            // Get input axes
            float horizontal = Input.GetAxisRaw("Horizontal");
            float vertical = Input.GetAxisRaw("Vertical");

            // Create movement vector relative to camera
            Vector3 forward = mainCamera.transform.forward;
            Vector3 right = mainCamera.transform.right;
            forward.y = 0;
            right.y = 0;
            forward.Normalize();
            right.Normalize();

            moveDirection = (forward * vertical + right * horizontal).normalized;

            // Handle movement type (walking/running)
            bool isShiftHeld = Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift);
            currentMoveType = moveDirection.magnitude > 0.1f ? 
                (isShiftHeld ? MoveType.Walking : MoveType.Running) : 
                MoveType.Idle;

            // Apply movement
            if (moveDirection.magnitude > 0.1f)
            {
                // Cancel any existing NavMesh path
                if (agent.hasPath)
                {
                    agent.ResetPath();
                }

                // Set speed based on movement type
                float currentSpeed = currentMoveType == MoveType.Running ? runningSpeed : walkingSpeed;
                agent.velocity = moveDirection * currentSpeed;

                // Rotate player to face movement direction
                Quaternion targetRotation = Quaternion.LookRotation(moveDirection);
                transform.rotation = Quaternion.Lerp(transform.rotation, targetRotation, rotationSpeed * Time.deltaTime);
            }
            else if (!agent.hasPath)
            {
                agent.velocity = Vector3.zero;
            }
        }

        protected override void HandleInteractionInput()
        {
            if (Input.GetMouseButtonDown(0))
            {
                Ray ray = mainCamera.ScreenPointToRay(Input.mousePosition);
                RaycastHit hit;

                if (Physics.Raycast(ray, out hit))
                {
                    // Check for interactable objects
                    IInteractable interactable = hit.collider.GetComponent<IInteractable>();
                    if (interactable != null)
                    {
                        float distance = Vector3.Distance(transform.position, hit.point);
                        if (distance <= interactionRange)
                        {
                            interactable.Interact();
                        }
                        else
                        {
                            // Move to interaction point
                            agent.stoppingDistance = interactionRange * 0.8f;
                            agent.SetDestination(hit.point);
                            StartCoroutine(InteractWhenInRange(interactable));
                        }
                    }
                    else
                    {
                        // Regular movement
                        agent.stoppingDistance = 0.1f;
                        agent.SetDestination(hit.point);
                    }
                }
            }
        }

        protected override void UpdateAnimations()
        {
            if (animator != null)
            {
                // Update animator parameters
                animator.SetFloat("Speed", agent.velocity.magnitude);
                animator.SetBool("IsWalking", currentMoveType == MoveType.Walking);
                animator.SetBool("IsRunning", currentMoveType == MoveType.Running);
            }
        }

        private System.Collections.IEnumerator InteractWhenInRange(IInteractable interactable)
        {
            isInteracting = true;

            // Wait until we reach the destination
            while (agent.pathStatus == NavMeshPathStatus.PathPartial ||
                   agent.remainingDistance > agent.stoppingDistance)
            {
                yield return null;
            }

            // Face the interactable
            Vector3 lookPosition = interactable.GetTransform().position;
            lookPosition.y = transform.position.y;
            transform.LookAt(lookPosition);

            // Perform interaction
            interactable.Interact();
            
            // Reset
            agent.stoppingDistance = 0.1f;
            isInteracting = false;
        }

        protected override void FaceTarget(Vector3 targetPosition)
        {
            Vector3 lookPosition = targetPosition;
            lookPosition.y = transform.position.y;
            transform.LookAt(lookPosition);
        }
    }

    // Interface for interactable objects
    public interface IInteractable
    {
        void Interact();
        Transform GetTransform();
    }
}