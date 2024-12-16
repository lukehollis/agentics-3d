using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.AI;

#if UNITY_EDITOR

using UnityEditor;

#endif

namespace Agentics
{

    public class Player2D : Player
    {
        [Header("2D Specific")]
        public ParticleSystem MoveDust;
        public ParticleSystem JumpDust;

        protected override void ConfigureNavMeshAgent()
        {
            agent.updateRotation = false;
            agent.updateUpAxis = false;
            agent.radius = 0.2f;
            agent.obstacleAvoidanceType = ObstacleAvoidanceType.HighQualityObstacleAvoidance;
            agent.avoidancePriority = 50;
        }

        protected override void HandleMovementInput()
        {
            // Reference to original Player2D movement input:
            // startLine: 509
            // endLine: 530
        }

        protected override void FaceTarget(Vector3 targetPosition)
        {
            // Reference to original Player2D FaceTarget:
            // startLine: 642
            // endLine: 658
        }

        protected override void HandleInteractionInput()
        {
            if (Input.GetMouseButtonDown(0))
            {
                Ray ray = mainCamera.ScreenPointToRay(Input.mousePosition);
                RaycastHit2D hit = Physics2D.Raycast(ray.origin, ray.direction);

                if (hit.collider != null)
                {
                    // Check for interactable objects
                    IInteractable interactable = hit.collider.GetComponent<IInteractable>();
                    if (interactable != null)
                    {
                        float distance = Vector2.Distance(transform.position, hit.point);
                        if (distance <= interactionRange)
                        {
                            interactable.Interact();
                        }
                        else
                        {
                            // Move to interaction point
                            agent.stoppingDistance = interactionStoppingDistance;
                            SetDestination(hit.point);
                            StartCoroutine(InteractWhenInRange(interactable));
                        }
                    }
                    else
                    {
                        // Regular movement
                        agent.stoppingDistance = 0.1f;
                        SetDestination(hit.point);
                    }
                }
            }
        }

        protected override void UpdateAnimations()
        {
            if (animator != null)
            {
                // Get movement direction
                Vector2 movement = new Vector2(agent.velocity.x, agent.velocity.y);
                float speed = movement.magnitude;

                // Update animator parameters
                animator.SetFloat("Speed", speed);
                animator.SetBool("IsWalking", currentMoveType == MoveType.Walking);
                animator.SetBool("IsRunning", currentMoveType == MoveType.Running);

                // Update dust effects
                if (speed > 0.1f)
                {
                    if (MoveDust != null && !MoveDust.isPlaying)
                    {
                        MoveDust.Play();
                    }
                }
                else
                {
                    if (MoveDust != null && MoveDust.isPlaying)
                    {
                        MoveDust.Stop();
                    }
                }
            }
        }

        private IEnumerator InteractWhenInRange(IInteractable interactable)
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
            FaceTarget(lookPosition);

            // Perform interaction
            interactable.Interact();
            
            // Reset
            agent.stoppingDistance = 0.1f;
            isInteracting = false;
        }
    }
}