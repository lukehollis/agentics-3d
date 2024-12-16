using UnityEngine;
using UnityEngine.AI;
using System.Collections;
using System.Collections.Generic;
using Agentics;

namespace Agentics
{
    [RequireComponent(typeof(AgenticCharacter2D))]
    [RequireComponent(typeof(NavMeshAgent))]
    public class AgenticController2D : AgenticController
    {
        public NavMeshAgent agent;
        public NavMeshObstacle navMeshObstacle;
        public ParticleSystem MoveDust;
        public ParticleSystem JumpDust;

        protected override void SetupComponents()
        {
            character = GetComponent<AgenticCharacter2D>();
            agent = GetComponent<NavMeshAgent>();
            navMeshObstacle = GetComponent<NavMeshObstacle>();
            
            // Configure NavMesh for 2D
            agent.updateRotation = false;
            agent.updateUpAxis = false;
            agent.speed = walkSpeed;
            agent.stoppingDistance = stoppingDistance;
        }

        protected override void CheckMovement()
        {
            if (agent.velocity.magnitude > 0.1f)
            {
                isMoving = true;
                Vector2 movement = new Vector2(agent.velocity.x, agent.velocity.y).normalized;
                
                MoveType moveType = agent.velocity.magnitude > 3f ? 
                    MoveType.Running : MoveType.Walking;
                    
                UpdateOrientation();
                ((AgenticCharacter2D)character).UpdateAnimationState(movement, moveType);
                PlayMoveDust();
            }
            else if (isMoving)
            {
                isMoving = false;
                ((AgenticCharacter2D)character).UpdateAnimationState(Vector2.zero, MoveType.Idle);
                StopMoveDust();
            }
        }

        public override void SetDestination(Vector3 position)
        {
            if (agent != null && agent.enabled)
            {
                agent.SetDestination(position);
                isMoving = true;
            }
        }

        public override void Interact()
        {
            Debug.Log("Character interacted with " + name);

            isInteracting = true;
            // Store current state if needed
            if (agent.hasPath)
            {
                interruptedDestination = agent.destination;
                wasNavigating = true;
            }
            
            // Stop any current movement
            agent.ResetPath();
            agent.velocity = Vector3.zero;
            
            // Find and face the player
            var player = GameObject.FindGameObjectWithTag("Player");
            if (player != null)
            {
                FaceTarget(player.transform.position);
            }
        }

        protected virtual void UpdateOrientation()
        {
            if (agent.velocity.x > 0.1f)
            {
                Turn(1);
            }
            else if (agent.velocity.x < -0.1f)
            {
                Turn(-1);
            }
        }

        protected virtual void Turn(int direction)
        {
            animator.SetInteger("direction", direction < 0 ? (int)Direction.Left : (int)Direction.Right);
        }

        protected virtual void PlayMoveDust()
        {
            if (MoveDust != null && !MoveDust.isPlaying)
            {
                MoveDust.Play();
            }
        }

        protected virtual void StopMoveDust()
        {
            if (MoveDust != null && MoveDust.isPlaying)
            {
                MoveDust.Stop();
            }
        }

        protected virtual void FaceTarget(Vector3 targetPosition)
        {
            Vector2 direction = (targetPosition - transform.position).normalized;
            
            // Determine which direction to face based on the dominant axis
            if (Mathf.Abs(direction.x) > Mathf.Abs(direction.y))
            {
                // Moving horizontally
                ((AgenticCharacter2D)character).UpdateAnimationState(
                    new Vector2(direction.x > 0 ? 1 : -1, 0), 
                    MoveType.Idle
                );
            }
            else
            {
                // Moving vertically
                ((AgenticCharacter2D)character).UpdateAnimationState(
                    new Vector2(0, direction.y > 0 ? 1 : -1), 
                    MoveType.Idle
                );
            }
        }

        protected virtual void OnDrawGizmosSelected()
        {
            // Draw interaction radius
            Gizmos.color = Color.yellow;
            Gizmos.DrawWireSphere(transform.position, interactionRadius);
        }

        public override void UpdatePlan(string planJson)
        {
            if (string.IsNullOrEmpty(planJson)) return;
            
            currentDayPlan = JsonUtility.FromJson<DayPlan>(planJson);
            if (currentDayPlan != null && currentDayPlan.actions.Count > 0)
            {
                currentDayPlanAction = currentDayPlan.actions[0];
                Debug.Log($"Updated plan for {name}: {currentDayPlan.day_overview}");
            }
        }

        protected override IEnumerator ExecuteCurrentAction()
        {
            if (currentDayPlanAction == null) yield break;

            isInteracting = true;
            Debug.Log($"{name} executing action: {currentDayPlanAction.action}");

            // Wait for task duration
            yield return new WaitForSeconds(taskDuration);

            // Clear current action
            currentDayPlanAction = null;
            isInteracting = false;

            // Resume previous navigation if it was interrupted
            if (wasNavigating && interruptedDestination.HasValue)
            {
                SetDestination(interruptedDestination.Value);
                wasNavigating = false;
                interruptedDestination = null;
            }
        }
    }
}