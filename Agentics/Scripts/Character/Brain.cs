using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine;
using UnityEngine.AI;
using Agentics;

namespace Agentics
{
    public class Brain : Agent
    {
        private NavMeshAgent navAgent;
        private Animator animator;
        private SpriteRenderer spriteRenderer;
        
        [Header("Movement Settings")]
        public float moveSpeed = 2f;
        public float rotationSpeed = 100f;
        public float stoppingDistance = 0.1f;

        [Header("Interaction Settings")]
        public float interactionRadius = 2f;
        public LayerMask interactableLayers;

        private Vector3 previousPosition;
        private bool isInteracting;
        private Vector3? currentGoalPosition;
        private bool hasGoal;

        public override void Initialize()
        {
            navAgent = GetComponent<NavMeshAgent>();
            animator = GetComponent<Animator>();
            spriteRenderer = GetComponent<SpriteRenderer>();

            // Configure NavMeshAgent for 2D
            navAgent.updateRotation = false;
            navAgent.updateUpAxis = false;
            
            previousPosition = transform.position;
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            // Agent's current position and velocity
            sensor.AddObservation(transform.position);
            sensor.AddObservation(navAgent.velocity);
            
            // Navigation state
            sensor.AddObservation(navAgent.hasPath);
            sensor.AddObservation(navAgent.remainingDistance);
            sensor.AddObservation((int)navAgent.pathStatus);
            
            // Interaction state
            sensor.AddObservation(isInteracting);
            
            // Add nearby interactable objects
            Collider2D[] nearbyObjects = Physics2D.OverlapCircleAll(
                transform.position, 
                interactionRadius, 
                interactableLayers
            );
            
            // Observe closest interactable
            float closestDistance = float.MaxValue;
            Vector3 closestDirection = Vector3.zero;
            
            foreach (Collider2D obj in nearbyObjects)
            {
                float distance = Vector3.Distance(transform.position, obj.transform.position);
                if (distance < closestDistance)
                {
                    closestDistance = distance;
                    closestDirection = obj.transform.position - transform.position;
                }
            }
            
            sensor.AddObservation(closestDistance);
            sensor.AddObservation(closestDirection);
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            var discreteActions = actions.DiscreteActions;
            
            // Handle movement
            HandleMovement(discreteActions[0]);
            
            // Handle interaction
            if (discreteActions[1] == 1)
            {
                TryInteract();
            }
        }

        private void HandleMovement(int moveAction)
        {
            Vector3 targetPosition = transform.position;
            
            switch (moveAction)
            {
                case 1: // Up
                    targetPosition += Vector3.up;
                    break;
                case 2: // Right
                    targetPosition += Vector3.right;
                    break;
                case 3: // Down
                    targetPosition += Vector3.down;
                    break;
                case 4: // Left
                    targetPosition += Vector3.left;
                    break;
            }

            if (moveAction != 0)
            {
                navAgent.SetDestination(targetPosition);
                UpdateAnimation();
            }
        }

        private void UpdateAnimation()
        {
            if (navAgent.velocity.magnitude > 0.1f)
            {
                animator.SetBool("IsMoving", true);
                // Flip sprite based on movement direction
                if (navAgent.velocity.x != 0)
                {
                    spriteRenderer.flipX = navAgent.velocity.x < 0;
                }
            }
            else
            {
                animator.SetBool("IsMoving", false);
            }
        }

        private void TryInteract()
        {
            Collider2D[] nearbyObjects = Physics2D.OverlapCircleAll(
                transform.position,
                interactionRadius,
                interactableLayers
            );

            float closestDistance = float.MaxValue;
            Interactable closestInteractable = null;

            foreach (Collider2D obj in nearbyObjects)
            {
                float distance = Vector3.Distance(transform.position, obj.transform.position);
                if (distance < closestDistance)
                {
                    var interactable = obj.GetComponent<Interactable>();
                    if (interactable != null)
                    {
                        closestDistance = distance;
                        closestInteractable = interactable;
                    }
                }
            }

            if (closestInteractable != null)
            {
                closestInteractable.Interact();
                isInteracting = true;
            }
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var discreteActions = actionsOut.DiscreteActions;
            
            // Map keyboard input to actions for testing
            if (Input.GetKey(KeyCode.W)) discreteActions[0] = 1;
            else if (Input.GetKey(KeyCode.D)) discreteActions[0] = 2;
            else if (Input.GetKey(KeyCode.S)) discreteActions[0] = 3;
            else if (Input.GetKey(KeyCode.A)) discreteActions[0] = 4;
            else discreteActions[0] = 0;

            discreteActions[1] = Input.GetKey(KeyCode.Space) ? 1 : 0;
        }

        public bool HasCurrentGoal()
        {
            return hasGoal;
        }

        public Vector3 GetCurrentGoalPosition()
        {
            return currentGoalPosition ?? transform.position;
        }

        public void SetCurrentGoal(Vector3 position)
        {
            currentGoalPosition = position;
            hasGoal = true;
        }

        public void ClearCurrentGoal()
        {
            currentGoalPosition = null;
            hasGoal = false;
        }
    }
}