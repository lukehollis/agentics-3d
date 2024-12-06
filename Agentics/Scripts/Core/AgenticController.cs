using UnityEngine;
using UnityEngine.AI;
using System.Collections;
using ManaSeedTools.CharacterAnimator;
using System.Collections.Generic;

namespace Agentics.Core
{
    [RequireComponent(typeof(AgenticCharacter))]
    [RequireComponent(typeof(NavMeshAgent))]
    public class AgenticController : MonoBehaviour, IInteractable
    {
        [Header("Core References")]
        public AgenticCharacter character;
        public NavMeshAgent agent;
        public NavMeshObstacle navMeshObstacle;
        public ParticleSystem MoveDust;
        public ParticleSystem JumpDust;
        
        [Header("Movement Settings")]
        public float moveSpeed = 2f;
        public float stoppingDistance = 0.1f;
        public float interactionRadius = 2f;
        public LayerMask interactableLayers;

        [Header("Planning")]
        public DayPlan currentDayPlan;
        public DayPlanAction currentDayPlanAction;
        public ActionTaskList currentActionTasks;
        public GameObject taskIndicator;
        
        [TextArea(minLines: 5, maxLines: 20)]
        public string initialDayPlanJson;
        [TextArea(minLines: 5, maxLines: 20)]
        public string initialActionTasksJson;

        public bool isMoving = false;
        public bool isInteracting = false;
        public bool isInDialog = false;
        public string sleepWakeMode = "wake";
        public Vector3? interruptedDestination;
        public string interruptedTaskEmoji;
        public bool wasNavigating;
        public float taskDuration = 30f;

        protected Animator animator;

        protected virtual void Awake()
        {
            character = GetComponent<AgenticCharacter>();
            agent = GetComponent<NavMeshAgent>();
            navMeshObstacle = GetComponent<NavMeshObstacle>();
            animator = GetComponent<Animator>();

            // Configure NavMesh for 2D
            agent.updateRotation = false;
            agent.updateUpAxis = false;
            agent.speed = moveSpeed;
            agent.stoppingDistance = stoppingDistance;
        }

        protected virtual void Start()
        {
            if (!string.IsNullOrEmpty(initialDayPlanJson))
            {
                UpdatePlan(initialDayPlanJson);
            }
            else
            {
                // If you're using NetworkingController, uncomment this
                // NetworkingController.Instance.OnWebSocketConnected += RequestCharacterPlan;
                
                // For testing, let's create a simple plan
                var testPlan = new DayPlan
                {
                    day_overview = "Test Day",
                    actions = new List<DayPlanAction>
                    {
                        new DayPlanAction
                        {
                            action = "Walk to market",
                            emoji = "🚶",
                            location = "market"  // Make sure this waypoint exists in your TaskWaypoints
                        }
                    }
                };
                
                UpdatePlan(JsonUtility.ToJson(testPlan));
            }
        }

        protected virtual void Update()
        {
            CheckMovement();
            
            if (currentDayPlanAction != null && !isMoving && !isInteracting)
            {
                StartCoroutine(ExecuteCurrentAction());
            }
        }

        
        protected virtual IEnumerator ExecuteCurrentAction()
        {
            if (currentDayPlanAction == null) yield break;
            
            isInteracting = true;
            Vector3 targetPosition = TaskWaypoints.Instance.GetWaypointLocation(currentDayPlanAction.location);
            
            if (targetPosition != Vector3.zero)
            {
                SetDestination(targetPosition);

                if (taskIndicator != null)
                {
                    taskIndicator.SetActive(false);
                }
                
                // Wait until we reach the destination
                while (isMoving)
                {
                    if (Vector3.Distance(transform.position, targetPosition) < stoppingDistance)
                    {
                        isMoving = false;
                    }
                    yield return null;
                }

                // Execute tasks at location
                if (currentActionTasks != null && currentActionTasks.tasks != null)
                {
                    foreach (var task in currentActionTasks.tasks)
                    {
                        yield return StartCoroutine(ExecuteTask(task));
                    }
                }
            }
            
            isInteracting = false;
        }

        protected virtual IEnumerator ExecuteTask(ActionTask task)
        {
            // Show task indicator if available
            if (taskIndicator != null)
            {
                // First verify we have the TextMeshProUGUI component directly
                var tmpText = taskIndicator.GetComponentInChildren<TMPro.TMP_Text>();
                if (tmpText != null) 
                {
                    taskIndicator.SetActive(true);
                    // tmpText.text = task.emoji;
                }
                else
                {
                    Debug.LogWarning("No TextMeshProUGUI component found in taskIndicator or its children");
                }
            }

            // Wait for task duration
            yield return new WaitForSeconds(taskDuration);

            if (taskIndicator != null)
            {
                taskIndicator.SetActive(false);
            }
        }

        protected virtual void CheckMovement()
        {
            if (agent.velocity.magnitude > 0.1f)
            {
                isMoving = true;
                Vector2 movement = new Vector2(agent.velocity.x, agent.velocity.y).normalized;
                
                MoveType moveType = agent.velocity.magnitude > 3f ? 
                    MoveType.running : MoveType.walking;
                    
                character.UpdateAnimationState(movement, moveType);
                PlayMoveDust();
            }
            else if (isMoving)
            {
                isMoving = false;
                character.UpdateAnimationState(Vector2.zero, MoveType.idle);
                StopMoveDust();
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
            animator.SetInteger("direction", direction < 0 ? (int)Direction.left : (int)Direction.right);
        }

        public virtual void SetDestination(Vector3 position)
        {
            if (agent != null && agent.enabled)
            {
                agent.SetDestination(position);
                isMoving = true;
            }
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

        public virtual void UpdatePlan(string planJson)
        {
            if (string.IsNullOrEmpty(planJson)) return;

            // Parse the JSON into DayPlan
            currentDayPlan = JsonUtility.FromJson<DayPlan>(planJson);
            
            if (currentDayPlan != null && currentDayPlan.actions != null && currentDayPlan.actions.Count > 0)
            {
                // Set the first action as current
                currentDayPlanAction = currentDayPlan.actions[0];
                
                // Parse any tasks for this action
                if (!string.IsNullOrEmpty(initialActionTasksJson))
                {
                    currentActionTasks = JsonUtility.FromJson<ActionTaskList>(initialActionTasksJson);
                }
            }
        }

        public virtual void Interact()
        {
            isInteracting = true;
            // Store current state if needed
            if (agent.hasPath)
            {
                interruptedDestination = agent.destination;
                wasNavigating = true;
            }
            
            // Handle interaction logic
            // Will be implemented in derived class
        }

        public virtual void EndInteraction()
        {
            isInteracting = false;
            
            // Restore previous state if needed
            if (wasNavigating && interruptedDestination.HasValue)
            {
                SetDestination(interruptedDestination.Value);
                wasNavigating = false;
                interruptedDestination = null;
            }
        }

        protected virtual void OnDrawGizmosSelected()
        {
            // Draw interaction radius
            Gizmos.color = Color.yellow;
            Gizmos.DrawWireSphere(transform.position, interactionRadius);
        }
    }
}