using UnityEngine;
using UnityEngine.AI;
using System.Collections;
using System.Collections.Generic;
using Agentics;


namespace Agentics
{
    /// <summary>
    /// Main NPC character class that handles movement, animations, combat state, and texture management.   
    /// Manages both keyboard/mouse input and NavMesh pathfinding for movement and interactions.
    /// </summary>
    /// <remarks>
    /// The animator's direction parameter uses the following integer values:
    /// - 0: Right
    /// - 1: Left  
    /// - 2: Up
    /// - 3: Down
    /// 
    /// This differs from the Direction enum order to match the sprite sheet layout and animation system.
    /// The Direction enum is used for internal logic while the animator parameter handles the actual
    /// animation states.
    /// </remarks>
    [RequireComponent(typeof(AgenticCharacter))]
    [RequireComponent(typeof(NavMeshAgent))]
    public class AgenticController : MonoBehaviour, Interactable
    {
        [Header("Core References")]
        public AgenticCharacter character;
        public NavMeshAgent agent;
        public NavMeshObstacle navMeshObstacle;
        public ParticleSystem MoveDust;
        public ParticleSystem JumpDust;
        public Animator animator;
        public GameObject characterInfoPanel;
        
        [Header("Movement Settings")]
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
        [TextArea(minLines: 5, maxLines: 20)]
        public string initialDialogOpener;

        public string sleepWakeMode = "wake";
        public Vector3? interruptedDestination;
        public string interruptedTaskEmoji;
        public bool wasNavigating;

        private Coroutine currentActionCoroutine;
        private IEnumerator currentActionCoroutineState;

        private readonly string[] basicAnimations = {
            "idle", "idle2"
        };

        // Action animations
        private readonly string[] actionAnimations = {
            "throwing", "hacking", "watering", "lifting", "fishing", "smithing",
            "climbing", "pushing", "pulling", "jumping"
        };

        // State animations
        private readonly string[] stateAnimations = {
            "sleeping", "schock", "sideglance", "sittingstool", "sittingstoolasleep",
            "sittingstooldrinking", "sittingground", "sittingground2", "toetapping"
        };

        // Combat animations
        private readonly string[] combatAnimations = {
            "combat_swordshield", "combat_spear", "combat_bow", "sheath",
            "slash1", "slash2", "thrust", "thrust2", "shieldbash",
            "retreat", "lunge", "parry", "dodge", "hurt", "dead",
            "shootup", "shootstraight"
        };

        public List<string> animations;

        // Add near other state flags
        private CharacterState currentState = CharacterState.Idle;
        public CharacterState State => currentState;

        public enum CharacterState
        {
            Idle,
            Moving,
            ExecutingTask,
            InDialog
        }

        protected virtual void Awake()
        {
            character = GetComponent<AgenticCharacter>();
            agent = GetComponent<NavMeshAgent>();
            navMeshObstacle = GetComponent<NavMeshObstacle>();

            // Register this controller with the NetworkingController
            NetworkingController.Instance.RegisterAgenticController(character.ID, this);

            // Configure NavMesh for 2D
            agent.updateRotation = false;
            agent.updateUpAxis = false;
            agent.speed = 2f;
            agent.stoppingDistance = 1.0f;
            agent.autoBraking = false;
            agent.angularSpeed = 0;
            agent.acceleration = 9999;

            // Combine all animation arrays into the main animations list
            animations = new List<string>();
            animations.AddRange(basicAnimations);
            animations.AddRange(actionAnimations);
            animations.AddRange(stateAnimations);
            animations.AddRange(combatAnimations);

            // Add these lines to prevent pushing
            var rb = GetComponent<Rigidbody2D>();
            if (rb != null)
            {
                rb.bodyType = RigidbodyType2D.Kinematic;
                rb.collisionDetectionMode = CollisionDetectionMode2D.Continuous;
                rb.interpolation = RigidbodyInterpolation2D.Interpolate;
                rb.constraints = RigidbodyConstraints2D.FreezeAll;
                rb.useFullKinematicContacts = true;
            }

            // initial state is idle
            SetState(CharacterState.Idle);

            // character must always have initial plan
            UpdatePlan(initialDayPlanJson);
        }

        protected virtual void Update()
        {
            // Only check movement and start new actions if not in dialog
            if (currentState != CharacterState.InDialog)
            {
                CheckMovement();
                
                if (
                    currentState == CharacterState.Idle 
                    && currentDayPlanAction != null 
                    && currentActionCoroutine == null
                )
                {
                    currentActionCoroutine = StartCoroutine(ExecuteCurrentAction());
                }
            }
        }

        private void SetNavMeshComponentsMoveable(bool isMoveable)
        {
            if (agent != null && navMeshObstacle != null)
            {
                // When moving, use NavMeshAgent
                if (isMoveable)
                {
                    if (navMeshObstacle.enabled)
                    {
                        navMeshObstacle.enabled = false;
                        // Wait a frame to ensure obstacle is fully disabled
                        StartCoroutine(EnableAgentNextFrame());
                    }
                    else if (!agent.enabled)
                    {
                        agent.enabled = true;
                    }
                }
                // When stationary, use NavMeshObstacle
                else
                {
                    agent.enabled = false;
                    navMeshObstacle.enabled = true;
                }
            }
        }

        private IEnumerator EnableAgentNextFrame()
        {
            yield return null;
            agent.enabled = true;
        }

        protected virtual void RequestPlan()
        {
            if (GameController.Instance.isOfflineMode)    
            {
                OnPlanRequestComplete(true, initialDayPlanJson);
            }
            else
            {
                NetworkingController.Instance.RequestAgenticPlan(character.ID, OnPlanRequestComplete);
            }
        }

        protected virtual void OnPlanRequestComplete(bool success, string planJson)
        {
            if (success && !string.IsNullOrEmpty(planJson))
            {
                UpdatePlan(planJson);
            }
            else
            {
                // Restart the existing plan from the beginning
                if (currentDayPlan != null && currentDayPlan.actions != null && currentDayPlan.actions.Count > 0)
                {
                    currentDayPlanAction = currentDayPlan.actions[0];
                }
            }
        }

        protected virtual IEnumerator ExecuteCurrentAction()
        {
            if (currentDayPlanAction == null)
            {
                currentActionCoroutine = null;
                yield break;
            }

            Vector3 targetPosition = TaskWaypoints.Instance.GetWaypointLocation(currentDayPlanAction.location);

            if (targetPosition != Vector3.zero)
            {
                // Wait for destination to be set and path to be calculated
                yield return StartCoroutine(SetDestinationCoroutine(targetPosition));
                
                // Wait until we reach the destination
                while (agent.enabled && agent.hasPath && agent.remainingDistance > agent.stoppingDistance)
                {
                    // Ensure task indicator is hidden while moving
                    taskIndicator.SetActive(false);
                    
                    // Check if path becomes invalid while walking
                    if (agent.pathStatus == NavMeshPathStatus.PathInvalid)
                    {
                        Debug.LogWarning("Path became invalid while walking");
                        if (agent.enabled)
                        {
                            agent.ResetPath();
                        }
                        yield break;
                    }

                    yield return null;
                }

                // Force stop immediately when we reach destination
                SetState(CharacterState.Idle);
                if (agent.enabled)
                {
                    agent.ResetPath();
                    agent.velocity = Vector3.zero;
                }

                // Execute tasks at location
                if (currentActionTasks != null && currentActionTasks.tasks != null)
                {
                    foreach (var task in currentActionTasks.tasks)
                    {
                        yield return StartCoroutine(ExecuteTask(task));
                    }
                }

                // Check if this was the last action in the plan
                if (currentDayPlan != null && 
                    currentDayPlan.actions != null && 
                    currentDayPlan.actions.IndexOf(currentDayPlanAction) == currentDayPlan.actions.Count - 1)
                {
                    RequestPlan();
                }
                else
                {
                    // Move to next action in the plan
                    int currentIndex = currentDayPlan.actions.IndexOf(currentDayPlanAction);
                    currentDayPlanAction = currentDayPlan.actions[currentIndex + 1];
                }
            }

            currentActionCoroutine = null;
        }

        protected virtual IEnumerator ExecuteTask(ActionTask task)
        {
            SetState(CharacterState.ExecutingTask);
            
            // Show task indicator with emoji
            var tmpText = taskIndicator.GetComponentInChildren<TMPro.TMP_Text>();
            taskIndicator.SetActive(true);
            tmpText.text = task.emoji;

            // generate a random duration for the task
            float taskDuration = Random.Range(30f, 90f);
            float elapsedTime = 0f;

            // Animator uses: 0=Right, 1=Left, 2=Up, 3=Down
            int[] directionValues = { 0, 2, 1, 3 }; // Right, Up, Left, Down
            int currentDirectionIndex = 0;

            while (elapsedTime < taskDuration)
            {
                // Check if we're in dialog - if so, pause the task execution
                if (currentState == CharacterState.InDialog)
                {
                    taskIndicator.SetActive(false);
                    yield return new WaitUntil(() => currentState != CharacterState.InDialog);
                    taskIndicator.SetActive(true);
                }

                float waitTime = Random.Range(1f, 3f);
                yield return new WaitForSeconds(waitTime);
                elapsedTime += waitTime;  // Add the actual wait time

                // Only update animations if not in dialog
                if (currentState != CharacterState.InDialog)
                {
                    // Get the current direction value for the animator
                    int currentDirection = directionValues[Random.Range(0, directionValues.Length)];
                    
                    // Set animation parameters directly
                    animator.SetFloat("xInput", 0);
                    animator.SetFloat("yInput", 0);
                    animator.SetInteger("direction", currentDirection);
                    animator.SetBool("isWalking", false);
                    animator.SetBool("isRunning", false);
                    animator.SetTrigger("idle");

                    // Update direction index
                    currentDirectionIndex = (currentDirectionIndex + 1) % directionValues.Length;
                }
            }

            // Return to idle
            animator.SetTrigger("idle");
            SetState(CharacterState.Idle);
            taskIndicator.SetActive(false);
        }

        protected virtual void SetAnimationState(Vector2 movement, MoveType moveType, Direction direction)
        {
            // Set movement parameters
            animator.SetFloat("xInput", movement.x);
            animator.SetFloat("yInput", movement.y);
            animator.SetInteger("direction", (int)direction);

            // Set movement type
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
                    animator.SetBool("isWalking", false);
                    animator.SetBool("isRunning", false);
                    break;
            }
        }

        protected virtual void CheckMovement()
        {
            // Don't check movement if in dialog or performing task
            if (currentState == CharacterState.InDialog || currentState == CharacterState.ExecutingTask)
                return;

            // Increase threshold and check both velocity and remaining distance
            bool isMoving = agent.velocity.magnitude > 0.01f || 
                            (agent.hasPath && agent.remainingDistance > agent.stoppingDistance);
                            
            if (isMoving)
            {
                SetState(CharacterState.Moving);
                Vector2 movement = new Vector2(agent.velocity.x, agent.velocity.y).normalized;
                
                // Determine direction based on movement
                Direction moveDirection;
                if (Mathf.Abs(movement.x) > Mathf.Abs(movement.y))
                {
                    moveDirection = movement.x > 0 ? Direction.right : Direction.left;
                }
                else
                {
                    moveDirection = movement.y > 0 ? Direction.up : Direction.down;
                }
                
                MoveType moveType = agent.velocity.magnitude > 3f ? 
                    MoveType.running : MoveType.walking;
                    
                SetAnimationState(movement, moveType, moveDirection);
                PlayMoveDust();
            }
            else if (currentState == CharacterState.Moving)
            {
                SetState(CharacterState.Idle);
            }
        }

        public virtual void SetDestination(Vector3 position)
        {
            if (agent != null && agent.enabled)
            {
                agent.SetDestination(position);
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
                } else {
                    currentActionTasks = new ActionTaskList();
                    // Create a default task using the action's emoji
                    if (currentDayPlanAction != null && !string.IsNullOrEmpty(currentDayPlanAction.emoji))
                    {
                        currentActionTasks.tasks = new List<ActionTask> 
                        { 
                            new ActionTask { emoji = currentDayPlanAction.emoji }
                        };
                    }
                }

            }
        }

        public virtual void Interact()
        {
            SetState(CharacterState.InDialog);

            // Store current state if needed
            if (agent.hasPath)
            {
                interruptedDestination = agent.destination;
                wasNavigating = true;
            }
            
            if (agent.enabled)
            {
                // Stop any current movement
                agent.ResetPath();
                agent.velocity = Vector3.zero;
            }
            
            // Find and face the player
            var player = GameObject.FindGameObjectWithTag("Player");
            if (player != null)
            {
                FaceTarget(player.transform.position);
            }

            // Create a new Dialog instance for the initial conversation
            var initialDialog = new Dialog();
            if (!string.IsNullOrEmpty(initialDialogOpener))
            {
                initialDialog.Lines = new List<string> { initialDialogOpener };
            }
            else
            {
                initialDialog.Lines = new List<string>();
            }

            // Start the dialog with the initial opener
            StartCoroutine(DialogManager.Instance.ShowDialog(
                initialDialog,
                character.ID,
                character.CharacterName,
                character.Avatar,
                true
            ));
        }

        public virtual void SetDialogState(bool inDialog)
        {
            if (inDialog)
            {
                if (agent.hasPath)
                {
                    interruptedDestination = agent.destination;
                    wasNavigating = true;
                }
                
                // Stop any movement
                if (agent.enabled)
                {
                    agent.isStopped = true;
                    agent.ResetPath();
                    agent.velocity = Vector3.zero;
                }

                SetState(CharacterState.InDialog);
            }
            else
            {
                agent.isStopped = false;
                
                if (wasNavigating && interruptedDestination.HasValue)
                {
                    // Resume navigation to interrupted destination
                    SetState(CharacterState.Moving);
                    SetDestination(interruptedDestination.Value);
                    wasNavigating = false;
                    interruptedDestination = null;
                }
                else
                {
                    SetState(CharacterState.Idle);
                }
            }
        }

        protected virtual void OnDrawGizmosSelected()
        {
            // Draw interaction radius
            Gizmos.color = Color.yellow;
            Gizmos.DrawWireSphere(transform.position, interactionRadius);
        }

        protected virtual void FaceTarget(Vector3 targetPosition)
        {
            Vector2 direction = (targetPosition - transform.position).normalized;

            // Determine facing direction (using animator values: 0=Right, 1=Left, 2=Up, 3=Down)
            int faceDirection;
            if (Mathf.Abs(direction.x) > Mathf.Abs(direction.y))
            {
                faceDirection = direction.x > 0 ? 0 : 1; // Right = 0, Left = 1
            }
            else
            {
                faceDirection = direction.y > 0 ? 2 : 3; // Up = 2, Down = 3
            }

            // Set animation parameters directly
            animator.SetFloat("xInput", 0);
            animator.SetFloat("yInput", 0);
            // This SET IDLE MUST REMAIN HERE FOR IT TO WORK 
            animator.SetTrigger("idle");
            animator.SetInteger("direction", faceDirection);
            animator.SetBool("isWalking", false);
            animator.SetBool("isRunning", false);
        }

        public virtual IEnumerator SetDestinationCoroutine(Vector3 position)
        {
            if (currentState != CharacterState.InDialog)
            {
                // Ensure NavMeshAgent is enabled and properly configured for movement
                SetNavMeshComponentsMoveable(true);
                
                // Wait a frame to ensure NavMeshAgent is fully enabled
                yield return null;
                
                // Set destination and ensure the agent isn't stopped
                agent.isStopped = false;
                agent.SetDestination(position);

                // Wait until we have a valid path or timeout after a few seconds
                float timeout = 3f;
                float elapsed = 0f;
                while (elapsed < timeout && 
                      (agent.pathStatus == NavMeshPathStatus.PathInvalid ||
                       !agent.hasPath ||
                       agent.pathStatus == NavMeshPathStatus.PathPartial))
                {
                    elapsed += Time.deltaTime;
                    yield return null;
                }

                if (elapsed >= timeout)
                {
                    Debug.LogWarning($"Path finding timed out for {character.CharacterName}");
                }
            }
        }

        private void SetState(CharacterState newState)
        {
            if (currentState == newState) return;
            
            // Exit current state
            switch (currentState)
            {
                case CharacterState.Moving:
                    agent.ResetPath();
                    agent.velocity = Vector3.zero;
                    StopMoveDust();
                    SetNavMeshComponentsMoveable(false);
                    break;
                case CharacterState.ExecutingTask:
                    taskIndicator.SetActive(false);
                    break;
                case CharacterState.InDialog:
                    break;
            }

            currentState = newState;

            // Enter new state
            switch (newState)
            {
                case CharacterState.Idle:
                    animator.SetFloat("xInput", 0);
                    animator.SetFloat("yInput", 0);
                    animator.SetBool("isWalking", false);
                    animator.SetBool("isRunning", false);
                    animator.SetTrigger("idle");
                    SetNavMeshComponentsMoveable(false);
                    break;
                case CharacterState.ExecutingTask:
                    taskIndicator.SetActive(true);
                    SetNavMeshComponentsMoveable(false);
                    break;
                case CharacterState.InDialog:
                    SetNavMeshComponentsMoveable(false);
                    break;
                case CharacterState.Moving:
                    SetNavMeshComponentsMoveable(true);
                    break;
            }
        }

        // Also add cleanup in OnDestroy
        protected virtual void OnDestroy()
        {
            // Unregister when the object is destroyed
            if (character != null)
            {
                NetworkingController.Instance.UnregisterAgenticController(character.ID);
            }
        }

        public virtual void ShowCharacterInfoPanel()
        {
            characterInfoPanel.SetActive(true);
        }

        public virtual void HideCharacterInfoPanel()
        {
            characterInfoPanel.SetActive(false);
        }

    }

}