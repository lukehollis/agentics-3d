using UnityEngine;
using UnityEngine.AI;
using System.Collections;
using System.Collections.Generic;
using Agentics;
using Polyperfect.Common;

namespace Agentics
{
    [RequireComponent(typeof(AgenticCharacter))]
    [RequireComponent(typeof(CharacterController))]
    [RequireComponent(typeof(TransportationController))]
    public class AgenticController : MonoBehaviour, Interactable
    {
        [Header("Core References")]
        public AgenticCharacter character;
        public CharacterController characterController;
        public Animator animator;
        public GameObject characterInfoPanel;
        private TransportationController transportationController;
        
        [Header("Movement Settings")]
        public float interactionRadius = 2f;
        public LayerMask interactableLayers;
        public float SpeedMultiplier = 1f;
        public Vector3 targetPos;
        private float moveSpeed = 3.5f; // Add this field to control movement speed

        [Header("Planning")]
        public DayPlan currentDayPlan;
        public DayPlanAction currentDayPlanAction;
        public ActionTaskList currentActionTasks;
        public GameObject characterIndicator;
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
        private float turnSpeed = 120f;

        private Coroutine currentActionCoroutine;
        private IEnumerator currentActionCoroutineState;

        [Header("Animation States")]
        public IdleState[] idleStates;
        public MovementState[] movementStates;

        private HashSet<string> animatorParameters = new HashSet<string>();


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

        private Material characterIndicatorMaterial;
        private Color healthyColor = Color.white;
        private Color infectedColor = new Color(0.5f, 0f, 0f, 1f); // Dark red

        private float characterHeight = 2f; // Height above ground to maintain
        private float maxRaycastDistance = 1000f; // Maximum distance to check for ground
        private float currentElevation; // Track current elevation when no ground is found

        protected virtual void Awake()
        {
            character = GetComponent<AgenticCharacter>();
            characterController = GetComponent<CharacterController>();
            transportationController = GetComponent<TransportationController>();

            // Get the indicator material
            if (characterIndicator != null)
            {
                var renderer = characterIndicator.GetComponent<Renderer>();
                if (renderer != null)
                {
                    characterIndicatorMaterial = renderer.material;
                    // Make sure this material is set to a queue that renders on top
                    renderer.material.renderQueue = 4000;
                    UpdateIndicatorColor();
                }
                else
                {
                    Debug.LogError($"No Renderer found on characterIndicator for {character.CharacterName}");
                }
            }
            else
            {
                Debug.LogError($"No characterIndicator assigned for {character.CharacterName}");
            }

            // initial state is idle
            SetState(CharacterState.Idle);

            // character must always have initial plan
            UpdatePlan(initialDayPlanJson);

            // Initialize current elevation
            currentElevation = transform.position.y;
            
            // Disable gravity on character controller since we're handling elevation
            characterController.slopeLimit = 90f;
            characterController.stepOffset = 0.3f;
            characterController.minMoveDistance = 0f;
        }

        private void Start()
        {
            if (characterIndicator != null)
            {
                var renderer = characterIndicator.GetComponent<Renderer>();
                if (renderer != null && renderer.material != null)
                {
                    // Make sure this material is set to a queue that renders on top (3000 is Transparent, 4000 is Overlay)
                    renderer.material.renderQueue = 4000;
                }
            }
        }

        protected virtual void Update()
        {
            // Only check movement and start new actions if not in dialog
            if (currentState != CharacterState.InDialog)
            {
                // Update ground position every frame
                UpdateGroundPosition();
                
                if (
                    currentState == CharacterState.Idle 
                    && currentDayPlanAction != null 
                    && currentActionCoroutine == null
                )
                {
                    currentActionCoroutine = StartCoroutine(ExecuteCurrentAction());
                }
            }

            // Check distance to player and scale indicator
            if (characterIndicator != null)
            {
                GameObject player = GameObject.FindGameObjectWithTag("Player");
                if (player != null)
                {
                    float distanceToPlayer = Vector3.Distance(transform.position, player.transform.position);
                    Vector3 indicatorScale = characterIndicator.transform.localScale;
                    
                    // Set scale based on distance
                    float targetScale = distanceToPlayer <= 100f ? 1f : 10f;
                    indicatorScale = Vector3.one * targetScale;
                    
                    characterIndicator.transform.localScale = indicatorScale;
                }
            }
        }

        protected virtual void RequestPlan()
        {
            if (SimulationController.Instance.isOfflineMode)    
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
            float taskRadius = 10f;

            if (targetPosition != Vector3.zero)
            {
                // Start walking animation before movement
                if (animator != null)
                {
                    animator.SetBool("isWalking", true);
                    animator.SetBool("isRunning", false);
                }

                SetState(CharacterState.Moving);

                Debug.Log("Character: " + character.CharacterName + " Traveling to destination: " + currentDayPlanAction.location + " " + targetPosition);

                // // Use the transportation controller to handle the journey
                yield return StartCoroutine(
                    transportationController.TravelToDestination(
                        currentDayPlanAction.location, 
                        targetPosition
                    )
                );

                // Once we've arrived, continue with task execution
                SetState(CharacterState.Idle);
                if (animator != null)
                {
                    animator.SetBool("isWalking", false);
                    animator.SetBool("isRunning", false);
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
                    
                    // Update direction index
                    currentDirectionIndex = (currentDirectionIndex + 1) % directionValues.Length;
                }
            }

            // Return to idle
            SetState(CharacterState.Idle);
            taskIndicator.SetActive(false);
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
            if (characterController.enabled)
            {
                interruptedDestination = transform.position;
                wasNavigating = true;
            }
            
            if (characterController.enabled)
            {
                // Stop any current movement
                characterController.enabled = false;
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
        }

        private IdleState GetRandomIdleState()
        {
            if (idleStates == null || idleStates.Length == 0)
                return null;

            int totalWeight = 0;
            foreach (var state in idleStates)
                totalWeight += state.stateWeight;

            int randomWeight = Random.Range(0, totalWeight);
            int currentWeight = 0;

            foreach (var state in idleStates)
            {
                currentWeight += state.stateWeight;
                if (randomWeight <= currentWeight)
                    return state;
            }

            return idleStates[0];
        }

        void FaceDirection(Vector3 facePosition)
        {
            transform.rotation = Quaternion.LookRotation(Vector3.ProjectOnPlane(Vector3.RotateTowards(transform.forward,
                facePosition, turnSpeed * Time.deltaTime*Mathf.Deg2Rad, 0f), Vector3.up), Vector3.up);
        }

        public virtual void SetDestination(Vector3 position)
        {
            Debug.Log("Setting destination: " + position);
            targetPos = position;
            
            // Calculate direction vector (only X and Z)
            Vector3 direction = (position - transform.position);
            direction.y = 0; // Ignore vertical component for movement direction
            direction.Normalize();
            
            // Update ground position/elevation
            UpdateGroundPosition();
            
            // Move using CharacterController
            characterController.SimpleMove(direction * moveSpeed * SpeedMultiplier);
            
            // Face the movement direction
            FaceDirection(direction);
        }


        public virtual void SetDialogState(bool inDialog)
        {
            if (inDialog)
            {
                if (characterController.enabled)
                {
                    interruptedDestination = transform.position;
                    wasNavigating = true;
                }
                
                // Stop any movement
                if (characterController.enabled)
                {
                    characterController.enabled = false;
                }

                SetState(CharacterState.InDialog);
            }
            else
            {
                characterController.enabled = true;
                
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

            // Draw navigation line and destination point when moving
            if (currentState == CharacterState.Moving)
            {
                // Draw destination sphere

                // Draw line from character to destination
                Gizmos.DrawLine(transform.position, targetPos);
            }
        }

        public virtual IEnumerator SetDestinationCoroutine(Vector3 position)
        {
            Debug.Log("Setting destination for SetDestinationCoroutine: " + position);
            targetPos = position;
            float moveSpeed = 5f;
            float rotationSpeed = 2f;
            float stoppingDistance = 2f;

            SetState(CharacterState.Moving);
            
            while (Vector3.Distance(transform.position, position) > stoppingDistance)
            {
                float currentDistance = Vector3.Distance(transform.position, position);
                
                Vector3 direction = (position - transform.position).normalized;
                characterController.SimpleMove(direction * moveSpeed * SpeedMultiplier);
                
                // Face the movement direction
                FaceDirection(direction);
                
                yield return null;
            }

            Debug.Log("Reached destination: " + position);
            SetState(CharacterState.Idle);
        }

        private void SetState(CharacterState newState)
        {
            if (currentState == newState) return;
            
            // Exit current state
            switch (currentState)
            {
                case CharacterState.Moving:
                    if (animator != null)
                    {
                        animator.SetBool("isWalking", false);
                        animator.SetBool("isRunning", false);
                    }
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
                    if (animator != null)
                    {
                        animator.SetBool("isWalking", false);
                        animator.SetBool("isRunning", false);
                    }
                    break;
                case CharacterState.Moving:
                    if (animator != null)
                    {
                        animator.SetBool("isWalking", true);
                        animator.SetBool("isRunning", false);
                    }
                    break;
                case CharacterState.ExecutingTask:
                    taskIndicator.SetActive(true);
                    break;
                case CharacterState.InDialog:
                    if (animator != null)
                    {
                        animator.SetBool("isWalking", false);
                        animator.SetBool("isRunning", false);
                    }
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

        public void UpdateIndicatorColor()
        {
            if (characterIndicatorMaterial != null)
            {
                Debug.Log($"Updating indicator color for {character.CharacterName}. Has conditions: {character.healthConditions.Count > 0}");
                characterIndicatorMaterial.color = character.healthConditions.Count > 0 ? 
                    infectedColor : healthyColor;
            }
            else
            {
                Debug.LogError($"No material found for character indicator on {character.CharacterName}");
            }
        }

        private void UpdateGroundPosition()
        {
            Vector3 rayStart = transform.position + Vector3.up * 10f; // Start above current position
            RaycastHit hitInfo;
            
            // Cast ray downward to find ground
            if (Physics.Raycast(rayStart, Vector3.down, out hitInfo, maxRaycastDistance))
            {
                // Check if the hit object is a Cesium tile
                // we can't use teh cesium georeference... just use any meshrenderer 
                if (hitInfo.transform.IsChildOf(FindObjectOfType<MeshRenderer>().transform))
                {
                    // Set position to hit point plus character height
                    Vector3 newPosition = transform.position;
                    newPosition.y = hitInfo.point.y + characterHeight;
                    transform.position = newPosition;
                    currentElevation = newPosition.y;
                }
                else
                {
                    // If not hitting Cesium terrain, maintain current elevation
                    Vector3 newPosition = transform.position;
                    newPosition.y = currentElevation;
                    transform.position = newPosition;
                }
            }
            else
            {
                // No ground found, maintain current elevation
                Vector3 newPosition = transform.position;
                newPosition.y = currentElevation;
                transform.position = newPosition;
            }
        }

    }

}
