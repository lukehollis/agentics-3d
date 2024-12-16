using UnityEngine;
using System.Collections;
using System.Collections.Generic;

namespace Agentics
{
    public abstract class AgenticController : MonoBehaviour, Interactable
    {
        [Header("Core References")]
        public AgenticCharacter character;
        
        [Header("Movement Settings")]
        public float baseWalkSpeed = 2f;
        public float baseRunSpeed = 4f;
        private float speedMultiplier = 1f;

        public float walkSpeed 
        {
            get { return baseWalkSpeed * speedMultiplier; }
            set { baseWalkSpeed = value; }
        }

        public float runSpeed
        {
            get { return baseRunSpeed * speedMultiplier; }
            set { baseRunSpeed = value; }
        }

        public float SpeedMultiplier
        {
            get { return speedMultiplier; }
            set { speedMultiplier = Mathf.Clamp(value, 0.1f, 2f); }
        }

        public float stoppingDistance = 0.1f;
        public float interactionRadius = 2f;
        public LayerMask interactableLayers;
        public string sleepWakeMode = "wake";

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
        public Vector3? interruptedDestination;
        public bool wasNavigating;
        public float taskDuration = 30f;

        protected Animator animator;
        protected readonly int speedHash = Animator.StringToHash("Speed");
        protected readonly int motionSpeedHash = Animator.StringToHash("MotionSpeed");
        protected readonly int groundedHash = Animator.StringToHash("Grounded");

        public float moveSpeed
        {
            get { return currentMoveType == MoveType.Running ? runSpeed : walkSpeed; }
        }

        protected MoveType currentMoveType = MoveType.Idle;

        protected virtual void Awake()
        {
            animator = GetComponent<Animator>();
            SetupComponents();
        }

        protected abstract void SetupComponents();
        
        protected virtual void Start()
        {
            if (!string.IsNullOrEmpty(initialDayPlanJson))
            {
                UpdatePlan(initialDayPlanJson);
            }
            else
            {
                CreateDefaultPlan();
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

        protected abstract void CheckMovement();
        public abstract void SetDestination(Vector3 position);
        public abstract void Interact();
        protected abstract IEnumerator ExecuteCurrentAction();
        public abstract void UpdatePlan(string planJson);
        
        protected virtual void CreateDefaultPlan()
        {
            var testPlan = new DayPlan
            {
                day_overview = "Test Day",
                actions = new List<DayPlanAction>
                {
                    new DayPlanAction
                    {
                        action = "Walk to market",
                        emoji = "🚶",
                        location = "market"
                    }
                }
            };
            
            UpdatePlan(JsonUtility.ToJson(testPlan));
        }

        public virtual void SetDialogState(bool inDialog)
        {
            isInDialog = inDialog;
            
            // Stop movement when entering dialog
            if (inDialog)
            {
                isMoving = false;
                if (animator != null)
                {
                    animator.SetFloat("Speed", 0f);
                }
            }
        }
    }
}