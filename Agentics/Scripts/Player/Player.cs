using UnityEngine;
using UnityEngine.AI;

namespace Agentics
{
    public abstract class Player : MonoBehaviour
    {
        [Header("Movement Settings")]
        [SerializeField] protected float runningSpeed = 6f;
        [SerializeField] protected float walkingSpeed = 3f;
        [SerializeField] protected float rotationSpeed = 10f;
        [SerializeField] protected float interactionRange = 3f;
        [SerializeField] protected float interactionStoppingDistance = 1.1f;
        
        [Header("Components")]
        [SerializeField] protected NavMeshAgent agent;
        [SerializeField] protected Animator animator;

        [Header("Inventory")]
        public Inventory inventory;

        protected Vector3 moveDirection;
        protected MoveType currentMoveType;
        protected bool isInteracting;
        protected Camera mainCamera;

        protected virtual void Awake()
        {
            if (agent == null) agent = GetComponent<NavMeshAgent>();
            if (animator == null) animator = GetComponent<Animator>();
            mainCamera = Camera.main;
            
            ConfigureNavMeshAgent();
            inventory = new Inventory("Player", 24);
        }

        protected abstract void ConfigureNavMeshAgent();
        
        public virtual void HandleUpdate()
        {
            if (!isInteracting)
            {
                HandleMovementInput();
                HandleInteractionInput();
            }
            
            UpdateAnimations();
        }

        protected abstract void HandleMovementInput();
        protected abstract void HandleInteractionInput();
        protected abstract void UpdateAnimations();

        protected virtual void ResetMovement()
        {
            if (agent != null)
            {
                agent.ResetPath();
                agent.velocity = Vector3.zero;
            }
            moveDirection = Vector3.zero;
            currentMoveType = MoveType.Idle;
        }

        protected virtual void SetDestination(Vector3 position)
        {
            if (agent != null && agent.enabled)
            {
                agent.stoppingDistance = 0.1f;
                agent.SetDestination(position);
            }
        }

        protected abstract void FaceTarget(Vector3 targetPosition);
    }
}