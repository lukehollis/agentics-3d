using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;

#if UNITY_EDITOR

using UnityEditor;

#endif

namespace Agentics
{

    public class Player : MonoBehaviour
    {
        private float xInput, yInput, movementSpeed;
        private MoveType moveType;
        private Direction playerDirection;

        private float runningSpeed = 3.5f;
        private float walkingSpeed = 1.75f;
        private float combatMoveSpeed = 0.5f;

        private string testTrigger;
        private bool inCombat;


        private Animator animator;


        public UnityEngine.AI.NavMeshAgent agent;
        public Inventory inventory;

        public int Money { get; set; } = 0;
        public int Health { get; set; } = 100;
        public int Stamina { get; set; } = 100;

        private void Awake()
        {
            animator = GetComponent<Animator>();
            inventory = GetComponent<Inventory>();
        }

        // Start is called before the first frame update
        private void Start()
        {
            moveType = MoveType.idle;
            playerDirection = Direction.none;

            SetCharacterTextures();

            if (agent == null)
            {
                agent = GetComponent<UnityEngine.AI.NavMeshAgent>();
            }

            // Configure NavMeshAgent for 2D
            agent.updateRotation = false;
            agent.updateUpAxis = false;
        }

        // Update is called once per frame
        public void HandleUpdate()
        {
            ResetAnimationTriggers();
            PlayerMovementInput();
            CheckMovement();
        }


        private void PlayerMovementInput()
        {
            // Handle keyboard input
            yInput = Input.GetAxisRaw("Vertical");
            xInput = Input.GetAxisRaw("Horizontal");

            // Prioritize horizontal movement over vertical
            if (Mathf.Abs(xInput) > 0.01f)
            {
                yInput = 0;
            }

            // If using keyboard movement, cancel any NavMesh path
            if (xInput != 0 || yInput != 0)
            {
                // this must check both if it has a path or if the path is complete
                // keep both conditions in case the path is not complete but the agent is still moving
                if (agent.hasPath || agent.pathStatus == UnityEngine.AI.NavMeshPathStatus.PathComplete)
                {
                    agent.ResetPath();
                }

                // Move using keyboard input
                Vector3 movement = new Vector3(xInput, yInput, 0);

                // Apply diagonal movement correction
                if (xInput != 0 && yInput != 0)
                {
                    movement.x *= 0.71f;
                    movement.y *= 0.71f;
                }

                // Set the agent's velocity based on movement input
                Vector3 targetVelocity = movement * (moveType == MoveType.running ? runningSpeed : walkingSpeed);
                agent.velocity = targetVelocity;
            }

            // Handle mouse input
            if (Input.GetMouseButtonDown(0) && GameController.Instance.state == GameState.FreeRoam)
            {
                // Create a ray from the mouse cursor
                Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
                RaycastHit hit;

                // Check if the pointer is over a UI element
                if (EventSystem.current.IsPointerOverGameObject())
                {
                    return;
                }

                // Get the world point where the mouse clicked
                Vector3 worldPoint = Camera.main.ScreenToWorldPoint(Input.mousePosition);
                worldPoint.z = 0; // Ensure z is 0 for 2D

                // Show the cursor indicator at the clicked position
                if (cursorIndicator != null)
                {
                    cursorIndicator.ShowAtPosition(worldPoint);
                }

                // Get direction to target
                Vector2 direction = (worldPoint - transform.position).normalized;
                
                // Determine dominant axis
                if (Mathf.Abs(direction.x) > Mathf.Abs(direction.y))
                {
                    // Horizontal movement
                    worldPoint.y = transform.position.y;
                }
                else
                {
                    // Vertical movement
                    worldPoint.x = transform.position.x;
                }

                // Check NavMesh and set destination
                UnityEngine.AI.NavMeshHit navMeshHit;
                if (UnityEngine.AI.NavMesh.SamplePosition(worldPoint, out navMeshHit, 1.0f, UnityEngine.AI.NavMesh.AllAreas))
                {
                    agent.SetDestination(navMeshHit.position);
                }
            }

            // Handle keyboard movement direction
            if (xInput != 0 && yInput != 0)
            {
                xInput *= .71f;
                yInput *= .71f;
            }

            //check if player walks
            if (Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift))
            {
                moveType = MoveType.walking;
                movementSpeed = walkingSpeed;
            }
            else
            {
                moveType = MoveType.running;
                movementSpeed = runningSpeed;
            }

            if (inCombat) movementSpeed = combatMoveSpeed;
        }

        private void ResetAnimationTriggers()
        {
            // Only reset moveType if we're not using NavMesh navigation
            if (agent.velocity.magnitude < 0.1f)
            {
                moveType = MoveType.idle;
            }
            //playerDirection = Direction.none;
        }

        private void ResetMovement()
        {
            //reset movement
            xInput = 0f;
            yInput = 0f;
        }

        private void CheckMovement()
        {
            Vector2 velocity = new Vector2(agent.velocity.x, agent.velocity.y);
            float speed = velocity.magnitude;

            if (speed > 0.01f)
            {
                moveType = MoveType.running;
                Vector2 normalizedVelocity = velocity.normalized;

                // Force movement to cardinal directions only
                if (Mathf.Abs(normalizedVelocity.x) > Mathf.Abs(normalizedVelocity.y))
                {
                    normalizedVelocity.y = 0;
                    normalizedVelocity.x = normalizedVelocity.x > 0 ? 1 : -1;
                }
                else
                {
                    normalizedVelocity.x = 0;
                    normalizedVelocity.y = normalizedVelocity.y > 0 ? 1 : -1;
                }

                xInput = normalizedVelocity.x;
                yInput = normalizedVelocity.y;

                // Determine direction based on dominant axis
                if (Mathf.Abs(normalizedVelocity.x) > Mathf.Abs(normalizedVelocity.y))
                {
                    // Moving horizontally
                    playerDirection = normalizedVelocity.x > 0 ? Direction.right : Direction.left;
                }
                else
                {
                    // Moving vertically
                    playerDirection = normalizedVelocity.y > 0 ? Direction.up : Direction.down;
                }

                // Call movement event with the current movement values
                EventHandler.CallMovementEvent(
                    xInput,
                    yInput,
                    moveType,
                    playerDirection,
                    this,
                    null
                );
            }
            else
            {
                moveType = MoveType.idle;
                xInput = 0;
                yInput = 0;
                EventHandler.CallMovementEvent(0, 0, moveType, playerDirection, this, null);
            }
        }
    }
}