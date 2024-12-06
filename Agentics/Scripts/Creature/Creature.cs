using UnityEngine;
using UnityEngine.AI;
using System.Collections;

public class Creature : MonoBehaviour
{
    private Animator animator;
    private SpriteRenderer spriteRenderer;
    private NavMeshAgent navMeshAgent;
    private static readonly int DirectionHash = Animator.StringToHash("direction");
    private static readonly int IsRunningHash = Animator.StringToHash("isRunning");

    void Start()
    {
        animator = GetComponent<Animator>();
        spriteRenderer = GetComponent<SpriteRenderer>();
        navMeshAgent = GetComponent<NavMeshAgent>();

        // Lock NavMeshAgent to XY plane
        navMeshAgent.updateUpAxis = false;
        navMeshAgent.updateRotation = false;

        // Start the random movement coroutine
        StartCoroutine(RandomMovementRoutine());
    }

    void Update()
    {
        UpdateAnimator();

        // Keep Z position at 0
        Vector3 position = transform.position;
        position.z = 0;
        transform.position = position;
    }

    public void Move(Vector2 direction)
    {
        // Common movement logic
        navMeshAgent.SetDestination((Vector3)direction);
    }

    public void Eat()
    {
        // Common eating logic
    }

    private void MoveToRandomPosition()
    {
        Vector3 randomDirection = Random.insideUnitSphere * 3f; // 3 units radius
        randomDirection += transform.position;

        NavMeshHit hit;
        NavMesh.SamplePosition(randomDirection, out hit, 1.5f, NavMesh.AllAreas);

        Vector3 finalPosition = hit.position;
        navMeshAgent.SetDestination(finalPosition);
    }

    private IEnumerator RandomMovementRoutine()
    {
        while (true)
        {
            MoveToRandomPosition();

            // Wait until the agent reaches its destination
            while (navMeshAgent.pathPending || navMeshAgent.remainingDistance > navMeshAgent.stoppingDistance)
            {
                yield return null;
            }

            // Idle for a random duration between 2 and 6 seconds
            float idleTime = Random.Range(2f, 6f);
            yield return new WaitForSeconds(idleTime);
        }
    }

    private void UpdateAnimator()
    {
        Vector2 velocity = new Vector2(navMeshAgent.velocity.x, navMeshAgent.velocity.y);
        float speed = velocity.magnitude;

        // Set isRunning based on whether we're moving
        animator.SetBool(IsRunningHash, speed > 0.1f);

        // Set direction based on movement
        if (speed > 0.1f)
        {
            Vector2 normalizedVelocity = velocity.normalized;
            
            // Determine direction based on dominant axis
            if (Mathf.Abs(normalizedVelocity.x) > Mathf.Abs(normalizedVelocity.y))
            {
                // Moving horizontally - use actual Left/Right states
                animator.SetInteger(DirectionHash, normalizedVelocity.x > 0 ? 2 : 3); // Right : Left
            }
            else
            {
                // Moving vertically - FIXED: normalizedVelocity.y > 0 means moving UP, so we want DOWN animation
                animator.SetInteger(DirectionHash, normalizedVelocity.y > 0 ? 0 : 1); // Down : Up (reversed from before)
            }
        }
    }
}