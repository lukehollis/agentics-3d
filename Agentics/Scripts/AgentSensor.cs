using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using UnityEngine.AI;

public class AgentSensor : MonoBehaviour
{
    [Header("Observation Settings")]
    [SerializeField] private float viewRadius = 5f;
    [SerializeField] private LayerMask relevantLayers;
    [SerializeField] private int gridSize = 8;
    [SerializeField] private float maxPathDistance = 20f;

    [Header("Debug Visualization")]
    [SerializeField] private bool showDebugVisuals = true;
    [SerializeField] private Color pathColor = Color.yellow;
    
    private Collider2D[] nearbyObjects = new Collider2D[20];
    private GridState[,] observationGrid;
    private NavMeshAgent agent;
    private AgentBrain agentBrain;
    
    private void Awake()
    {
        observationGrid = new GridState[gridSize, gridSize];
        agent = GetComponent<NavMeshAgent>();
        agentBrain = GetComponent<AgentBrain>();
    }

    public struct GridState
    {
        public bool isOccupied;
        public bool isWalkable;
        public bool hasInteractable;
        public int interactableType;
        public float distanceToAgent;
        public bool isOnPath;
    }

    public void CollectObservations(VectorSensor sensor)
    {
        UpdateObservationGrid();
        
        // Add navigation-related observations
        sensor.AddObservation(agent.velocity.normalized);
        sensor.AddObservation(agent.hasPath);
        sensor.AddObservation(agent.pathStatus);
        sensor.AddObservation(agent.remainingDistance / maxPathDistance); // Normalized distance
        
        // Add grid observations
        for (int x = 0; x < gridSize; x++)
        {
            for (int y = 0; y < gridSize; y++)
            {
                var state = observationGrid[x, y];
                sensor.AddObservation(state.isOccupied);
                sensor.AddObservation(state.isWalkable);
                sensor.AddObservation(state.hasInteractable);
                sensor.AddObservation(state.interactableType);
                sensor.AddObservation(state.distanceToAgent / viewRadius); // Normalized distance
                sensor.AddObservation(state.isOnPath);
            }
        }
    }

    private void UpdateObservationGrid()
    {
        Vector2 agentPos = transform.position;
        System.Array.Clear(observationGrid, 0, observationGrid.Length);
        
        // Sample NavMesh for walkable areas
        NavMeshHit hit;
        for (int x = 0; x < gridSize; x++)
        {
            for (int y = 0; y < gridSize; y++)
            {
                Vector2 worldPos = GridToWorldPosition(new Vector2Int(x, y));
                Vector3 pos3D = new Vector3(worldPos.x, worldPos.y, 0);
                
                var state = new GridState();
                state.distanceToAgent = Vector2.Distance(worldPos, agentPos);
                state.isWalkable = NavMesh.SamplePosition(pos3D, out hit, 0.5f, NavMesh.AllAreas);
                state.isOnPath = IsPositionOnPath(worldPos);
                
                observationGrid[x, y] = state;
            }
        }

        // Find and process nearby objects
        int numFound = Physics2D.OverlapCircleNonAlloc(
            agentPos,
            viewRadius,
            nearbyObjects,
            relevantLayers
        );

        for (int i = 0; i < numFound; i++)
        {
            var obj = nearbyObjects[i];
            if (obj == null) continue;

            Vector2Int gridPos = WorldToGridPosition(obj.transform.position - transform.position);
            if (!IsInGridBounds(gridPos)) continue;

            UpdateGridCell(gridPos, obj);
        }
    }

    private bool IsPositionOnPath(Vector2 worldPos)
    {
        if (!agent.hasPath) return false;

        foreach (Vector3 corner in agent.path.corners)
        {
            if (Vector2.Distance(worldPos, corner) < 0.5f)
                return true;
        }
        return false;
    }

    private Vector2 GridToWorldPosition(Vector2Int gridPos)
    {
        float cellSize = (viewRadius * 2) / gridSize;
        Vector2 bottomLeft = (Vector2)transform.position - new Vector2(viewRadius, viewRadius);
        return bottomLeft + new Vector2(
            (gridPos.x + 0.5f) * cellSize,
            (gridPos.y + 0.5f) * cellSize
        );
    }

    private Vector2Int WorldToGridPosition(Vector2 worldPos)
    {
        float normalizedX = (worldPos.x + viewRadius) / (viewRadius * 2);
        float normalizedY = (worldPos.y + viewRadius) / (viewRadius * 2);
        
        return new Vector2Int(
            Mathf.FloorToInt(normalizedX * gridSize),
            Mathf.FloorToInt(normalizedY * gridSize)
        );
    }

    private bool IsInGridBounds(Vector2Int pos)
    {
        return pos.x >= 0 && pos.x < gridSize && 
               pos.y >= 0 && pos.y < gridSize;
    }

    private void UpdateGridCell(Vector2Int gridPos, Collider2D obj)
    {
        var state = observationGrid[gridPos.x, gridPos.y];
        state.isOccupied = true;

        if (obj.TryGetComponent<IInteractable>(out var interactable))
        {
            state.hasInteractable = true;
            state.interactableType = GetInteractableType(obj);
        }

        observationGrid[gridPos.x, gridPos.y] = state;
    }

    private int GetInteractableType(Collider2D obj)
    {
        // Implement your interactable type classification here
        // For example:
        if (obj.CompareTag("Item")) return 1;
        if (obj.CompareTag("NPC")) return 2;
        if (obj.CompareTag("Resource")) return 3;
        return 0;
    }

    private void OnDrawGizmos()
    {
        if (!showDebugVisuals) return;

        // Draw view radius
        Gizmos.color = Color.yellow;
        Gizmos.DrawWireSphere(transform.position, viewRadius);

        // Draw grid and path
        if (observationGrid != null)
        {
            float cellSize = (viewRadius * 2) / gridSize;
            Vector3 bottomLeft = transform.position - new Vector3(viewRadius, viewRadius, 0);

            for (int x = 0; x < gridSize; x++)
            {
                for (int y = 0; y < gridSize; y++)
                {
                    var state = observationGrid[x, y];
                    if (state.isOccupied || state.isOnPath)
                    {
                        Gizmos.color = state.isOnPath ? pathColor :
                                      state.hasInteractable ? Color.blue :
                                      state.isWalkable ? Color.green :
                                      Color.red;

                        Vector3 cellPos = bottomLeft + new Vector3(
                            x * cellSize + cellSize/2,
                            y * cellSize + cellSize/2,
                            0
                        );
                        Gizmos.DrawWireCube(cellPos, Vector3.one * cellSize * 0.8f);
                    }
                }
            }
        }
    }
}