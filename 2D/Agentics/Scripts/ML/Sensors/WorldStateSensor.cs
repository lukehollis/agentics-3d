using UnityEngine;
using Unity.MLAgents.Sensors;
using UnityEngine.AI;
using System.Collections.Generic;
using Agentics;

namespace Agentics
{
    public class WorldStateSensor : MonoBehaviour
    {
        [Header("References")]
        private AgenticController controller;
        private NavMeshAgent agent;
        private Brain agentBrain;

        [Header("Observation Settings")]
        [SerializeField] private float detectionRadius = 10f;
        [SerializeField] private int maxDetectedObjects = 10;
        [SerializeField] private LayerMask environmentLayers;
        [SerializeField] private LayerMask npcLayers;
        [SerializeField] private LayerMask interactableLayers;

        [Header("Grid Settings")]
        [SerializeField] private int gridSize = 16;
        [SerializeField] private float cellSize = 1f;
        [SerializeField] private bool visualizeGrid = false;

        private readonly Dictionary<GameObject, float> lastInteractionTimes = new Dictionary<GameObject, float>();
        private readonly Collider2D[] nearbyColliders = new Collider2D[20];

        private void Awake()
        {
            controller = GetComponent<AgenticController>();
            agent = GetComponent<NavMeshAgent>();
            agentBrain = GetComponent<Brain>();
        }

        public void CollectObservations(VectorSensor sensor)
        {
            // Environment state
            sensor.AddObservation(IsIndoors());
            sensor.AddObservation(GetTimeOfDay());
            sensor.AddObservation(GetCurrentWeather());

            // Nearby NPCs
            ObserveNearbyCharacters(sensor);

            // Grid-based environment observations
            ObserveLocalGrid(sensor);

            // Path and navigation data
            ObservePathData(sensor);

            // Interactable objects
            ObserveInteractables(sensor);
        }

        private bool IsIndoors()
        {
            return Physics2D.OverlapPoint(transform.position, LayerMask.GetMask("Indoors")) != null;
        }

        private float GetTimeOfDay()
        {
            return System.DateTime.Now.Hour / 24f;
        }

        private float GetCurrentWeather()
        {
            // Placeholder for weather system integration
            return 0f;
        }

        private void ObserveNearbyCharacters(VectorSensor sensor)
        {
            int numFound = Physics2D.OverlapCircleNonAlloc(
                transform.position,
                detectionRadius,
                nearbyColliders,
                npcLayers
            );

            // Observe count
            sensor.AddObservation(numFound / (float)maxDetectedObjects);

            // Observe closest NPCs (up to maxDetectedObjects)
            var sortedNPCs = new List<(Vector2 direction, float distance)>();
            for (int i = 0; i < numFound && i < maxDetectedObjects; i++)
            {
                Vector2 direction = nearbyColliders[i].transform.position - transform.position;
                float distance = direction.magnitude;
                sortedNPCs.Add((direction.normalized, distance / detectionRadius));
            }

            // Sort by distance and add observations
            sortedNPCs.Sort((a, b) => a.distance.CompareTo(b.distance));
            foreach (var npc in sortedNPCs)
            {
                sensor.AddObservation(npc.direction);
                sensor.AddObservation(npc.distance);
            }
        }

        private void ObserveLocalGrid(VectorSensor sensor)
        {
            Vector2 gridOrigin = (Vector2)transform.position - new Vector2(gridSize * cellSize / 2f, gridSize * cellSize / 2f);

            for (int y = 0; y < gridSize; y++)
            {
                for (int x = 0; x < gridSize; x++)
                {
                    Vector2 cellCenter = gridOrigin + new Vector2(x * cellSize + cellSize/2, y * cellSize + cellSize/2);
                    
                    // Check for obstacles
                    bool isObstacle = Physics2D.OverlapBox(cellCenter, Vector2.one * cellSize * 0.9f, 0f, environmentLayers);
                    sensor.AddObservation(isObstacle ? 1f : 0f);
                }
            }
        }

        private void ObservePathData(VectorSensor sensor)
        {
            if (agent.hasPath)
            {
                var path = agent.path;
                sensor.AddObservation(path.corners.Length);
                
                // Observe next few corners
                int maxCorners = Mathf.Min(path.corners.Length, 3);
                for (int i = 0; i < maxCorners; i++)
                {
                    Vector3 cornerDirection = path.corners[i] - transform.position;
                    sensor.AddObservation(cornerDirection.normalized);
                    sensor.AddObservation(cornerDirection.magnitude);
                }
            }
            else
            {
                // No path - zero observations
                sensor.AddObservation(0);
                for (int i = 0; i < 3; i++)
                {
                    sensor.AddObservation(Vector3.zero);
                    sensor.AddObservation(0f);
                }
            }
        }

        private void ObserveInteractables(VectorSensor sensor)
        {
            // Similar to Brain.cs but with additional context
            int numFound = Physics2D.OverlapCircleNonAlloc(
                transform.position,
                controller.interactionRadius,
                nearbyColliders,
                interactableLayers
            );

            float closestDistance = float.MaxValue;
            Vector2 closestDirection = Vector2.zero;
            float lastInteractionTime = 0f;

            for (int i = 0; i < numFound; i++)
            {
                float distance = Vector2.Distance(transform.position, nearbyColliders[i].transform.position);
                if (distance < closestDistance)
                {
                    closestDistance = distance;
                    closestDirection = (nearbyColliders[i].transform.position - transform.position).normalized;
                    lastInteractionTimes.TryGetValue(nearbyColliders[i].gameObject, out lastInteractionTime);
                }
            }

            sensor.AddObservation(closestDistance / controller.interactionRadius);
            sensor.AddObservation(closestDirection);
            sensor.AddObservation(Time.time - lastInteractionTime);
        }

        public float[] GetObservationData()
        {
            // Calculate total observation size
            int environmentStates = 3;  // indoors, time, weather
            int npcObservations = maxDetectedObjects * 3;  // direction (2) + distance (1) per NPC
            int gridObservations = gridSize * gridSize;  // one value per grid cell
            int pathObservations = 7;  // path length (1) + 3 corners with direction and distance (2*3)
            int interactableObservations = 4;  // closest distance, direction (2), last interaction time

            int totalObservations = environmentStates + npcObservations + gridObservations + 
                                   pathObservations + interactableObservations;

            float[] observations = new float[totalObservations];
            int index = 0;

            // Environment state
            observations[index++] = IsIndoors() ? 1f : 0f;
            observations[index++] = GetTimeOfDay();
            observations[index++] = GetCurrentWeather();

            // Nearby NPCs
            int numFound = Physics2D.OverlapCircleNonAlloc(
                transform.position,
                detectionRadius,
                nearbyColliders,
                npcLayers
            );

            // Add NPC count
            observations[index++] = numFound / (float)maxDetectedObjects;

            // Nearby NPCs (up to maxDetectedObjects)
            var sortedNPCs = new List<(Vector2 direction, float distance)>();
            for (int i = 0; i < numFound && i < maxDetectedObjects; i++)
            {
                Vector2 direction = nearbyColliders[i].transform.position - transform.position;
                float distance = direction.magnitude;
                sortedNPCs.Add((direction.normalized, distance / detectionRadius));
            }

            // Sort by distance and add observations
            sortedNPCs.Sort((a, b) => a.distance.CompareTo(b.distance));
            foreach (var npc in sortedNPCs)
            {
                observations[index++] = npc.direction.x;
                observations[index++] = npc.direction.y;
                observations[index++] = npc.distance;
            }

            // Grid-based environment observations
            Vector2 gridOrigin = (Vector2)transform.position - new Vector2(gridSize * cellSize / 2f, gridSize * cellSize / 2f);

            for (int y = 0; y < gridSize; y++)
            {
                for (int x = 0; x < gridSize; x++)
                {
                    Vector2 cellCenter = gridOrigin + new Vector2(x * cellSize + cellSize/2, y * cellSize + cellSize/2);
                    
                    // Check for obstacles
                    bool isObstacle = Physics2D.OverlapBox(cellCenter, Vector2.one * cellSize * 0.9f, 0f, environmentLayers);
                    observations[index++] = isObstacle ? 1f : 0f;
                }
            }

            // Path and navigation data
            if (agent.hasPath)
            {
                var path = agent.path;
                observations[index++] = path.corners.Length;
                
                // Observe next few corners
                int maxCorners = Mathf.Min(path.corners.Length, 3);
                for (int i = 0; i < maxCorners; i++)
                {
                    Vector3 cornerDirection = path.corners[i] - transform.position;
                    observations[index++] = cornerDirection.normalized.x;
                    observations[index++] = cornerDirection.normalized.y;
                    observations[index++] = cornerDirection.magnitude;
                }
            }
            else
            {
                // No path - zero observations
                observations[index++] = 0;
                for (int i = 0; i < 3; i++)
                {
                    observations[index++] = Vector3.zero.x;
                    observations[index++] = Vector3.zero.y;
                    observations[index++] = 0f;
                }
            }

            // Interactable objects
            int numFoundInteractables = Physics2D.OverlapCircleNonAlloc(
                transform.position,
                controller.interactionRadius,
                nearbyColliders,
                interactableLayers
            );

            float closestDistanceInteractables = float.MaxValue;
            Vector2 closestDirectionInteractables = Vector2.zero;
            float lastInteractionTimeInteractables = 0f;

            for (int i = 0; i < numFoundInteractables; i++)
            {
                float distance = Vector2.Distance(transform.position, nearbyColliders[i].transform.position);
                if (distance < closestDistanceInteractables)
                {
                    closestDistanceInteractables = distance;
                    closestDirectionInteractables = (nearbyColliders[i].transform.position - transform.position).normalized;
                    lastInteractionTimeInteractables = lastInteractionTimes.TryGetValue(nearbyColliders[i].gameObject, out float lastInteractionTime) ? lastInteractionTime : 0f;
                }
            }

            observations[index++] = closestDistanceInteractables / controller.interactionRadius;
            observations[index++] = closestDirectionInteractables.x;
            observations[index++] = closestDirectionInteractables.y;
            observations[index++] = Time.time - lastInteractionTimeInteractables;

            return observations;
        }

        private void OnDrawGizmos()
        {
            if (!visualizeGrid) return;

            Vector2 gridOrigin = (Vector2)transform.position - new Vector2(gridSize * cellSize / 2f, gridSize * cellSize / 2f);
            Gizmos.color = new Color(0.2f, 1f, 0.2f, 0.2f);

            for (int y = 0; y < gridSize; y++)
            {
                for (int x = 0; x < gridSize; x++)
                {
                    Vector2 cellCenter = gridOrigin + new Vector2(x * cellSize + cellSize/2, y * cellSize + cellSize/2);
                    if (Physics2D.OverlapBox(cellCenter, Vector2.one * cellSize * 0.9f, 0f, environmentLayers))
                    {
                        Gizmos.color = new Color(1f, 0.2f, 0.2f, 0.3f);
                    }
                    else
                    {
                        Gizmos.color = new Color(0.2f, 1f, 0.2f, 0.1f);
                    }
                    Gizmos.DrawCube((Vector3)cellCenter, Vector3.one * cellSize);
                }
            }
        }
    }
}