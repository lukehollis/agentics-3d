using UnityEngine;
using UnityEngine.AI;
using System.Collections;
using System.Linq;
using System.Collections.Generic;

namespace Agentics 
{
    public class TransportationController : MonoBehaviour 
    {

        [Header("Transportation Settings")]
        public float trainPreferenceWeight = 0.7f; // Higher value = more likely to take train
        public bool enablePathVisualization = false; // Default to false for performance
        
        [Header("References")]
        public GameObject carGeometry;
        public Car car;
        public GameObject characterGeometry;  // Reference to the character's visual mesh
        public AgenticController character;
        public CharacterController characterController;
        public LineRenderer pathLineRenderer;
        private Vector3 destination;
        private TransportMode currentMode = TransportMode.Walking;
        private BartStation targetStation;
        private Train targetTrain;
        private Vector3 debugDestinationRoadPoint;
        private bool isJourneyInProgress = false;
        private Coroutine currentJourneyCoroutine;

        public enum TransportMode 
        {
            Walking,
            Train,
            Car
        }

        private void Awake()
        {
            character = GetComponent<AgenticController>();
            characterController = GetComponent<CharacterController>();
        }

        public IEnumerator TravelToDestination(string destinationType, Vector3 targetPosition)
        {
            isJourneyInProgress = true;
            destination = targetPosition;
            currentMode = DecideTransportationMode(targetPosition);

            Debug.Log($"Starting journey to {destinationType} via {currentMode}");
            
            UpdatePathVisualization();

            currentJourneyCoroutine = StartCoroutine(UpdatedJourney(currentMode));
            yield return currentJourneyCoroutine;

            isJourneyInProgress = false;
            currentJourneyCoroutine = null;
            
            pathLineRenderer.positionCount = 0;
        }

        private TransportMode DecideTransportationMode(Vector3 targetPosition)
        {
            // Find nearest stations
            BartStation nearestStation = FindNearestStation(transform.position);
            BartStation destinationStation = FindNearestStation(targetPosition);

            // Calculate estimated times for each mode
            float walkingTime = CalculateWalkingTime(targetPosition);
            float carTime = CalculateCarTime(targetPosition);
            float trainTime = float.MaxValue; // Default to max if no train route available

            if (nearestStation != null && destinationStation != null)
            {
                trainTime = CalculateTrainTime(nearestStation, destinationStation);
            }

            // return TransportMode.Walking;

            // Choose fastest mode
            if (walkingTime <= carTime && walkingTime <= trainTime)
                return TransportMode.Walking;
            else if (carTime <= trainTime)
                return TransportMode.Car;
            else
                return TransportMode.Train;
        }

        private float CalculateWalkingTime(Vector3 targetPosition)
        {
            float distance = Vector3.Distance(transform.position, targetPosition);
            float walkingSpeed = 1.4f; // Average walking speed in m/s (about 5 km/h)
            return distance / walkingSpeed;
        }

        private float CalculateCarTime(Vector3 targetPosition)
        {
            // Find nearest roads
            Road startRoad = FindNearestRoad(transform.position);
            Road endRoad = FindNearestRoad(targetPosition);
            
            if (startRoad == null || endRoad == null)
                return float.MaxValue;

            // Calculate components of the journey
            float walkToCarDistance = Vector3.Distance(transform.position, startRoad.GetNearestPoint(transform.position));
            float carDistance = Vector3.Distance(startRoad.GetNearestPoint(transform.position), 
                                            endRoad.GetNearestPoint(targetPosition));
            float walkFromCarDistance = Vector3.Distance(endRoad.GetNearestPoint(targetPosition), targetPosition);

            // Calculate times for each segment
            float walkToCarTime = walkToCarDistance / 1.4f;  // Walking speed 1.4 m/s
            float carTime = carDistance / 13.9f;             // Car speed 13.9 m/s (about 50 km/h average in city)
            float walkFromCarTime = walkFromCarDistance / 1.4f;

            return walkToCarTime + carTime + walkFromCarTime;
        }

        private float CalculateTrainTime(BartStation nearStation, BartStation destStation)
        {
            // Calculate walking segments
            float walkToStationDistance = Vector3.Distance(transform.position, nearStation.transform.position);
            float walkFromStationDistance = Vector3.Distance(destStation.transform.position, destination);
            
            // Estimate train travel time (assuming average speed of 27.8 m/s or 100 km/h)
            float trainDistance = Vector3.Distance(nearStation.transform.position, destStation.transform.position);
            
            // Calculate total time
            float walkToStationTime = walkToStationDistance / 1.4f;
            float trainTime = trainDistance / 35f;
            float walkFromStationTime = walkFromStationDistance / 1.4f;
            
            // Add average wait time for train 0 seconds
            float averageWaitTime = 0f; 

            return walkToStationTime + averageWaitTime + trainTime + walkFromStationTime;
        }

        private IEnumerator TakeTrainJourney()
        {
            // 1. Walk to nearest station's platform
            BartStation nearestStation = FindNearestStation(transform.position);
            if (nearestStation.platform == null)
            {
                Debug.LogError("Station has no platform assigned!");
                yield break;
            }

            Debug.Log("Walking to platform: " + nearestStation.platform.position);
            
            // Walk to platform
            yield return StartCoroutine(character.SetDestinationCoroutine(nearestStation.platform.position));

            // 2. Wait for next train going to our destination
            BartStation destinationStation = FindNearestStation(destination);

            Debug.Log("Waiting for train to destination: " + destinationStation.name);
            yield return StartCoroutine(WaitForTrain(nearestStation, destinationStation));

            Debug.Log("Boarding train...");

            // 3. Board and ride train
            yield return StartCoroutine(RideTrainToDestination(nearestStation, destinationStation));

            Debug.Log("Disembarking at destination: " + destinationStation.name);

            // 4. Walk from destination platform to final destination
            yield return StartCoroutine(character.SetDestinationCoroutine(destination));
        }

        private IEnumerator WaitForTrain(BartStation currentStation, BartStation targetStation)
        {
            // Find train heading to our destination
            Train nextTrain = null;
            float waitTime = 0f;
            float maxWaitTime = 9999f;

            while (nextTrain == null && waitTime < maxWaitTime)
            {
                // Look for trains serving both stations
                Train[] trains = GameObject.FindObjectsOfType<Train>();
                
                foreach (var train in trains)
                {
                    // Check if train's route includes both stations
                    if (train.ServesStations(currentStation, targetStation))
                    {
                        nextTrain = train;
                        targetTrain = train;
                        break;
                    }
                }

                if (nextTrain == null)
                {
                    waitTime += Time.deltaTime;
                    yield return null;
                }
            }

            if (nextTrain == null)
            {
                Debug.LogWarning("No train found after maximum wait time");
                yield break;
            }

            // Wait for train to arrive at station
            while (nextTrain.GetCurrentStation() != currentStation)
            {
                yield return new WaitForSeconds(1f); // Log every second instead of every frame
            }
        }

        private IEnumerator RideTrainToDestination(BartStation fromStation, BartStation toStation)
        {
            Debug.Log("Riding train to destination: " + toStation.name);
            targetStation = toStation;
            
            // Find the train we're riding
            Train currentTrain = targetTrain;
            
            // Board the train
            Debug.Log($"Boarding train: {character.name}");
            BoardTrain(currentTrain);

            // Wait until we reach destination
            while (true)
            {
                BartStation currentStation = currentTrain.GetCurrentStation();
                if (currentStation == toStation)
                {
                    break;
                }
                Debug.Log($"Waiting for train to reach {toStation.name}. Currently at: {currentStation?.name}");
                yield return new WaitForSeconds(0.1f);  // Check every 0.1 seconds
            }

            // Disembark
            Debug.Log($"Disembarking at {toStation.name}: {character.name}");
            DisembarkTrain(currentTrain, toStation);

            targetStation = null;
            targetTrain = null;
        }

        private void BoardTrain(Train train)
        {
            // Disable both controllers to prevent conflicts with parenting
            characterController.enabled = false;
            
            Debug.Log($"Before boarding - World Pos: {transform.position}, Parent: {transform.parent?.name}");
            
            // Set parent with worldPositionStays = false to maintain local position relative to train
            transform.SetParent(train.transform, false);
            
            Debug.Log($"After setting parent - World Pos: {transform.position}, Local Pos: {transform.localPosition}, Parent: {transform.parent?.name}");
            
            // Optionally, you might want to set a specific position within the train
            transform.localPosition = Vector3.zero; // Or any designated passenger position
            
            Debug.Log($"After setting local pos - World Pos: {transform.position}, Local Pos: {transform.localPosition}, Parent: {transform.parent?.name}");
            
            characterGeometry.SetActive(false);
            Debug.Log($"Successfully boarded train: {character.name}");
        }


        private void DisembarkTrain(Train train, BartStation station)
        {
            // Find or create the Humans parent object
            GameObject humansParent = GameObject.Find("Humans");
            if (humansParent == null)
            {
                humansParent = new GameObject("Humans");
            }

            // First set the position, then change parent
            transform.position = station.platform.position;
            transform.SetParent(humansParent.transform);
            
            // Re-enable both controllers
            characterController.enabled = true;
            
            characterGeometry.SetActive(true);
            Debug.Log($"Successfully disembarked at {station.name}: {character.name}");
        }

        private IEnumerator TakeCarJourney()
        {
            // 1. Walk to nearest road
            Road nearestRoad = FindNearestRoad(transform.position);
            Vector3 nearestRoadPoint = nearestRoad.GetNearestPoint(transform.position);
            yield return StartCoroutine(character.SetDestinationCoroutine(nearestRoadPoint));
            
            // 2. Switch to car and disable controllers
            characterGeometry.SetActive(false);
            characterController.enabled = false;
            
            carGeometry.SetActive(true);
            car.enabled = true;
            car.road = nearestRoad;
            
            // 3. Calculate start and target positions
            Road destinationRoad = FindNearestRoad(destination);
            Vector3 destinationRoadPoint = destinationRoad.GetNearestPoint(destination);
            debugDestinationRoadPoint = destinationRoadPoint;
            
            UpdatePathVisualization();

            // Start the journey with world positions
            car.StartJourneyToPositions(nearestRoadPoint, destinationRoadPoint);
            
            // Wait until car has reached destination
            while (!car.HasReachedDestination())
            {
                yield return null;
            }

            // 4. Switch back to character
            transform.SetParent(null);
            carGeometry.SetActive(false);
            car.road = null;
            car.enabled = false;
            characterGeometry.SetActive(true);
            
            // Re-enable controllers
            characterController.enabled = true;

            // 5. Walk to final destination
            yield return StartCoroutine(character.SetDestinationCoroutine(destination));
        }

        private Vector3 GetNearestPointOnSegment(Vector3 point, Vector3 start, Vector3 end)
        {
            Vector3 segment = end - start;
            Vector3 pointVector = point - start;
            
            float segmentLength = segment.magnitude;
            Vector3 segmentDirection = segment / segmentLength;
            
            float projection = Vector3.Dot(pointVector, segmentDirection);
            
            if (projection <= 0)
                return start;
            if (projection >= segmentLength)
                return end;
            
            return start + segmentDirection * projection;
        }

        // Helper methods to be implemented
        private BartStation FindNearestStation(Vector3 position)
        {
            BartStation[] stations = GameObject.FindObjectsOfType<BartStation>();
            BartStation nearest = null;
            float nearestDistance = float.MaxValue;

            foreach (var station in stations)
            {
                float distance = Vector3.Distance(position, station.transform.position);
                if (distance < nearestDistance)
                {
                    nearest = station;
                    nearestDistance = distance;
                }
            }

            return nearest;
        }

        private Road FindNearestRoad(Vector3 position)
        {
            Road[] roads = GameObject.FindObjectsOfType<Road>();
            Road nearest = null;
            float nearestDistance = float.MaxValue;

            foreach (var road in roads)
            {
                float distance = Vector3.Distance(position, road.GetNearestPoint(position));
                if (distance < nearestDistance)
                {
                    nearest = road;
                    nearestDistance = distance;
                }
            }

            return nearest;
        }

        public bool IsWaitingForTrain(Train train)
        {
            return train == targetTrain;
        }

        public bool ShouldDisembarkAt(BartStation station)
        {
            return station == targetStation;
        }

        private void UpdatePathVisualization()
        {
            if (!enablePathVisualization || pathLineRenderer == null)
            {
                return;
            }

            List<Vector3> pathPoints = new List<Vector3>();
            
            // Always use current position as first point
            pathPoints.Add(transform.position);

            switch (currentMode)
            {
                case TransportMode.Train:
                    // Add path to nearest station
                    BartStation nearestStation = FindNearestStation(transform.position);
                    if (nearestStation != null && nearestStation.platform != null)
                    {
                        // Only add station point if we're not already very close to it
                        if (Vector3.Distance(transform.position, nearestStation.platform.position) > 1f)
                        {
                            pathPoints.Add(nearestStation.platform.position);
                        }
                        
                        // Add path to destination station
                        BartStation destinationStation = FindNearestStation(destination);
                        if (destinationStation != null && destinationStation.platform != null)
                        {
                            pathPoints.Add(destinationStation.platform.position);
                        }
                    }
                    break;

                case TransportMode.Car:
                    // Add path to nearest road point
                    Road startRoad = FindNearestRoad(transform.position);
                    if (startRoad != null)
                    {
                        Vector3 nearestRoadPoint = startRoad.GetNearestPoint(transform.position);
                        // Only add road point if we're not already very close to it
                        if (Vector3.Distance(transform.position, nearestRoadPoint) > 1f)
                        {
                            pathPoints.Add(nearestRoadPoint);
                        }
                        
                        // Add path to destination road point
                        Road endRoad = FindNearestRoad(destination);
                        if (endRoad != null)
                        {
                            Vector3 destinationRoadPoint = endRoad.GetNearestPoint(destination);
                            pathPoints.Add(destinationRoadPoint);
                        }
                    }
                    break;
            }

            // Only add final destination if we're not already very close to it
            if (Vector3.Distance(transform.position, destination) > 1f)
            {
                pathPoints.Add(destination);
            }

            // Update line renderer
            pathLineRenderer.positionCount = pathPoints.Count;
            for (int i = 0; i < pathPoints.Count; i++)
            {
                pathLineRenderer.SetPosition(i, pathPoints[i]);
            }
        }

        private void Update()
        {
            if (isJourneyInProgress)
            {
                UpdatePathVisualization();  // Update every frame while journey is in progress
            }
        }

        public void OnStationMoved(BartStation movedStation)
        {
            Debug.Log("Station moved: " + movedStation.name);
            Debug.Log("Recalculating transportation mode");
            // Recalculate transportation mode
            TransportMode newMode = DecideTransportationMode(destination);
            
            if (newMode != currentMode)
            {
                // Stop current journey
                if (currentJourneyCoroutine != null)
                    StopCoroutine(currentJourneyCoroutine);

                // Start new journey with updated mode
                currentJourneyCoroutine = StartCoroutine(UpdatedJourney(newMode));
                
                // Add this line to update the visualization
                UpdatePathVisualization();
            }
        }

        private IEnumerator UpdatedJourney(TransportMode newMode)
        {
            Debug.Log($"Updating journey to mode: {newMode}");
            currentMode = newMode;

            // Save current position as starting point
            Vector3 currentPos = transform.position;

            switch (newMode)
            {
                case TransportMode.Train:
                    yield return StartCoroutine(TakeTrainJourney());
                    break;

                case TransportMode.Car:
                    yield return StartCoroutine(TakeCarJourney());
                    break;

                default:
                    yield return StartCoroutine(character.SetDestinationCoroutine(destination));
                    break;
            }
        }
    }
}