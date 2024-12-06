using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using UnityEngine.AI;
using Agentics.Core;

namespace Agentics
{
    public class CharacterStateSensor : MonoBehaviour
    {
        [Header("References")]
        private AgenticCharacter character;
        private AgenticController controller;
        private AgenticNeuralState neuralState;
        private NavMeshAgent agent;
        private Brain agentBrain;

        [Header("Observation Settings")]
        [SerializeField] private float maxPathDistance = 20f;
        [SerializeField] private bool includeNavigationState = true;

        private void Awake()
        {
            character = GetComponent<AgenticCharacter>();
            controller = GetComponent<AgenticController>();
            neuralState = GetComponent<AgenticNeuralState>();
            agent = GetComponent<NavMeshAgent>();
            agentBrain = GetComponent<Brain>();
        }

        public void CollectObservations(VectorSensor sensor)
        {
            // Character needs observations
            sensor.AddObservation(neuralState.needs.energy / 100f);
            sensor.AddObservation(neuralState.needs.hunger / 100f);
            sensor.AddObservation(neuralState.needs.social / 100f);
            sensor.AddObservation(neuralState.needs.comfort / 100f);
            sensor.AddObservation((neuralState.needs.mood + 100f) / 200f); // Normalize -100 to 100 range

            // Character state observations
            sensor.AddObservation(character.Health / 100f);
            sensor.AddObservation(character.Stamina / 100f);
            sensor.AddObservation(controller.isInteracting);
            sensor.AddObservation(controller.sleepWakeMode == "sleep");

            // Navigation state (if enabled)
            if (includeNavigationState)
            {
                sensor.AddObservation(agent.velocity.normalized);
                sensor.AddObservation(agent.hasPath);
                sensor.AddObservation(agent.remainingDistance / maxPathDistance);
            }
        }

        public float[] GetObservationData()
        {
            // Calculate total observation size
            int needsObservations = 5;
            int stateObservations = 4;
            int navigationObservations = includeNavigationState ? 4 : 0;
            int totalObservations = needsObservations + stateObservations + navigationObservations;

            float[] observations = new float[totalObservations];
            int index = 0;

            // Add needs observations
            observations[index++] = neuralState.needs.energy / 100f;
            observations[index++] = neuralState.needs.hunger / 100f;
            observations[index++] = neuralState.needs.social / 100f;
            observations[index++] = neuralState.needs.comfort / 100f;
            observations[index++] = (neuralState.needs.mood + 100f) / 200f;

            // Add state observations
            observations[index++] = character.Health / 100f;
            observations[index++] = character.Stamina / 100f;
            observations[index++] = controller.isInteracting ? 1f : 0f;
            observations[index++] = controller.sleepWakeMode == "sleep" ? 1f : 0f;

            // Add navigation observations if enabled
            if (includeNavigationState)
            {
                observations[index++] = agent.velocity.normalized.x;
                observations[index++] = agent.velocity.normalized.y;
                observations[index++] = agent.hasPath ? 1f : 0f;
                observations[index++] = agent.remainingDistance / maxPathDistance;
            }

            return observations;
        }

        private void OnValidate()
        {
            if (maxPathDistance <= 0)
                maxPathDistance = 20f;
        }
    }
}