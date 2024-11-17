using UnityEngine;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Agentics;

namespace Agentics
{
    public class ConsciousnessSystem : MonoBehaviour
    {
        [Header("Consciousness Settings")]
        [SerializeField] private float introspectionInterval = 0.5f;
        [SerializeField] private float thoughtDecayRate = 0.1f;
        [SerializeField] private int maxActiveThoughts = 5;
        [SerializeField] private float minThoughtSalience = 0.3f;

        private MotivationSystem motivationSystem;
        private Brain agentBrain;
        private Sensor agentSensor;
        private InnerState innerState = new InnerState();
        private float lastIntrospectionTime;

        [System.Serializable]
        public class ThoughtPattern
        {
            public float salience;          // How "loud" this thought is (0-1)
            public string content;          // The actual thought
            public float emotionalCharge;   // Emotional intensity (-1 to 1)
            public string[] associations;   // Related concepts
            public float duration;          // How long this thought persists
            public float creationTime;      // When the thought was created
        }

        public class InnerState
        {
            public List<ThoughtPattern> activeThoughts = new List<ThoughtPattern>();
            public Dictionary<string, float> conceptActivation = new Dictionary<string, float>();
            public Queue<string> shortTermMemory = new Queue<string>(10);
            public float mentalClarity;     // Overall clarity of thinking (0-1)
            public float attentionFocus;    // Current focus level (0-1)
            public Vector3 attentionPoint;  // World position of current focus
        }

        private void Awake()
        {
            motivationSystem = GetComponent<MotivationSystem>();
            agentBrain = GetComponent<Brain>();
            agentSensor = GetComponent<Sensor>();
        }

        private void Update()
        {
            if (Time.time - lastIntrospectionTime >= introspectionInterval)
            {
                UpdateConsciousness();
                lastIntrospectionTime = Time.time;
            }
        }

        private void UpdateConsciousness()
        {
            UpdateThoughts();
            ProcessSensoryInput();
            IntegrateMotivationalState();
            UpdateAttentionalFocus();
            GenerateInnerNarrative();
        }

        private void UpdateThoughts()
        {
            // Update existing thoughts
            for (int i = innerState.activeThoughts.Count - 1; i >= 0; i--)
            {
                var thought = innerState.activeThoughts[i];
                
                // Decay thought salience over time
                thought.salience -= thoughtDecayRate * Time.deltaTime;
                
                // Remove thoughts that are no longer salient
                if (thought.salience < minThoughtSalience || 
                    Time.time - thought.creationTime > thought.duration)
                {
                    innerState.activeThoughts.RemoveAt(i);
                }
            }

            // Ensure we don't exceed max active thoughts
            while (innerState.activeThoughts.Count > maxActiveThoughts)
            {
                // Remove least salient thought
                int leastSalientIndex = FindLeastSalientThoughtIndex();
                innerState.activeThoughts.RemoveAt(leastSalientIndex);
            }
        }

        private void ProcessSensoryInput()
        {
            // Get sensory observations from Sensor
            var gridState = agentSensor.GetObservationGrid();
            
            // Process environmental awareness
            foreach (var state in gridState)
            {
                if (state.hasInteractable)
                {
                    AddThought(new ThoughtPattern
                    {
                        content = $"I notice {state.interactableType} nearby",
                        salience = 1f - (state.distanceToAgent / agentSensor.ViewRadius),
                        emotionalCharge = 0.2f,
                        duration = 3f,
                        creationTime = Time.time
                    });
                }
            }
        }

        private void IntegrateMotivationalState()
        {
            var motivationalContext = motivationSystem.GetMotivationalContext();
            
            // Create thoughts based on strongest motivational factors
            foreach (var need in GetSignificantNeeds(motivationalContext))
            {
                AddThought(new ThoughtPattern
                {
                    content = $"I feel a strong need for {need.Key}",
                    salience = need.Value,
                    emotionalCharge = need.Value > 0.7f ? 0.8f : 0.4f,
                    duration = 5f,
                    creationTime = Time.time
                });
            }
        }

        private void UpdateAttentionalFocus()
        {
            // Update attention based on current goals and sensory input
            if (agentBrain.HasCurrentGoal())
            {
                innerState.attentionPoint = agentBrain.GetCurrentGoalPosition();
                innerState.attentionFocus = 0.8f;
            }
            else
            {
                // Default to most salient sensory input
                var mostSalientThought = GetMostSalientThought();
                if (mostSalientThought != null)
                {
                    innerState.attentionFocus = mostSalientThought.salience;
                }
                else
                {
                    innerState.attentionFocus = 0.3f; // Base attention level
                }
            }
        }

        public void AddThought(ThoughtPattern thought)
        {
            if (innerState.activeThoughts.Count < maxActiveThoughts)
            {
                innerState.activeThoughts.Add(thought);
            }
            else
            {
                // Replace least salient thought if new thought is more salient
                int leastSalientIndex = FindLeastSalientThoughtIndex();
                if (innerState.activeThoughts[leastSalientIndex].salience < thought.salience)
                {
                    innerState.activeThoughts[leastSalientIndex] = thought;
                }
            }
        }

        private void GenerateInnerNarrative()
        {
            var narrative = new StringBuilder();
            
            // Add current focus
            narrative.AppendLine($"Attention: {(innerState.attentionFocus > 0.7f ? "Focused" : "Wandering")}");
            
            // Add most salient thoughts
            foreach (var thought in innerState.activeThoughts.OrderByDescending(t => t.salience))
            {
                narrative.AppendLine($"[{thought.salience:F2}] {thought.content}");
            }

            // Store in short-term memory
            innerState.shortTermMemory.Enqueue(narrative.ToString());
            if (innerState.shortTermMemory.Count > 10)
            {
                innerState.shortTermMemory.Dequeue();
            }
        }

        private int FindLeastSalientThoughtIndex()
        {
            float minSalience = float.MaxValue;
            int minIndex = 0;
            
            for (int i = 0; i < innerState.activeThoughts.Count; i++)
            {
                if (innerState.activeThoughts[i].salience < minSalience)
                {
                    minSalience = innerState.activeThoughts[i].salience;
                    minIndex = i;
                }
            }
            
            return minIndex;
        }

        private ThoughtPattern GetMostSalientThought()
        {
            return innerState.activeThoughts
                .OrderByDescending(t => t.salience)
                .FirstOrDefault();
        }

        private Dictionary<string, float> GetSignificantNeeds(float[] motivationalContext)
        {
            var needs = new Dictionary<string, float>();
            string[] needTypes = { "rest", "hunger", "social", "achievement" };
            
            for (int i = 0; i < needTypes.Length && i < motivationalContext.Length; i++)
            {
                if (motivationalContext[i] > 0.6f) // Only include significant needs
                {
                    needs.Add(needTypes[i], motivationalContext[i]);
                }
            }
            
            return needs;
        }

        // Interface for ML-Agents
        public void AddToObservations(VectorSensor sensor)
        {
            sensor.AddObservation(innerState.mentalClarity);
            sensor.AddObservation(innerState.attentionFocus);
            sensor.AddObservation(innerState.activeThoughts.Count / (float)maxActiveThoughts);
            
            // Add most salient thought's emotional charge
            var mostSalientThought = GetMostSalientThought();
            sensor.AddObservation(mostSalientThought?.emotionalCharge ?? 0f);
        }

        public float[] GetConsciousnessState()
        {
            var state = new List<float>();
            
            // Add basic consciousness metrics
            state.Add(innerState.mentalClarity);
            state.Add(innerState.attentionFocus);
            
            // Add thought-related metrics
            state.Add(innerState.activeThoughts.Count / (float)maxActiveThoughts);
            
            // Add emotional state from most salient thought
            var mostSalientThought = GetMostSalientThought();
            state.Add(mostSalientThought?.emotionalCharge ?? 0f);
            
            // Add attention point
            state.Add(innerState.attentionPoint.x);
            state.Add(innerState.attentionPoint.y);
            state.Add(innerState.attentionPoint.z);
            
            return state.ToArray();
        }
    }
}