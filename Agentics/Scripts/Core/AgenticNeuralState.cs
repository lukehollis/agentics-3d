using UnityEngine;
using Unity.Collections;
using Unity.Jobs;
using System.Collections.Generic;

namespace Agentics
{
    /// <summary>
    /// Manages the neural state for a single character
    /// </summary>
    public class AgenticNeuralState : MonoBehaviour
    {
        [Header("State Configuration")]
        [SerializeField] private float stateUpdateInterval = 0.1f;
        
        [Header("Debug Visualization")]
        [SerializeField] private bool showDebugUI = false;
        [SerializeField] private bool logStateChanges = false;

        // Core references
        private AgenticCharacter character;
        private AgenticController controller;

        // Neural state buffers
        private NativeArray<float> currentState;
        private NativeArray<float> latentState;

        // State dimensions
        private const int RAW_DIMENSIONS = 256;
        private const int LATENT_DIMENSIONS = 32;

        // Major state categories for easy access
        public struct CharacterNeeds
        {
            public float energy;      // 0-100
            public float hunger;      // 0-100
            public float social;      // 0-100
            public float comfort;     // 0-100
            public float mood;        // -100 to 100
        }

        public CharacterNeeds needs;

        private void Awake()
        {
            character = GetComponent<AgenticCharacter>();
            controller = GetComponent<AgenticController>();

            // Initialize native arrays
            currentState = new NativeArray<float>(RAW_DIMENSIONS, Allocator.Persistent);
            latentState = new NativeArray<float>(LATENT_DIMENSIONS, Allocator.Persistent);

            // Initialize with default values
            InitializeDefaultState();
        }

        private void OnEnable()
        {
            // Start state update cycle
            InvokeRepeating(nameof(UpdateNeuralState), 0f, stateUpdateInterval);
        }

        private void OnDisable()
        {
            CancelInvoke(nameof(UpdateNeuralState));
        }

        private void OnDestroy()
        {
            // Clean up native arrays
            if (currentState.IsCreated) currentState.Dispose();
            if (latentState.IsCreated) latentState.Dispose();
        }

        private void InitializeDefaultState()
        {
            // Set initial values based on character stats
            needs = new CharacterNeeds
            {
                energy = character.Stamina * 10f,
                hunger = 70f, // Start somewhat hungry
                social = 50f, // Neutral social need
                comfort = 60f,
                mood = 0f // Neutral mood
            };

            // Pack needs into state array
            PackNeedsIntoState();
        }

        private void PackNeedsIntoState()
        {
            // Pack core needs into the start of the state array
            currentState[0] = needs.energy / 100f;
            currentState[1] = needs.hunger / 100f;
            currentState[2] = (needs.social + 100f) / 200f; // Normalize -100 to 100 to 0 to 1
            currentState[3] = needs.comfort / 100f;
            currentState[4] = (needs.mood + 100f) / 200f;

            // Pack character stats
            currentState[5] = character.Money / 100f;
            currentState[6] = character.Health / 10f;
            currentState[7] = character.Stamina / 10f;
        }

        private void UnpackStateToNeeds()
        {
            // Unpack core needs
            needs.energy = currentState[0] * 100f;
            needs.hunger = currentState[1] * 100f;
            needs.social = (currentState[2] * 200f) - 100f;
            needs.comfort = currentState[3] * 100f;
            needs.mood = (currentState[4] * 200f) - 100f;
        }

        private void UpdateNeuralState()
        {
            // Schedule neural state update job
            var job = new NeuralStateUpdateJob
            {
                currentState = currentState,
                latentState = latentState,
                deltaTime = stateUpdateInterval,
                
                // Environment influences
                timeOfDay = System.DateTime.Now.Hour / 24f,
                isIndoors = Physics2D.OverlapPoint(transform.position, LayerMask.GetMask("Indoors")) != null,
                nearbyNPCs = Physics2D.OverlapCircleNonAlloc(transform.position, 5f, new Collider2D[10], 
                    LayerMask.GetMask("NPC"))
            };

            // Schedule and complete immediately (for now)
            job.Schedule().Complete();

            // Update needs from new state
            UnpackStateToNeeds();

            // Update character behavior based on new state
            UpdateCharacterBehavior();

            if (logStateChanges)
            {
                Debug.Log($"[{character.CharacterName}] Updated needs - Energy: {needs.energy:F1}, " +
                         $"Hunger: {needs.hunger:F1}, Social: {needs.social:F1}, Mood: {needs.mood:F1}");
            }
        }

        private void UpdateCharacterBehavior()
        {
            // Update animation state based on energy
            if (needs.energy < 20f)
            {
                // character.SetState(PixelHeroAnimationState.Dead); // Use for exhausted
                controller.SpeedMultiplier = 0.3f; // Severely reduced speed
            }
            else if (needs.energy < 40f)
            {
                controller.SpeedMultiplier = 0.7f; // Reduced speed
            }
            else
            {
                controller.SpeedMultiplier = 1f; // Normal speed
            }

            // Update day plan based on urgent needs
            if (needs.hunger > 80f)
            {
                // Interrupt current plan to find food
                TryFindFood();
            }

            // Affect social interactions
            if (needs.mood < -50f)
            {
                // Maybe avoid social interactions
                controller.interactionRadius *= 0.5f;
            }
        }

        private void TryFindFood()
        {
            // Example of how needs affect behavior
            var foodSources = GameObject.FindGameObjectsWithTag("FoodSource");
            if (foodSources.Length > 0)
            {
                // Find closest food source
                GameObject closest = null;
                float closestDist = float.MaxValue;
                foreach (var food in foodSources)
                {
                    float dist = Vector3.Distance(transform.position, food.transform.position);
                    if (dist < closestDist)
                    {
                        closest = food;
                        closestDist = dist;
                    }
                }

                if (closest != null)
                {
                    controller.SetDestination(closest.transform.position);
                }
            }
        }

        private void OnGUI()
        {
            if (showDebugUI)
            {
                Vector2 screenPos = Camera.main.WorldToScreenPoint(transform.position + Vector3.up);
                Rect rect = new Rect(screenPos.x - 50, Screen.height - screenPos.y - 60, 100, 50);
                
                GUI.Box(rect, "");
                GUI.Label(rect, $"Energy: {needs.energy:F0}\n" +
                               $"Hunger: {needs.hunger:F0}\n" +
                               $"Mood: {needs.mood:F0}");
            }
        }

#if UNITY_EDITOR
        private void OnDrawGizmosSelected()
        {
            // Visualize mood radius
            Gizmos.color = needs.mood > 0 ? Color.green : Color.red;
            Gizmos.DrawWireSphere(transform.position, Mathf.Abs(needs.mood) / 50f);
        }
#endif
    }

    /// <summary>
    /// Job struct for updating neural state
    /// </summary>
    public struct NeuralStateUpdateJob : IJob
    {
        public NativeArray<float> currentState;
        public NativeArray<float> latentState;
        
        // Delta time for updates
        public float deltaTime;
        
        // Environmental inputs
        public float timeOfDay;
        public bool isIndoors;
        public int nearbyNPCs;

        public void Execute()
        {
            // Basic needs decay
            currentState[0] -= deltaTime * 0.1f; // Energy decreases slowly
            currentState[1] += deltaTime * 0.05f; // Hunger increases slowly
            
            // Social need affected by nearby NPCs
            float socialChange = (nearbyNPCs > 0) ? 0.1f : -0.05f;
            currentState[2] += deltaTime * socialChange;

            // Comfort affected by environment
            float comfortChange = isIndoors ? 0.1f : -0.05f;
            currentState[3] += deltaTime * comfortChange;

            // Mood affected by all other needs
            float moodInfluence = 0f;
            moodInfluence += (currentState[0] - 0.5f) * 0.3f; // Energy
            moodInfluence += (0.7f - currentState[1]) * 0.3f; // Hunger (inverse)
            moodInfluence += (currentState[2] - 0.5f) * 0.2f; // Social
            moodInfluence += (currentState[3] - 0.5f) * 0.2f; // Comfort
            
            currentState[4] = Mathf.Lerp(currentState[4], 
                                       Mathf.Clamp01(0.5f + moodInfluence), 
                                       deltaTime);

            // Clamp all values
            for (int i = 0; i < currentState.Length; i++)
            {
                currentState[i] = Mathf.Clamp01(currentState[i]);
            }
        }
    }

    /// <summary>
    /// Manager for handling batch updates of multiple characters
    /// </summary>
    public class AgenticNeuralStateManager : MonoBehaviour
    {
        private static AgenticNeuralStateManager instance;
        public static AgenticNeuralStateManager Instance => instance;

        private List<AgenticNeuralState> activeCharacters = new List<AgenticNeuralState>();
        
        private void Awake()
        {
            if (instance == null)
            {
                instance = this;
                DontDestroyOnLoad(gameObject);
            }
            else
            {
                Destroy(gameObject);
            }
        }

        public void RegisterCharacter(AgenticNeuralState character)
        {
            if (!activeCharacters.Contains(character))
            {
                activeCharacters.Add(character);
            }
        }

        public void UnregisterCharacter(AgenticNeuralState character)
        {
            activeCharacters.Remove(character);
        }
    }
}