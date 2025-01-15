using UnityEngine;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Agentics
{
    [Serializable]
    public class DiseaseParameters
    {
        public string diseaseId;
        public string diseaseName;
        public float baseInfectionRadius = 2f;
        public float baseInfectionProbability = 0.1f;
        public float infectionDuration = 300f; // in seconds
        public float immunityDuration = 600f;  // in seconds
        public bool canReinfect = true;
        public string[] immunities = new string[0]; // Other conditions that make you immune
        public float severityMultiplier = 1f;
        public bool requiresContact = false; // If true, needs direct interaction
    }

    public class DiseaseController : MonoBehaviour
    {
        private static DiseaseController instance;
        public static DiseaseController Instance
        {
            get
            {
                if (instance == null)
                {
                    instance = FindObjectOfType<DiseaseController>();
                    if (instance == null)
                    {
                        GameObject go = new GameObject("DiseaseController");
                        instance = go.AddComponent<DiseaseController>();
                    }
                }
                return instance;
            }
        }

        [SerializeField]
        private List<DiseaseParameters> diseases = new List<DiseaseParameters>();
        
        private Dictionary<string, DiseaseParameters> diseaseMap = new Dictionary<string, DiseaseParameters>();
        private List<AgenticCharacter> characters = new List<AgenticCharacter>();
        private Dictionary<AgenticCharacter, Dictionary<string, float>> infectionTimers = new Dictionary<AgenticCharacter, Dictionary<string, float>>();
        private Dictionary<AgenticCharacter, Dictionary<string, float>> immunityTimers = new Dictionary<AgenticCharacter, Dictionary<string, float>>();

        private void Awake()
        {
            if (instance != null && instance != this)
            {
                Destroy(gameObject);
                return;
            }
            
            instance = this;
            DontDestroyOnLoad(gameObject);

            // Initialize disease map
            foreach (var disease in diseases)
            {
                diseaseMap[disease.diseaseId] = disease;
            }
        }

        private void Update()
        {
            // UpdateInfectionSpread();
            // UpdateTimers();
        }

        private void UpdateInfectionSpread()
        {
            // Log total characters being checked
            Debug.Log($"Checking {characters.Count} total characters");
            
            var infectedChars = characters.Where(c => c.healthConditions.Count > 0).ToList();
            Debug.Log($"Found {infectedChars.Count} infected characters");

            foreach (var infectedChar in infectedChars)
            {
                Debug.Log($"Checking spread from {infectedChar.CharacterName} with conditions: {string.Join(", ", infectedChar.healthConditions)}");
                
                foreach (var condition in infectedChar.healthConditions)
                {
                    if (!diseaseMap.TryGetValue(condition, out DiseaseParameters disease))
                    {
                        Debug.LogWarning($"Disease {condition} not found in disease map!");
                        continue;
                    }

                    // Skip if character is quarantined
                    if (infectedChar.isQuarantined)
                    {
                        Debug.Log($"{infectedChar.CharacterName} is quarantined, skipping");
                        continue;
                    }

                    // Get all susceptible characters
                    var susceptibleChars = characters.Where(c => 
                        !c.healthConditions.Contains(condition) && 
                        !IsImmune(c, condition)).ToList();
                    
                    Debug.Log($"Found {susceptibleChars.Count} susceptible characters for {condition}");

                    foreach (var susceptibleChar in susceptibleChars)
                    {
                        float distance = Vector3.Distance(
                            infectedChar.transform.position,
                            susceptibleChar.transform.position
                        );

                        Debug.Log($"Distance between {infectedChar.CharacterName} and {susceptibleChar.CharacterName}: {distance:F1}m (radius: {disease.baseInfectionRadius}m)");

                        // For any disease with 100% infection probability, immediately infect if within radius
                        if (disease.baseInfectionProbability >= 100f && distance <= disease.baseInfectionRadius)
                        {
                            Debug.Log($"Infecting {susceptibleChar.CharacterName} with {disease.diseaseName} (100% probability within range)");
                            susceptibleChar.AddHealthCondition(condition);
                        }
                        else if (distance <= disease.baseInfectionRadius)
                        {
                            float infectionChance = disease.baseInfectionProbability / 100f;
                            float roll = UnityEngine.Random.value;
                            Debug.Log($"Rolling for infection: {roll:F3} vs chance {infectionChance:F3}");
                            
                            if (roll < infectionChance)
                            {
                                Debug.Log($"Infecting {susceptibleChar.CharacterName} with {disease.diseaseName} (probability roll succeeded)");
                                susceptibleChar.AddHealthCondition(condition);
                            }
                        }
                    }
                }
            }
        }

        private void UpdateTimers()
        {
            foreach (var character in characters.ToList())
            {
                if (!infectionTimers.ContainsKey(character))
                    continue;

                foreach (var condition in character.healthConditions.ToList())
                {
                    if (!diseaseMap.ContainsKey(condition))
                        continue;

                    // Update infection timer
                    if (infectionTimers[character].ContainsKey(condition))
                    {
                        infectionTimers[character][condition] += Time.deltaTime;
                        if (infectionTimers[character][condition] >= diseaseMap[condition].infectionDuration)
                        {
                            character.RemoveHealthCondition(condition);
                            StartImmunity(character, condition);
                        }
                    }
                }
            }

            // Update immunity timers
            foreach (var kvp in immunityTimers.ToList())
            {
                foreach (var immunity in kvp.Value.ToList())
                {
                    immunityTimers[kvp.Key][immunity.Key] += Time.deltaTime;
                    if (immunityTimers[kvp.Key][immunity.Key] >= diseaseMap[immunity.Key].immunityDuration)
                    {
                        immunityTimers[kvp.Key].Remove(immunity.Key);
                    }
                }
            }
        }

        public void RegisterCharacter(AgenticCharacter character)
        {
            if (!characters.Contains(character))
            {
                characters.Add(character);
                infectionTimers[character] = new Dictionary<string, float>();
                immunityTimers[character] = new Dictionary<string, float>();
                Debug.Log($"Registered character {character.CharacterName} with DiseaseController. Total characters: {characters.Count}");
            }
        }

        public void UnregisterCharacter(AgenticCharacter character)
        {
            characters.Remove(character);
            infectionTimers.Remove(character);
            immunityTimers.Remove(character);
        }

        public void OnCharacterInfected(AgenticCharacter character, string condition)
        {
            if (!infectionTimers.ContainsKey(character))
                infectionTimers[character] = new Dictionary<string, float>();
                
            infectionTimers[character][condition] = 0f;
        }

        public void OnCharacterRecovered(AgenticCharacter character, string condition)
        {
            if (infectionTimers.ContainsKey(character))
                infectionTimers[character].Remove(condition);
        }

        private void StartImmunity(AgenticCharacter character, string condition)
        {
            if (!immunityTimers.ContainsKey(character))
                immunityTimers[character] = new Dictionary<string, float>();

            immunityTimers[character][condition] = 0f;
        }

        private bool IsImmune(AgenticCharacter character, string condition)
        {
            // Check temporary immunity
            if (immunityTimers.ContainsKey(character) && 
                immunityTimers[character].ContainsKey(condition))
                return true;

            // Check disease-granted immunities
            if (diseaseMap.TryGetValue(condition, out DiseaseParameters disease))
            {
                foreach (string immunity in disease.immunities)
                {
                    if (character.HasCondition(immunity))
                        return true;
                }
            }

            return false;
        }
    }
} 