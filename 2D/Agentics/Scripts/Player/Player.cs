using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.EventSystems;

#if UNITY_EDITOR

using UnityEditor;

#endif




namespace Agentics
{
    /// <summary>
    /// Main player character class that handles movement, animations, combat state, and texture management.
    /// Manages both keyboard/mouse input and NavMesh pathfinding for movement and interactions.
    /// </summary>
    /// <remarks>
    /// The animator's direction parameter uses the following integer values:
    /// - 0: Right
    /// - 1: Left  
    /// - 2: Up
    /// - 3: Down
    /// 
    /// This differs from the Direction enum order to match the sprite sheet layout and animation system.
    /// The Direction enum is used for internal logic while the animator parameter handles the actual
    /// animation states.
    /// </remarks>
    public enum Direction
    {
        right,
        left,
        up,
        down,
        none
    }

    public enum MoveType
    {
        idle,
        walking,
        running
    }

    public class Player : MonoBehaviour
    {
        private float xInput, yInput, movementSpeed;
        private MoveType moveType;
        private Direction playerDirection;

        private float runningSpeed = 3.5f;
        private float walkingSpeed = 1.75f;

        public GameObject hat;
        public GameObject hair;
        public Texture2D toolTexture;

        [Header("BasePath of textures for runtime")]
        public string textureBasePath;

        [Header("Textures to use for showing the animation")]
        public Texture2D backgroundT;

        public Texture2D bodyT;
        public Texture2D outfitT;
        public Texture2D cloakT;
        public Texture2D faceitemsT;
        public Texture2D hairT;
        public Texture2D hatT;
        public Texture2D pritoolT;
        public Texture2D sectoolT;
        public Texture2D topT;

        [Header("GameObjects with Sprites for the farming actions")]
        public GameObject wateringAnimGo;

        [Header("Weapon Textures")]
        public Texture2D mainhandSwordAndShieldT;

        public Texture2D offhandSwordAndShieldT;
        public Texture2D spearT;
        public Texture2D mainhandBowT;
        public Texture2D offhandBowT;

        [Header("Location of the sprites used for the animation (inside 'Assets/Resouces')")]
        public string SpriteSetPath;

        public Animator animator;
        public Animator wateringAnimator;

        public CursorIndicator cursorIndicator;

        public UnityEngine.AI.NavMeshAgent agent;
        public Inventory inventory;

        public int Money { get; set; } = 0;
        public int Health { get; set; } = 100;
        public int Stamina { get; set; } = 100;

        private bool isInteracting = false;

        [Header("Interaction Settings")]
        [SerializeField]
        private readonly float interactionRange = 2.0f;
        [SerializeField]
        private readonly float interactionStoppingDistance = 1.8f;

        // Basic movement animations
        private readonly string[] basicAnimations = {
            "idle", "idle2"
        };

        // Action animations
        private readonly string[] actionAnimations = {
            "throwing", "hacking", "watering", "lifting", "fishing", "smithing",
            "climbing", "pushing", "pulling", "jumping"
        };

        // State animations
        private readonly string[] stateAnimations = {
            "sleeping", "schock", "sideglance", "sittingstool", "sittingstoolasleep",
            "sittingstooldrinking", "sittingground", "sittingground2", "toetapping"
        };

        // Combat animations
        private readonly string[] combatAnimations = {
            "combat_swordshield", "combat_spear", "combat_bow", "sheath",
            "slash1", "slash2", "thrust", "thrust2", "shieldbash",
            "retreat", "lunge", "parry", "dodge", "hurt", "dead",
            "shootup", "shootstraight"
        };

        public List<string> availableAnimations;

        [Header("Audio")]
        public AudioSource footstepAudioSource;

        [System.NonSerialized]
        private float footstepInterval = 0.2f;
        [System.NonSerialized]
        private float lastFootstepTime;

        private bool isSleeping = false;
        
        [Header("Sleep Settings")]
        public GameObject sleepingBag; // Assign in inspector
        [SerializeField] private float sleepTransitionTime = 1f;
        [SerializeField] private Vector2 sleepingBagOffset = new Vector2(0, -0.5f); // Adjust in inspector

        private float mouseHoldStartTime;
        private bool isHoldingMouse;
        private const float HOLD_THRESHOLD = 0.2f; // Time in seconds to trigger hold
        private AgenticController currentHoveredNPC;

        private void Awake()
        {
            animator = GetComponent<Animator>();
            if (animator == null)
            {
                Debug.LogError("No Animator component found on Player!");
                return;
            }
            
            // Verify animator has controller
            if (animator.runtimeAnimatorController == null)
            {
                Debug.LogError("Animator has no RuntimeAnimatorController assigned!");
                return;
            }
            
            // Create inventory through the controller
            GameController.Instance.InventoryController.CreateInventory("Player", 24);
            inventory = GameController.Instance.InventoryController.GetInventory("Player");
            
            // Create temporary GameObjects with ItemInstances for starting items
            CreateAndAddStartingItem("Wheat", 6);
            CreateAndAddStartingItem("Hoe", 1);
            CreateAndAddStartingItem("Watering Can", 1);

            // Combine all animation arrays into the main animations list
            availableAnimations = new List<string>();
            availableAnimations.AddRange(basicAnimations);
            availableAnimations.AddRange(actionAnimations);
            availableAnimations.AddRange(stateAnimations);
            availableAnimations.AddRange(combatAnimations);
        }

        private void CreateAndAddStartingItem(string itemName, int quantity)
        {
            var item = GameController.Instance.itemDatabase.GetItem(itemName);
            if (item != null)
            {
                // Create temporary GameObject
                var tempGO = new GameObject($"Temp_{itemName}");
                var itemInstance = tempGO.AddComponent<ItemInstance>();
                itemInstance.item = item;
                itemInstance.quantity = quantity;
                
                // Add to inventory using the same method as CollectableItem
                if (inventory.AddItem(itemInstance.item, itemInstance.quantity))
                {
                    Destroy(tempGO);
                }
                else
                {
                    Debug.LogError($"Failed to add starting item: {itemName}");
                    Destroy(tempGO);
                }
            }
            else
            {
                Debug.LogError($"Could not find item in database: {itemName}");
            }
        }

        // Start is called before the first frame update
        private void Start()
        {
            moveType = MoveType.idle;
            playerDirection = Direction.none;

            if (agent == null)
            {
                agent = GetComponent<UnityEngine.AI.NavMeshAgent>();
            }

            // Configure NavMeshAgent for 2D
            agent.updateRotation = false;
            agent.updateUpAxis = false;
            agent.radius = 0.2f; // Adjust this value based on your character size
            agent.obstacleAvoidanceType = UnityEngine.AI.ObstacleAvoidanceType.HighQualityObstacleAvoidance;
            agent.avoidancePriority = 50; // Middle priority (0-99 range, lower numbers = higher priority)

            // Add these lines to prevent pushing
            var rb = GetComponent<Rigidbody2D>();
            if (rb != null)
            {
                rb.collisionDetectionMode = CollisionDetectionMode2D.Continuous;
                rb.interpolation = RigidbodyInterpolation2D.Interpolate;
                rb.constraints = RigidbodyConstraints2D.FreezeRotation;
            }
        }

        // Update is called once per frame
        // HandleUpdate is called by GameController.Update() when the game state is FreeRoam
        public void HandleUpdate()
        {
            if (isSleeping) return; // Skip movement and animation handling if sleeping

            // Handle animation input   
            HandleAnimationInput();

            // Handle movement input
            PlayerMovementInput();

            // Check if the player is moving
            CheckMovement();
        }

        private void SetCharacterTextures()
        {
            if (backgroundT != null)
            {
                SetTexture(backgroundT, "background");
            }
            if (bodyT != null)
            {
                SetTexture(bodyT, "body");
            }
            if (outfitT != null)
            {
                SetTexture(outfitT, "outfit");
            }
            if (cloakT != null)
            {
                SetTexture(cloakT, "cloak");
            }
            if (faceitemsT != null)
            {
                SetTexture(faceitemsT, "faceitems");
            }
            if (hairT != null)
            {
                SetTexture(hairT, "hair");
            }
            if (hatT != null)
            {
                hair.SetActive(false);
                SetTexture(hatT, "hat");
            }
            if (pritoolT != null)
            {
                SetTexture(pritoolT, "pritool");
            }
            if (sectoolT != null)
            {
                SetTexture(sectoolT, "sectool");
            }
            if (topT != null)
            {
                SetTexture(topT, "top");
            }
            //Weapon Textures
            if (mainhandSwordAndShieldT != null)
            {
                SetTexture(mainhandSwordAndShieldT, "mainhand", true);
            }
            if (offhandSwordAndShieldT != null)
            {
                SetTexture(offhandSwordAndShieldT, "offhand", true);
            }
            if (spearT != null)
            {
                SetTexture(spearT, "mainhand", true);
            }
            if (mainhandBowT != null)
            {
                SetTexture(mainhandBowT, "mainhand", true);
            }
            if (offhandBowT != null)
            {
                SetTexture(offhandBowT, "offhand", true);
            }
        }

        private void SetTexture(Texture2D texture, string layer, bool combatAnimation = false)
        {
            //Base Location of all sprites
            string fileBasePath = textureBasePath;
            fileBasePath = fileBasePath.Replace("Assets/Resources/", "");

            //specific part of the path
            string filePath = "";
            string[] partedName = texture.name.Split('_');
            filePath += partedName[0] + "_" + partedName[1] + "_" + partedName[2] + "/";
            if (partedName[3] != "0bas") filePath += partedName[3] + "/";
            filePath += texture.name;

            Dictionary<string, string> textPaths = SetTextureFilePaths(filePath, partedName);

            //Base Textures
            if (!combatAnimation) SetBaseTextures(layer, fileBasePath, filePath, textPaths);

            //Combat Textures
            SetCombatTextures(layer, fileBasePath, filePath, textPaths);
        }

        private void SetCombatTextures(string layer, string fileBasePath, string filePath, Dictionary<string, string> textPaths)
        {
            if (layer == "mainhand") layer = "pritool";
            if (layer == "offhand") layer = "sectool";
            Texture2D pONE1Texture = SetTexture(fileBasePath, textPaths, "pONE1", true);
            Texture2D pONE2Texture = SetTexture(fileBasePath, textPaths, "pONE2", true);
            Texture2D pONE3Texture = SetTexture(fileBasePath, textPaths, "pONE3", true);
            Texture2D pPOL1Texture = SetTexture(fileBasePath, textPaths, "pPOL1", true);
            Texture2D pPOL2Texture = SetTexture(fileBasePath, textPaths, "pPOL2", true);
            Texture2D pPOL3Texture = SetTexture(fileBasePath, textPaths, "pPOL3", true);
            Texture2D pBOW1Texture = SetTexture(fileBasePath, textPaths, "pBOW1", true);
            Texture2D pBOW2Texture = SetTexture(fileBasePath, textPaths, "pBOW2", true);
            Texture2D pBOW3Texture = SetTexture(fileBasePath, textPaths, "pBOW3", true);

            if (pONE1Texture != null)
            {
                FillPlayerTexture(layer, pONE1Texture, "combat/pONE1");
                FillPlayerTexture(layer, pONE2Texture, "combat/pONE2");
                FillPlayerTexture(layer, pONE3Texture, "combat/pONE3");
            }
            if (pPOL1Texture != null)
            {
                FillPlayerTexture(layer, pPOL1Texture, "combat/pPOL1");
                FillPlayerTexture(layer, pPOL2Texture, "combat/pPOL2");
                FillPlayerTexture(layer, pPOL3Texture, "combat/pPOL3");
            }
            if (pBOW1Texture != null)
            {
                FillPlayerTexture(layer, pBOW1Texture, "combat/pBOW1");
                FillPlayerTexture(layer, pBOW2Texture, "combat/pBOW2");
                FillPlayerTexture(layer, pBOW3Texture, "combat/pBOW3");
            }
        }

        private void SetBaseTextures(string layer, string fileBasePath, string filePath, Dictionary<string, string> textPaths)
        {
            Texture2D p1Texture = SetTexture(fileBasePath, textPaths, "p1", false);
            Texture2D p1BTexture = SetTexture(fileBasePath, textPaths, "p1B", false);
            Texture2D p1CTexture = SetTexture(fileBasePath, textPaths, "p1C", false);
            Texture2D p2Texture = SetTexture(fileBasePath, textPaths, "p2", false);
            Texture2D p3Texture = SetTexture(fileBasePath, textPaths, "p3", false);
            if (layer == "pritool")
            {
                string fishing_test = filePath;
                if (fishing_test.Contains("6tla"))// && fishing_test.Contains("p3"))
                {
                    string[] p3pathParts = textPaths["p3"].Split('_');

                    string replacer = p3pathParts[6];
                    p3Texture = Resources.Load<Texture2D>(fileBasePath + textPaths["p3"].Replace(replacer, "roda").Replace(".png", ""));
                }
            }
            Texture2D p4Texture = SetTexture(fileBasePath, textPaths, "p4", false);

            FillPlayerTexture(layer, p1Texture, "p1");
            FillPlayerTexture(layer, p1BTexture, "p1B");
            FillPlayerTexture(layer, p1CTexture, "p1C");
            FillPlayerTexture(layer, p2Texture, "p2");
            FillPlayerTexture(layer, p3Texture, "p3");
            FillPlayerTexture(layer, p4Texture, "p4");
        }

        private void FillPlayerTexture(string layer, Texture2D pTexture, string key)
        {
            if (SpriteSetPath.EndsWith("/")) SpriteSetPath = SpriteSetPath.TrimEnd('/');
            Texture2D originp1 = Resources.Load<Texture2D>(SpriteSetPath + "/" + key + "/" + layer);
            if (pTexture != null && originp1 != null)
            {
                Color[] newPixelsp1 = pTexture.GetPixels();
                originp1.SetPixels(newPixelsp1);
                originp1.Apply();
            }
        }

        private static Texture2D SetTexture(string fileBasePath, Dictionary<string, string> textPaths, string textureKey, bool combatAnimation)
        {
            if (!fileBasePath.EndsWith("/")) fileBasePath += "/";
            Texture2D pTexture = null;
            if (textPaths[textureKey] != "")
            {
                pTexture = Resources.Load<Texture2D>(fileBasePath + textPaths[textureKey].Replace(".png", ""));
                if (combatAnimation)
                    if (pTexture == null)
                        pTexture = Resources.Load<Texture2D>(fileBasePath + "combat/" + textPaths[textureKey].Replace(".png", ""));
            }

            return pTexture;
        }

        private static Dictionary<string, string> SetTextureFilePaths(string filePath, string[] partedName)
        {
            Dictionary<string, string> textPaths = new Dictionary<string, string>()
            {
                { "p1", "" },
                { "p1B", "" },
                { "p1C", "" },
                { "p2", "" },
                { "p3", "" },
                { "p4", "" },
                { "pONE1", "" },
                { "pONE2", "" },
                { "pONE3", "" },
                { "pPOL1", "" },
                { "pPOL2", "" },
                { "pPOL3", "" },
                { "pBOW1", "" },
                { "pBOW2", "" },
                { "pBOW3", "" },
            };
            Dictionary<string, string> newPaths = new Dictionary<string, string>();
            foreach (KeyValuePair<string, string> tp in textPaths)
            {
                if (filePath.Contains("char_a_" + partedName[2]))
                    newPaths[tp.Key] = filePath.Replace("_" + partedName[2], "_" + tp.Key);
                else newPaths[tp.Key] = tp.Value;
            }

            return newPaths;
        }

        private void PlayerMovementInput()
        {
            // Handle keyboard input
            yInput = Input.GetAxisRaw("Vertical");
            xInput = Input.GetAxisRaw("Horizontal");

            // Prioritize horizontal movement over vertical for WASD
            if (Mathf.Abs(xInput) > 0.01f)
            {
                yInput = 0;
            }

            // Default to running, hold Shift to walk
            moveType = MoveType.running;
            movementSpeed = runningSpeed;

            if (Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift))
            {
                moveType = MoveType.walking;
                movementSpeed = walkingSpeed;
            }

            // If using keyboard movement, cancel any NavMesh path and use direct movement
            if (xInput != 0 || yInput != 0)
            {
                if (agent.hasPath)
                {
                    agent.ResetPath();
                }

                // Move using keyboard input
                Vector3 movement = new Vector3(xInput, yInput, 0);

                // Set the agent's velocity based on movement input
                Vector3 targetVelocity = movement * (moveType == MoveType.running ? runningSpeed : walkingSpeed);
                agent.velocity = targetVelocity;
            }
            else if (!agent.hasPath) // Only reset velocity if we're not following a path
            {
                agent.velocity = Vector3.zero;
            }

            // Handle mouse input
            if (Input.GetMouseButtonDown(0) && GameController.Instance.state == GameState.FreeRoam)
            {
                // Check if the pointer is over a UI element
                if (EventSystem.current.IsPointerOverGameObject())
                {
                    return;
                }

                // Start tracking hold time
                mouseHoldStartTime = Time.time;
                isHoldingMouse = true;

                // Get the world point where the mouse clicked
                Vector3 worldPoint = Camera.main.ScreenToWorldPoint(Input.mousePosition);
                worldPoint.z = 0;

                // Check for NPCs under the cursor
                Collider2D[] colliders = Physics2D.OverlapCircleAll(worldPoint, 0.5f);
                foreach (Collider2D collider in colliders)
                {
                    AgenticController npc = collider.GetComponent<AgenticController>();
                    if (npc != null)
                    {
                        currentHoveredNPC = npc;
                        break;
                    }
                }

                // If we're not over an NPC, handle normal click interactions immediately
                if (currentHoveredNPC == null)
                {
                    HandleClickInteraction(worldPoint);
                }
            }
            else if (Input.GetMouseButton(0) && isHoldingMouse && currentHoveredNPC != null)
            {
                // Check if we've held long enough to show character info
                if (Time.time - mouseHoldStartTime >= HOLD_THRESHOLD)
                {
                    currentHoveredNPC.ShowCharacterInfoPanel();
                    isHoldingMouse = false; // Prevent multiple triggers
                }
            }
            else if (Input.GetMouseButtonUp(0))
            {
                // If we were holding over an NPC but released before threshold, handle as normal click
                if (isHoldingMouse && currentHoveredNPC != null && 
                    Time.time - mouseHoldStartTime < HOLD_THRESHOLD)
                {
                    HandleClickInteraction(Camera.main.ScreenToWorldPoint(Input.mousePosition));
                }

                // Reset hold state
                isHoldingMouse = false;
                if (currentHoveredNPC != null)
                {
                    currentHoveredNPC.HideCharacterInfoPanel();
                    currentHoveredNPC = null;
                }
            }
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
                Vector2 normalizedVelocity = velocity.normalized;
                
                // Force movement to cardinal directions only
                if (Mathf.Abs(normalizedVelocity.x) > Mathf.Abs(normalizedVelocity.y))
                {
                    normalizedVelocity.y = 0;
                    normalizedVelocity.x = normalizedVelocity.x > 0 ? 1 : -1;
                    playerDirection = normalizedVelocity.x > 0 ? Direction.right : Direction.left;
                }
                else
                {
                    normalizedVelocity.x = 0;
                    normalizedVelocity.y = normalizedVelocity.y > 0 ? 1 : -1;
                    playerDirection = normalizedVelocity.y > 0 ? Direction.up : Direction.down;
                }

                // Set movement type based on speed
                moveType = speed > 2.5f ? MoveType.running : MoveType.walking;
                
                // Update animation parameters directly
                SetMovementAnimationParameters(normalizedVelocity.x, normalizedVelocity.y, moveType, playerDirection);

                // Add footstep sound handling
                float currentTime = Time.time;
                float interval = moveType == MoveType.running ? footstepInterval * 0.5f : footstepInterval;
                
                if (currentTime - lastFootstepTime >= interval)
                {
                    PlayFootstep();
                    lastFootstepTime = currentTime;
                }
            }
            else
            {
                moveType = MoveType.idle;
                SetMovementAnimationParameters(0, 0, MoveType.idle, playerDirection);
            }
        }

        private void PlayFootstep()
        {
            if (footstepAudioSource != null && !footstepAudioSource.isPlaying)
            {
                footstepAudioSource.Play();
            }
        }

        // Add this coroutine to handle delayed interaction
        private IEnumerator InteractWhenInRange(Interactable interactable)
        {
            var interactableObject = (interactable as MonoBehaviour);
            if (interactableObject == null) yield break;

            string objectName = interactableObject.gameObject.name;
            Transform interactableTransform = interactableObject.transform;
            
            // Stop any current movement input
            ResetMovement();
            
            // Set the stopping distance before pathfinding
            agent.stoppingDistance = interactionStoppingDistance;
            
            // Wait one frame to let the NavMeshAgent start pathfinding
            yield return null;
            
            // First wait until we have a valid path
            while (agent.pathStatus == UnityEngine.AI.NavMeshPathStatus.PathInvalid ||
                   !agent.hasPath ||
                   agent.pathStatus == UnityEngine.AI.NavMeshPathStatus.PathPartial)
            {
                yield return null;
            }

            // Then wait until we reach the destination
            while (agent.hasPath && 
                   agent.remainingDistance > agent.stoppingDistance)
            {
                // Check if path becomes invalid while walking
                if (agent.pathStatus == UnityEngine.AI.NavMeshPathStatus.PathInvalid)
                {
                    Debug.Log("Path became invalid while walking to " + objectName);
                    agent.ResetPath();
                    yield break;
                }
                yield return null;
            }

            // Stop movement before interaction
            agent.ResetPath();
            agent.velocity = Vector3.zero;
            
            // Final position check to ensure we actually reached the target
            float finalDistance = Vector3.Distance(transform.position, interactableTransform.position);
            if (finalDistance <= interactionRange)
            {
                yield return new WaitForSeconds(0.1f);
                
                // Ensure movement is completely stopped during interaction
                ResetMovement();
                moveType = MoveType.idle;
                
                isInteracting = true;

                // Face the interactable before interacting
                FaceTarget(interactableTransform.position);

                // Interact with the object
                interactable.Interact();
            }
            else
            {
                Debug.Log("Failed to reach " + objectName);
            }
            
            agent.stoppingDistance = 0.1f; // Reset stopping distance
        }

        public void SetDialogState(bool state)
        {
            isInteracting = state;
        }   

        private void FaceTarget(Vector3 targetPosition)
        {
            Debug.Log("FaceTarget: " + targetPosition);
            Vector2 direction = (targetPosition - transform.position).normalized;
            Debug.Log("Direction: " + direction);
            
            // Determine which direction to face based on the dominant axis
            int faceDirection;
            if (Mathf.Abs(direction.x) > Mathf.Abs(direction.y))
            {
                faceDirection = direction.x > 0 ? 0 : 1; // Right = 0, Left = 1
            }
            else
            {
                faceDirection = direction.y > 0 ? 2 : 3; // Up = 2, Down = 3
            }

            Debug.Log("FaceTarget: " + faceDirection);

            // Set animation parameters directly
            animator.SetFloat("xInput", 0);
            animator.SetFloat("yInput", 0);
            // This SET IDLE MUST REMAIN HERE FOR IT TO WORK 
            animator.SetInteger("direction", faceDirection);
            animator.SetBool("isWalking", false);
            animator.SetBool("isRunning", false);
            animator.SetTrigger("idle");
        }

        private IEnumerator InteractWithTileWhenInRange(Vector3Int cellPosition)
        {
            var interactableObject = GameController.Instance.EnvironmentManager;
            if (interactableObject == null) yield break;
            
            // Stop any current movement input
            ResetMovement();
            
            // Set the stopping distance before pathfinding
            agent.stoppingDistance = interactionStoppingDistance;
            
            // Wait one frame to let the NavMeshAgent start pathfinding
            yield return null;
            
            // First wait until we have a valid path
            while (agent.pathStatus == UnityEngine.AI.NavMeshPathStatus.PathInvalid ||
                   !agent.hasPath ||
                   agent.pathStatus == UnityEngine.AI.NavMeshPathStatus.PathPartial)
            {
                yield return null;
            }

            // Then wait until we reach the destination
            while (agent.hasPath && 
                   agent.remainingDistance > agent.stoppingDistance)
            {
                // Check if path becomes invalid while walking
                if (agent.pathStatus == UnityEngine.AI.NavMeshPathStatus.PathInvalid)
                {
                    Debug.Log("Path became invalid while walking to tile");
                    agent.ResetPath();
                    yield break;
                }
                yield return null;
            }

            // Stop movement before interaction
            agent.ResetPath();
            agent.velocity = Vector3.zero;
            
            // Final position check to ensure we actually reached the target
            Vector3 tileWorldPos = GameController.Instance.EnvironmentManager.interactableMap.GetCellCenterWorld(cellPosition);
            float finalDistance = Vector2.Distance(transform.position, tileWorldPos);

            yield return new WaitForSeconds(0.1f);
            GameController.Instance.EnvironmentManager.TileInteract(cellPosition);
            
            // Face the tile before interacting
            FaceTarget(tileWorldPos);
            
            // Ensure movement is completely stopped during interaction
            ResetMovement();
            moveType = MoveType.idle;
            
            agent.stoppingDistance = 0.1f; // Reset stopping distance
        }

        public void TriggerAnimation(string animationName)
        {
            // Stop movement while playing the animation
            moveType = MoveType.idle;
            ResetMovement();
            
            // Call the original animation trigger
            if (animator == null)
            {
                Debug.LogError("No Animator component found!");
                return;
            }

            // Check if the animation is in our available animations list
            if (!availableAnimations.Contains(animationName))
            {
                Debug.LogWarning($"Animation '{animationName}' not found in available animations!");
                return;
            }

            // Set movement parameters to idle first
            SetMovementAnimationParameters(0, 0, MoveType.idle, playerDirection);

            // Reset any existing triggers to avoid animation conflicts
            foreach (var param in animator.parameters)
            {
                if (param.type == AnimatorControllerParameterType.Trigger)
                {
                    animator.ResetTrigger(param.name);
                }
            }

            // Set the trigger for the requested animation
            animator.SetTrigger(animationName);
        }

        public void HandleAnimationInput()
        {
            // If we're moving, don't allow other animations to interrupt
            if (moveType != MoveType.idle)
                return;

            // Stop any existing movement and path when triggering an animation
            if (Input.GetKeyDown(KeyCode.Y) || Input.GetKeyDown(KeyCode.I) || 
                Input.GetKeyDown(KeyCode.O) || Input.GetKeyDown(KeyCode.P) ||
                Input.GetKeyDown(KeyCode.H) || Input.GetKeyDown(KeyCode.J) ||
                Input.GetKeyDown(KeyCode.K) || Input.GetKeyDown(KeyCode.L) ||
                Input.GetKeyDown(KeyCode.B) || Input.GetKeyDown(KeyCode.N) ||
                Input.GetKeyDown(KeyCode.M))
            {
                // Reset any existing path and movement
                if (agent.hasPath)
                    agent.ResetPath();
                agent.velocity = Vector3.zero;
                ResetMovement();
            }

            if (Input.GetKeyDown(KeyCode.Y)) {
                TriggerAnimation("idle");
            }
            else if (Input.GetKeyDown(KeyCode.I)) {
                TriggerAnimation("throwing");
            }
            else if (Input.GetKeyDown(KeyCode.O)) {
                TriggerAnimation("hacking");
            }
            else if (Input.GetKeyDown(KeyCode.P)) {
                TriggerAnimation("watering");
            }
            else if (Input.GetKeyDown(KeyCode.H)) {
                TriggerAnimation("lifting");
            }
            else if (Input.GetKeyDown(KeyCode.J)) {
                TriggerAnimation("fishing");
            }
            else if (Input.GetKeyDown(KeyCode.K)) {
                TriggerAnimation("smithing");
            }
            else if (Input.GetKeyDown(KeyCode.L)) {
                TriggerAnimation("climbing");
            }
            else if (Input.GetKeyDown(KeyCode.B)) {
                TriggerAnimation("pushing");
            }
            else if (Input.GetKeyDown(KeyCode.N)) {
                TriggerAnimation("pulling");
            }
            else if (Input.GetKeyDown(KeyCode.M)) {
                TriggerAnimation("jumping");
            }
        }

        private void SetMovementAnimationParameters(float inputX, float inputY,
            MoveType moveType, Direction direction)
        {
            animator.SetFloat("xInput", inputX);
            animator.SetFloat("yInput", inputY);
            animator.SetInteger("direction", (int)direction);

            switch (moveType)
            {
                case MoveType.walking:
                    animator.SetBool("isWalking", true);
                    animator.SetBool("isRunning", false);
                    break;

                case MoveType.running:
                    animator.SetBool("isWalking", false);
                    animator.SetBool("isRunning", true);
                    break;

                case MoveType.idle:
                    animator.SetBool("isWalking", false);
                    animator.SetBool("isRunning", false);
                    break;
            }
        }

        public void SetInteractState(bool state)
        {
            isInteracting = state;
        }

        public void PlayWateringAnimation()
        {
            if (wateringAnimGo != null && wateringAnimator != null)
            {
                // Activate the watering game object
                wateringAnimGo.SetActive(true);

                // Get current direction from animator
                int currentDirection = animator.GetInteger("direction");
                
                // Set the direction on watering animator
                wateringAnimator.SetInteger("direction", currentDirection);
                wateringAnimator.SetTrigger("watering");
                
                // Start coroutine to wait for animation to complete
                StartCoroutine(DeactivateAfterWateringAnimation());
            }
        }

        private IEnumerator DeactivateAfterWateringAnimation()
        {
            // Wait for the current animation state to finish
            yield return new WaitForSeconds(1.0f);
            
            // Deactivate the watering game object
            wateringAnimGo.SetActive(false);
        }

        public void StartSleeping(Vector3 position)
        {
            // Stop any current movement and interactions
            if (agent.hasPath)
                agent.ResetPath();
            agent.velocity = Vector3.zero;
            ResetMovement();
            
            sleepingBag.SetActive(true);

            // Face down when sleeping
            playerDirection = Direction.down;
            SetMovementAnimationParameters(0, 0, MoveType.idle, Direction.down);
            
            isSleeping = true;
            StartCoroutine(SleepTransition());
        }

        public void WakeUp()
        {
            // Reset sleeping state
            isSleeping = false;
            
            // Hide sleeping bag
            if (sleepingBag != null)
            {
                sleepingBag.SetActive(false);
            }
        }

        private IEnumerator SleepTransition()
        {
            // Optional: Add fade effect or other transition here
            yield return new WaitForSeconds(sleepTransitionTime);
            
            // Notify GameController that sleep transition is complete
            GameController.Instance.OnPlayerSleepComplete();
        }

        private void HandleClickInteraction(Vector3 worldPoint)
        {
            worldPoint.z = 0;
            bool interactionHandled = false;

            // Check for Interactables first
            Collider2D[] colliders = Physics2D.OverlapCircleAll(worldPoint, 0.5f);
            foreach (Collider2D collider in colliders)
            {
                Interactable interactable = collider.GetComponent<Interactable>();
                if (interactable != null)
                {
                    // Check if within interaction range
                    float distance = Vector2.Distance(transform.position, collider.transform.position);
                    if (distance <= interactionRange)
                    {
                        interactable.Interact();
                        isInteracting = true;
                        interactionHandled = true;
                        var interactableObject = (interactable as MonoBehaviour);
                        if (interactableObject != null)
                        {
                            FaceTarget(interactableObject.transform.position);
                        }
                        return;
                    }
                    else
                    {
                        // If too far, move to the interactable first
                        if (agent != null)
                        {
                            agent.stoppingDistance = interactionStoppingDistance;
                            agent.SetDestination(collider.transform.position);
                            StartCoroutine(InteractWhenInRange(interactable));
                        }
                        interactionHandled = true;
                        return;
                    }
                }
            }

            // Handle remaining click logic (tilemap interaction and movement)
            if (!interactionHandled)
            {
                Vector3Int cellPosition = GameController.Instance.EnvironmentManager.WorldToCell(worldPoint);
                if (GameController.Instance.EnvironmentManager.IsInteractable(cellPosition))
                {
                    float distance = Vector2.Distance(transform.position, worldPoint);
                    if (distance <= interactionRange)
                    {
                        GameController.Instance.EnvironmentManager.TileInteract(cellPosition);
                        interactionHandled = true;
                    }
                    else
                    {
                        if (agent != null)
                        {
                            agent.stoppingDistance = interactionStoppingDistance;
                            agent.SetDestination(worldPoint);
                            StartCoroutine(InteractWithTileWhenInRange(cellPosition));
                        }
                        interactionHandled = true;
                    }
                }
            }

            if (cursorIndicator != null)
            {
                cursorIndicator.ShowAtPosition(worldPoint);
            }

            if (!interactionHandled)
            {
                if (agent != null)
                {
                    agent.stoppingDistance = 0.1f;
                    UnityEngine.AI.NavMeshHit navMeshHit;
                    if (UnityEngine.AI.NavMesh.SamplePosition(worldPoint, out navMeshHit, 1.0f, UnityEngine.AI.NavMesh.AllAreas))
                    {
                        agent.SetDestination(navMeshHit.position);
                    }
                }
            }
        }
    }
}