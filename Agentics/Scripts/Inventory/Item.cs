using UnityEngine;

namespace Agentics
{
    public abstract class Item : ScriptableObject, IDatabaseEntry
    {
        public string Key => UniqueID;
        
        [Tooltip("Unique identifier used in the database and save system")]
        public string UniqueID = "DefaultID";
        
        public string DisplayName;
        public Sprite ItemSprite;
        public int MaxStackSize = 10;
        public bool Consumable = true;
        public int BuyPrice = -1;
        public int SellPrice = -1;

        [TextArea]
        public string Description;
        public string ScientificName; // For plants/crops
        public Sprite DetailImage; // For detailed view in UI
        
        [Tooltip("Prefab that will be instantiated in the player hand when this is equipped")]
        public GameObject VisualPrefab;
        public string PlayerAnimatorTriggerUse = "GenericToolSwing";
        
        [Tooltip("Sound triggered when using the item")]
        public AudioClip[] UseSound;

        public abstract bool CanUse(Vector3Int target);
        public abstract bool Use(Vector3Int target);
        
        public virtual bool NeedTarget()
        {
            return true;
        }
    }
}
