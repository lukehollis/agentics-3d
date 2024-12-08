using UnityEngine;

namespace Agentics
{
    [System.Serializable]
    public class InventorySlot
    {
        public Item Item { get; private set; }
        public int Quantity { get; private set; }
        
        public bool IsEmpty => Item == null || Quantity <= 0;
        
        public InventorySlot(Item item = null, int quantity = 0)
        {
            Item = item;
            Quantity = quantity;
        }

        public bool CanAddItems(int amount = 1)
        {
            if (Item == null) return true;
            return Quantity + amount <= Item.MaxStackSize;
        }

        public bool AddItems(Item item, int amount = 1)
        {
            if (Item != null && Item != item) return false;
            if (!CanAddItems(amount)) return false;
            
            Item = item;
            Quantity += amount;
            return true;
        }

        public bool RemoveItems(int amount = 1)
        {
            if (IsEmpty || amount > Quantity) return false;
            
            Quantity -= amount;
            if (Quantity <= 0)
            {
                Item = null;
                Quantity = 0;
            }
            return true;
        }
    }
}