using UnityEngine;

namespace Agentics
{
    [System.Serializable]
    public class InventoryItem
    {
        public Item Item { get; private set; }
        public int Quantity { get; private set; }

        public InventoryItem(Item item, int quantity)
        {
            Item = item;
            Quantity = quantity;
        }

        public bool CanAddQuantity(int amount)
        {
            return Quantity + amount <= Item.MaxStackSize;
        }

        public void AddQuantity(int amount)
        {
            Quantity = Mathf.Min(Quantity + amount, Item.MaxStackSize);
        }

        public void RemoveQuantity(int amount)
        {
            Quantity = Mathf.Max(0, Quantity - amount);
        }
    }
}