using System;
using System.Collections.Generic;
using UnityEngine;
using Agentics;

namespace Agentics
{
	[Serializable]
	public class Inventory
	{
		public string Name { get; private set; }
		public int MaxItems { get; private set; }
		
		[SerializeField]
		public List<InventoryItem> items;

    	public Item activeItem;
		
		public event Action OnInventoryChanged;
		public event Action OnActiveItemChanged;
		
		public IReadOnlyList<InventoryItem> Items => items;

		public Inventory(string name, int maxItems)
		{
			Name = name;
			MaxItems = maxItems;
			items = new List<InventoryItem>();
		}

		public bool AddItem(Item item, int quantity = 1)
		{
			// First try to stack with existing items
			var existingItem = items.Find(i => i.Item == item && i.CanAddQuantity(quantity));
			if (existingItem != null)
			{
				existingItem.AddQuantity(quantity);
				OnInventoryChanged?.Invoke();
				return true;
			}

			// Then try to add as new item if we haven't reached max items
			if (items.Count < MaxItems)
			{
				items.Add(new InventoryItem(item, quantity));
				
				// If this is the first item being added, set it as the active item
				if (items.Count == 1 && activeItem == null)
				{
					activeItem = item;
					OnActiveItemChanged?.Invoke();
				}
				
				OnInventoryChanged?.Invoke();
				return true;
			}

			return false;
		}

		public bool RemoveItem(Item item, int quantity = 1)
		{
			var inventoryItem = items.Find(i => i.Item == item);
			if (inventoryItem != null)
			{
				if (inventoryItem.Quantity <= quantity)
				{
					items.Remove(inventoryItem);
				}
				else
				{
					inventoryItem.RemoveQuantity(quantity);
				}
				OnInventoryChanged?.Invoke();
				return true;
			}
			return false;
		}

		public int GetItemCount(Item item)
		{
			var inventoryItem = items.Find(i => i.Item == item);
			return inventoryItem?.Quantity ?? 0;
		}

        public Item GetActiveItem()
        {
            // If no active item is set but inventory has items, use the first one
            if (activeItem == null && items.Count > 0)
            {
                activeItem = items[0].Item;
            }
            return activeItem;
        }

        public void SetActiveItem(Item item)
        {
            activeItem = item;
            OnActiveItemChanged?.Invoke();
        }
	}
}