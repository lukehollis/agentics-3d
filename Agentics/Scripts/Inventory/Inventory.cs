using System;
using System.Collections.Generic;
using UnityEngine;

namespace Agentics
{
	[Serializable]
	public class Inventory
	{
		public string Name { get; private set; }
		public int MaxSlots { get; private set; }
		
		[SerializeField]
		private List<InventorySlot> slots;
		
		public event Action OnInventoryChanged;

		public Inventory(string name, int maxSlots)
		{
			Name = name;
			MaxSlots = maxSlots;
			slots = new List<InventorySlot>();
			
			for (int i = 0; i < maxSlots; i++)
			{
				slots.Add(new InventorySlot());
			}
		}

		public bool AddItem(Item item, int quantity = 1)
		{
			// First try to stack with existing items
			for (int i = 0; i < slots.Count; i++)
			{
				if (slots[i].Item == item && slots[i].CanAddItems(quantity))
				{
					slots[i].AddItems(item, quantity);
					OnInventoryChanged?.Invoke();
					return true;
				}
			}

			// Then try to find empty slot
			for (int i = 0; i < slots.Count; i++)
			{
				if (slots[i].IsEmpty)
				{
					slots[i].AddItems(item, quantity);
					OnInventoryChanged?.Invoke();
					return true;
				}
			}

			return false;
		}

		public bool RemoveItem(Item item, int quantity = 1)
		{
			for (int i = 0; i < slots.Count; i++)
			{
				if (slots[i].Item == item && !slots[i].IsEmpty)
				{
					if (slots[i].RemoveItems(quantity))
					{
						OnInventoryChanged?.Invoke();
						return true;
					}
				}
			}
			return false;
		}

		public InventorySlot GetSlot(int index)
		{
			if (index < 0 || index >= slots.Count) return null;
			return slots[index];
		}

		public int GetItemCount(Item item)
		{
			int count = 0;
			foreach (var slot in slots)
			{
				if (slot.Item == item)
					count += slot.Quantity;
			}
			return count;
		}

		public bool DropItem(Item item, int quantity = 1)
		{
			for (int i = 0; i < slots.Count; i++)
			{
				if (slots[i].Item == item && !slots[i].IsEmpty)
				{
					if (slots[i].RemoveItems(quantity))
					{
						OnInventoryChanged?.Invoke();
						return true;
					}
				}
			}
			return false;
		}
	}
}