using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using System.Collections.Generic;
using Agentics;

public class InventoryUI : MonoBehaviour
{
	public string inventoryName;
	[SerializeField] private List<SlotUI> slots = new List<SlotUI>();
	public bool isToolbar;
	
	[SerializeField] private Canvas canvas;
	[SerializeField] private InventoryController inventoryController;
	
	private Inventory inventory;
	private static Image draggedIcon;
	private static SlotUI draggedSlot;
	private static bool dragStack;

	private void Awake()
	{
		canvas = FindObjectOfType<Canvas>();
		inventoryController = FindObjectOfType<InventoryController>();
		
		if (slots == null || slots.Count == 0)
		{
			slots = new List<SlotUI>(GetComponentsInChildren<SlotUI>());
		}

	}

	private void Start()
	{
		inventory = GameController.Instance.player.inventory;
		if (inventory != null)
		{
			inventory.OnInventoryChanged += Refresh;
			inventory.OnActiveItemChanged += Refresh;
			SetupSlots();
			Refresh();
		}
		else
		{
			Debug.LogError($"Could not find inventory '{inventoryName}' through GameController");
		}
	}

	private void OnDestroy()
	{
		if (inventory != null)
		{
			inventory.OnInventoryChanged -= Refresh;
			inventory.OnActiveItemChanged -= Refresh;
		}
	}

	public void Refresh()
	{
		// Clear all slots first
		foreach (var slot in slots)
		{
			slot.SetEmpty();
			slot.SetHighlight(false); // Reset highlight state
		}

		// Fill slots with items
		var items = inventory.Items;
		var activeItem = inventory.GetActiveItem();
		
		for (int i = 0; i < items.Count && i < slots.Count; i++)
		{
			if (isToolbar && i >= 3) break; // Only show first 3 items in toolbar
			
			slots[i].SetItem(items[i]);
			
			// Set highlight if this is the active item
			if (activeItem != null && items[i].Item == activeItem)
			{
				slots[i].SetHighlight(true);
			}
		}
	}

	public void Remove(int slotIndex)
	{
		if (slotIndex >= 0 && slotIndex < slots.Count)
		{
			var slot = slots[slotIndex];
			if (slot.currentItem != null)
			{
				inventory.RemoveItem(slot.currentItem.Item, slot.currentItem.Quantity);
			}
		}
	}

	public void SlotBeginDrag(SlotUI slot)
	{
		if (slot.currentItem == null) return;
		
		draggedSlot = slot;
		draggedIcon = Instantiate(slot.itemIcon, canvas.transform);
		draggedIcon.raycastTarget = false;
		draggedIcon.rectTransform.sizeDelta = new Vector2(50, 50);
		
		MoveToMousePosition(draggedIcon.gameObject);
	}

	public void SlotDrag()
	{
		if (draggedIcon != null)
		{
			MoveToMousePosition(draggedIcon.gameObject);
		}
	}

	public void SlotEndDrag()
	{
		if (draggedIcon != null)
		{
			Destroy(draggedIcon.gameObject);
			draggedIcon = null;
		}
	}

	public void SlotDrop(SlotUI targetSlot)
	{
		if (draggedSlot == null || targetSlot == null) return;

		var sourceInventory = inventoryController.GetInventory(draggedSlot.inventory.Name);
		var targetInventory = inventoryController.GetInventory(targetSlot.inventory.Name);

		if (sourceInventory == null || targetInventory == null) return;
		if (draggedSlot.currentItem == null) return;

		int quantity = dragStack ? draggedSlot.currentItem.Quantity : 1;
		
		inventoryController.TransferItem(
			draggedSlot.inventory.Name,
			targetSlot.inventory.Name,
			draggedSlot.currentItem.Item,
			quantity
		);
	}

	private void MoveToMousePosition(GameObject toMove)
	{
		if (canvas != null)
		{
			Vector2 position;
			RectTransformUtility.ScreenPointToLocalPointInRectangle(
				canvas.transform as RectTransform,
				Input.mousePosition,
				null,
				out position
			);
			toMove.transform.position = canvas.transform.TransformPoint(position);
		}
	}

	private void SetupSlots()
	{
		for (int i = 0; i < slots.Count; i++)
		{
			slots[i].slotID = i;
			slots[i].inventory = inventory;
		}
	}
}
