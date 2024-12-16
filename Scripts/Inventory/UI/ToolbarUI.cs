using UnityEngine;
using System.Collections.Generic;
using UnityEngine.UI;
using Agentics;

public class ToolbarUI : MonoBehaviour
{
    public const int TOOLBAR_SLOTS = 3;
    public string inventoryName;
    public List<SlotUI> slots = new List<SlotUI>();
    
    [SerializeField] private Canvas canvas;
    [SerializeField] private InventoryController inventoryController;
    
    private Inventory inventory;
    private static Image draggedIcon;
    private static SlotUI draggedSlot;

    private void Start()
    {
        inventory = inventoryController.GetInventory(inventoryName);
        if (inventory != null)
        {
            inventory.OnInventoryChanged += Refresh;
            SetupSlots();
            Refresh();
        }
    }

    private void OnDestroy()
    {
        if (inventory != null)
        {
            inventory.OnInventoryChanged -= Refresh;
        }
    }

    public void Refresh()
    {
        // Clear all slots
        foreach (var slot in slots)
        {
            slot.SetEmpty();
        }

        // Only show first 3 items
        var items = inventory.Items;
        for (int i = 0; i < items.Count && i < TOOLBAR_SLOTS; i++)
        {
            slots[i].SetItem(items[i]);
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

    // Drag and drop functionality can be added if needed for the toolbar
}

