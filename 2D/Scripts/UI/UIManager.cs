using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using Agentics;


public class UIManager : MonoBehaviour
{
	[Header("Inventory UIs")]
	public InventoryUI playerInventoryUI;
	public InventoryUI toolbarInventoryUI;

	[Header("UI components")]
	public ToolbarUI toolbarUI;
	public InventoryPanel inventoryPanel;

	[Header("Dragging")]
	public static SlotUI draggedSlot;
	public static UnityEngine.UI.Image draggedIcon;
	public static bool dragStack;


	void Start()
	{
		if (inventoryPanel != null)
			inventoryPanel.gameObject.SetActive(false); // starts out not open

		// Both UIs should reference the same inventory
		if (playerInventoryUI != null && toolbarInventoryUI != null)
		{
			playerInventoryUI.inventoryName = "Player";
			toolbarInventoryUI.inventoryName = "Player";
		}
	}

	void Update()
	{
        if (Input.GetKeyDown(KeyCode.Tab) || Input.GetKeyDown(KeyCode.B))
        	// ToggleInventoryUI();
        if (Input.GetKey(KeyCode.LeftShift))
        	dragStack = true;
        else
        	dragStack = false;
	}

	public void ToggleInventoryUI()
    {
    	if (inventoryPanel != null)
    	{
	    	inventoryPanel.gameObject.SetActive(!inventoryPanel.gameObject.activeSelf);
			Item activeItem = GameController.Instance.player.inventory.GetActiveItem();
			if (activeItem != null) {
				SetItemDetails(activeItem);
			}
    	}
    }

	public void SetItemDetails(Item item)	
	{
		Debug.Log("Setting item details for: " + item.DisplayName);
		inventoryPanel.SetItemDetails(
			item.DetailImage, 
			item.DisplayName,
			item.ScientificName,
			item.Description
		);
	}
}