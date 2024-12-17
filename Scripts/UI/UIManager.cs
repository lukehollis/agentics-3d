using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.InputSystem;
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

	private Keyboard keyboard;

	void Awake()
	{
		keyboard = Keyboard.current;
	}

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
		if (keyboard.tabKey.wasPressedThisFrame || keyboard.bKey.wasPressedThisFrame)
			ToggleInventoryUI();
			
		dragStack = keyboard.leftShiftKey.isPressed;
	}

	public void ToggleInventoryUI()
	{
		if (inventoryPanel != null)
		{
			inventoryPanel.gameObject.SetActive(!inventoryPanel.gameObject.activeSelf);
			Item activeItem = SimulationController.Instance.player.inventory.GetActiveItem();
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