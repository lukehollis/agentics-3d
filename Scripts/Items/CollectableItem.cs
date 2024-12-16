using UnityEngine;
using Agentics;

[RequireComponent(typeof(ItemInstance))]
public class CollectableItem : MonoBehaviour, Interactable
{
	private ItemInstance itemInstance;
	
	private void Awake()
	{
		itemInstance = GetComponent<ItemInstance>();
	}
	
	public void Interact()
	{
		var playerInventory = SimulationController.Instance.player.inventory;
		
		Debug.Log("Interacting with item: " + itemInstance.item.name + " with quantity: " + itemInstance.quantity);
		
		if (playerInventory.AddItem(itemInstance.item, itemInstance.quantity))
		{
			Destroy(gameObject);
		}
	}
}