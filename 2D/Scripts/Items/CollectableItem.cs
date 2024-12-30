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
		var playerInventory = GameController.Instance.player.inventory;
		
		if (playerInventory.AddItem(itemInstance.item, itemInstance.quantity))
		{
			Destroy(gameObject);
		}
	}
}