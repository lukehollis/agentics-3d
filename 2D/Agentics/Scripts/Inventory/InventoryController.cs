
using System.Collections.Generic;
using UnityEngine;  

namespace Agentics
{
    public class InventoryController : MonoBehaviour
    {
        private Dictionary<string, Inventory> inventories = new Dictionary<string, Inventory>();
        
        public void CreateInventory(string name, int slots)
        {
            if (!inventories.ContainsKey(name))
                inventories.Add(name, new Inventory(name, slots));
        }
        
        public Inventory GetInventory(string name)
        {
            return inventories.TryGetValue(name, out var inventory) ? inventory : null;
        }

        public bool TransferItem(string fromInventory, string toInventory, Item item, int quantity = 1)
        {
            var source = GetInventory(fromInventory);
            var destination = GetInventory(toInventory);
            
            if (source == null || destination == null)
                return false;
                
            if (source.RemoveItem(item, quantity))
            {
                if (destination.AddItem(item, quantity))
                    return true;
                    
                // If failed to add to destination, return item to source
                source.AddItem(item, quantity);
            }
            
            return false;
        }
    }
}
