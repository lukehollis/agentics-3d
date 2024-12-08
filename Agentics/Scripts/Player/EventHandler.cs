using System;
using System.Collections.Generic;
using UnityEngine;

namespace Agentics
{
    // Generic delegate for character actions
    public delegate void CharacterActionDelegate<T>(T args) where T : CharacterActionArgs;

    // Base class for action arguments
    public abstract class CharacterActionArgs
    {
        public Player Character { get; set; }
        public string TriggerID { get; set; }
    }

    // Movement specific arguments
    public class MovementActionArgs : CharacterActionArgs
    {
        public float InputX { get; set; }
        public float InputY { get; set; }
        public MoveType MoveType { get; set; }
        public Direction Direction { get; set; }
    }

    public static class EventHandler
    {
        // Dictionary to store different event types
        private static Dictionary<Type, Delegate> eventDictionary = new Dictionary<Type, Delegate>();

        // Subscribe to events
        public static void Subscribe<T>(CharacterActionDelegate<T> handler) where T : CharacterActionArgs
        {
            Type type = typeof(T);
            
            if (!eventDictionary.ContainsKey(type))
            {
                eventDictionary[type] = null;
            }
            
            eventDictionary[type] = Delegate.Combine(eventDictionary[type], handler);
        }

        // Unsubscribe from events
        public static void Unsubscribe<T>(CharacterActionDelegate<T> handler) where T : CharacterActionArgs
        {
            Type type = typeof(T);
            
            if (eventDictionary.ContainsKey(type))
            {
                eventDictionary[type] = Delegate.Remove(eventDictionary[type], handler);
            }
        }

        // Trigger events
        public static void Trigger<T>(T args) where T : CharacterActionArgs
        {
            Type type = typeof(T);
            
            if (eventDictionary.ContainsKey(type))
            {
                var handler = eventDictionary[type] as CharacterActionDelegate<T>;
                handler?.Invoke(args);
            }
        }

        // Helper method for movement events (for backward compatibility)
        public static void CallMovementEvent(float inputX, float inputY, 
            MoveType moveType, Direction direction, Player character, string triggerID)
        {
            var args = new MovementActionArgs
            {
                InputX = inputX,
                InputY = inputY,
                MoveType = moveType,
                Direction = direction,
                Character = character,
                TriggerID = triggerID
            };

            Trigger(args);
        }
    }
}