using System;
using System.Collections.Generic;
using UnityEngine;

namespace Agentics
{
    public delegate void MovementDelegate(float inputX, float inputY, MoveType moveType, Direction direction, Player2D character, string testTrigger);

    public static class EventHandler2D
    {
        //Movement Event
        public static event MovementDelegate MovementEvent;

        //Movement Event Call for Publishers
        public static void CallMovementEvent(float inputX, float inputY,
            MoveType moveType, Direction direction, Player2D character, string testTrigger)
        {
            if (MovementEvent != null)
            {
                MovementEvent(inputX, inputY, moveType, direction, character, testTrigger);
            }
        }
    }

    public static class MovementEventHandler
    {
        // Generic movement delegate that works for both 2D and 3D
        public delegate void MovementEventDelegate(
            float inputX, 
            float inputY, 
            MoveType moveType, 
            Direction direction, 
            GameObject source,
            string triggerAnimation = null
        );

        public static event MovementEventDelegate OnMovement;

        public static void CallMovementEvent(
            float inputX, 
            float inputY, 
            MoveType moveType, 
            Direction direction, 
            GameObject source,
            string triggerAnimation = null)
        {
            OnMovement?.Invoke(inputX, inputY, moveType, direction, source, triggerAnimation);
        }
    }
}