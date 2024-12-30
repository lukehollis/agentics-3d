using System;
using System.Collections.Generic;
using UnityEngine;

namespace Agentics
{
    public delegate void MovementDelegate(float inputX, float inputY, MoveType moveType, Direction direction, Player character, string testTrigger);

    public static class EventHandler
    {
        //Movement Event
        public static event MovementDelegate MovementEvent;

        //Movement Event Call for Publishers
        public static void CallMovementEvent(float inputX, float inputY,
            MoveType moveType, Direction direction, Player character, string testTrigger)
        {
            if (MovementEvent != null)
            {
                MovementEvent(inputX, inputY, moveType, direction, character, testTrigger);
            }
        }
    }
}