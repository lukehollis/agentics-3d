using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Agentics;
using UnityEngine.InputSystem;

public class Player : MonoBehaviour
{
    [SerializeField] private float acceleration = 50f;
    [SerializeField] private float accSprintMultiplier = 4f;
    [SerializeField] private float mouseSensitivity = 2f;
    [SerializeField] private float dampingCoefficient = 5f;
    [SerializeField] private float maxSpeed = 20f;
    
    private Vector3 velocity;
    private float rotationX = 0f;
    
    [Header("Inventory")]
    public Inventory inventory;

    private Vector2 moveInput;
    private Vector2 lookInput;
    private float verticalMoveInput;

    private bool isMouseLocked = false;
    private Vector2 previousLookInput;

    void Start()
    {
        // Lock and hide the cursor
        LockMouse();
    }

    private void LockMouse()
    {
        Cursor.lockState = CursorLockMode.Locked;
        Cursor.visible = false;
        isMouseLocked = true;
    }

    private void UnlockMouse()
    {
        Cursor.lockState = CursorLockMode.None;
        Cursor.visible = true;
        isMouseLocked = false;
    }

    void Update()
    {
        // Toggle mouse lock with Escape
        if (Keyboard.current.escapeKey.wasPressedThisFrame)
        {
            if (isMouseLocked)
                UnlockMouse();
            else
                LockMouse();
        }

        // Only process mouse look when locked
        if (!isMouseLocked)
            return;

        // Re-lock cursor if clicked
        if (Mouse.current.leftButton.wasPressedThisFrame)
        {
            LockMouse();
        }

        HandleUpdate();
    }

    public void OnMove(InputValue value)
    {
        moveInput = value.Get<Vector2>();
    }

    public void OnLook(InputValue value)
    {
        lookInput = value.Get<Vector2>();
    }

    public void OnUpDown(InputValue value)
    {
        verticalMoveInput = value.Get<float>();
    }

    public void HandleUpdate()
    {
        Vector3 cameraForward = Camera.main.transform.forward;
        Vector3 cameraRight = Camera.main.transform.right;
        
        cameraRight.y = 0;
        cameraRight.Normalize();

        Vector3 moveDirection = cameraRight * moveInput.x 
                              + cameraForward * moveInput.y 
                              + Vector3.up * verticalMoveInput;

        // Decide final acceleration based on shift
        float adjustedAcceleration = Keyboard.current.leftShiftKey.isPressed 
            ? acceleration * accSprintMultiplier 
            : acceleration;

        // Apply that acceleration
        velocity += moveDirection.normalized * adjustedAcceleration * Time.deltaTime;

        // Dampen velocity
        velocity = Vector3.Lerp(velocity, Vector3.zero, dampingCoefficient * Time.deltaTime);

        // (Optionally clamp if you’d like a max speed)
        velocity = Vector3.ClampMagnitude(velocity, maxSpeed);

        transform.position += velocity * Time.deltaTime;

        // Handle mouse rotation when cursor is locked
        if (isMouseLocked)
        {
            float mouseX = lookInput.x * mouseSensitivity;
            float mouseY = lookInput.y * mouseSensitivity;

            rotationX -= mouseY;
            rotationX = Mathf.Clamp(rotationX, -90f, 90f);

            transform.Rotate(Vector3.up * mouseX);
            Camera.main.transform.localRotation = Quaternion.Euler(rotationX, 0f, 0f);
        }
    }
}
