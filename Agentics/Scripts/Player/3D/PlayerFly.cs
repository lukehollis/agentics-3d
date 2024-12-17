using UnityEngine;
using UnityEngine.InputSystem;
using Agentics;

namespace Agentics
{
    public class PlayerFly : Player
    {
        [Header("Camera Settings")]
        public GameObject CinemachineCameraTarget;
        public float TopClamp = 89.0f;
        public float BottomClamp = -89.0f;
        
        private float _cinemachineTargetYaw;
        private float _cinemachineTargetPitch;
        private const float _threshold = 0.01f;
 
        private PlayerInput _playerInput;
        private InputAction _lookAction;
        private InputAction _moveAction;
        private InputAction _upDownAction;
        
        private Vector3 _currentVelocity;
        private float _verticalMovement;

        protected override void Awake()
        {
            mainCamera = Camera.main;
            if (animator == null) animator = GetComponent<Animator>();
            
            // Setup input system
            _playerInput = GetComponent<PlayerInput>();
            if (_playerInput == null)
                _playerInput = gameObject.AddComponent<PlayerInput>();
                
            _lookAction = _playerInput.actions["Look"];
            _moveAction = _playerInput.actions["Move"];
            _upDownAction = _playerInput.actions["UpDown"];

            // Initialize camera rotation
            _cinemachineTargetYaw = CinemachineCameraTarget.transform.rotation.eulerAngles.y;
            _cinemachineTargetPitch = CinemachineCameraTarget.transform.rotation.eulerAngles.x;

            // Lock and hide cursor
            Cursor.lockState = CursorLockMode.Locked;
            Cursor.visible = false;
        }

        protected override void ConfigureNavMeshAgent()
        {
            // Not using NavMeshAgent
        }

        private void Update()
        {
            HandleMovementInput();
            HandleInteractionInput();
            UpdateAnimations();

            // Toggle cursor lock with Escape
            if (Keyboard.current.escapeKey.wasPressedThisFrame)
            {
                Cursor.lockState = Cursor.lockState == CursorLockMode.Locked ? 
                                 CursorLockMode.None : CursorLockMode.Locked;
                Cursor.visible = !Cursor.visible;
            }
        }

        private void LateUpdate()
        {
            HandleCameraRotation();
        }

        private void HandleCameraRotation()
        {
            Vector2 lookInput = _lookAction.ReadValue<Vector2>();
            
            Debug.Log($"Look input: {lookInput}");
            Debug.Log($"Before - Pitch: {_cinemachineTargetPitch}, Yaw: {_cinemachineTargetYaw}");
            
            if (lookInput.sqrMagnitude >= _threshold)
            {
                _cinemachineTargetYaw += lookInput.x * rotationSpeed;
                _cinemachineTargetPitch += -lookInput.y * rotationSpeed;
            }

            // Clamp rotation
            _cinemachineTargetYaw = ClampAngle(_cinemachineTargetYaw, float.MinValue, float.MaxValue);
            _cinemachineTargetPitch = ClampAngle(_cinemachineTargetPitch, BottomClamp, TopClamp);

            Quaternion targetRotation = Quaternion.Euler(_cinemachineTargetPitch, _cinemachineTargetYaw, 0.0f);
            Debug.Log($"After - Pitch: {_cinemachineTargetPitch}, Yaw: {_cinemachineTargetYaw}");
            Debug.Log($"Target Rotation: {targetRotation.eulerAngles}");
            
            CinemachineCameraTarget.transform.rotation = targetRotation;
            Debug.Log($"Actual Rotation: {CinemachineCameraTarget.transform.rotation.eulerAngles}");
        }

        private static float ClampAngle(float lfAngle, float lfMin, float lfMax)
        {
            if (lfAngle < -360f) lfAngle += 360f;
            if (lfAngle > 360f) lfAngle -= 360f;
            return Mathf.Clamp(lfAngle, lfMin, lfMax);
        }

        protected override void HandleMovementInput()
        {
            float currentSpeed = runningSpeed * Time.deltaTime;

            // Get input for movement
            Vector2 moveInput = _moveAction.ReadValue<Vector2>();
            
            // Get camera's forward and right vectors, but only on the horizontal plane
            Vector3 forward = mainCamera.transform.forward;
            forward.y = 0;
            forward.Normalize();
            
            Vector3 right = mainCamera.transform.right;
            right.y = 0;
            right.Normalize();

            // Calculate horizontal movement direction relative to camera
            Vector3 movement = (forward * moveInput.y + right * moveInput.x);

            // Handle up/down movement separately in world space
            float verticalInput = _upDownAction.ReadValue<float>();
            Vector3 verticalMovement = Vector3.up * verticalInput * currentSpeed;

            // Apply movements separately
            transform.position += movement.normalized * currentSpeed;
            transform.position += verticalMovement;
            
            // Update movement type for animations
            currentMoveType = (movement.magnitude > 0.1f || Mathf.Abs(verticalInput) > 0.1f) ? 
                MoveType.Walking : MoveType.Idle;

            // Store movement direction for other systems
            moveDirection = movement + verticalMovement;
        }

        protected override void HandleInteractionInput()
        {
            // if (Input.GetMouseButtonDown(0))
            // {
            //     Ray ray = mainCamera.ScreenPointToRay(Input.mousePosition);
            //     RaycastHit hit;

            //     if (Physics.Raycast(ray, out hit))
            //     {
            //         IInteractable interactable = hit.collider.GetComponent<IInteractable>();
            //         if (interactable != null)
            //         {
            //             float distance = Vector3.Distance(transform.position, hit.point);
            //             if (distance <= interactionRange)
            //             {
            //                 interactable.Interact();
            //             }
            //         }
            //     }
            // }
        }

        protected override void UpdateAnimations()
        {
            if (animator != null)
            {
                animator.SetFloat("Speed", moveDirection.magnitude);
                animator.SetBool("IsWalking", currentMoveType == MoveType.Walking);
                animator.SetBool("IsRunning", currentMoveType == MoveType.Running);
            }
        }

        protected override void FaceTarget(Vector3 targetPosition)
        {
            Vector3 lookPosition = targetPosition;
            lookPosition.y = transform.position.y;
            transform.LookAt(lookPosition);
        }
    }
}
