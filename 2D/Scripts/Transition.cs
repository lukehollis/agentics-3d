using UnityEngine;
using UnityEngine.AI;
using Cinemachine;
using System.Collections;

public class Transition : MonoBehaviour
{
    private Transform destination; // The destination where the player will be teleported to
    private CinemachineVirtualCamera virtualCamera; // Reference to the Cinemachine virtual camera
    public ScreenFader screenFader; // Reference to the ScreenFader

    private void Start()
    {
        // Find the destination transform in the child objects
        destination = transform.Find("Destination");
        if (destination == null)
        {
            Debug.LogError("Destination transform not found in children.");
        }

        // Find the Cinemachine virtual camera in the scene
        virtualCamera = FindObjectOfType<CinemachineVirtualCamera>();
        if (virtualCamera == null)
        {
            Debug.LogError("CinemachineVirtualCamera not found in the scene.");
        }

        // screenFader = FindObjectOfType<ScreenFader>();
        // if (screenFader == null)
        // {
        //     Debug.LogError("ScreenFader not found in the scene.");
        // }
        // else
        // {
        //     screenFader.gameObject.SetActive(false); // Deactivate ScreenFader at the start
        //     Debug.Log("ScreenFader found and assigned.");
        // }
    }

    private void OnTriggerEnter2D(Collider2D other)
    {
        // if (other.CompareTag("Player") || other.CompareTag("NPC"))
        Debug.Log("OnTriggerEnter2D");
        if (other.CompareTag("Player"))
        {
            StartCoroutine(TransitionRoutine(other));
        }
    }

    private IEnumerator TransitionRoutine(Collider2D other)
    {
        screenFader.gameObject.SetActive(true); // Activate ScreenFader before fading out
        yield return StartCoroutine(screenFader.FadeOut());

        NavMeshAgent agent = other.GetComponent<NavMeshAgent>();
        if (agent != null)
        {
            agent.enabled = false; // Disable the NavMeshAgent
        }

        other.transform.position = destination.position;

        if (virtualCamera != null && other.CompareTag("Player"))
        {
            virtualCamera.ForceCameraPosition(destination.position, virtualCamera.transform.rotation);
        }

        if (agent != null)
        {
            agent.enabled = true; // Re-enable the NavMeshAgent
        }

        yield return StartCoroutine(screenFader.FadeIn());
        screenFader.gameObject.SetActive(false); // Deactivate ScreenFader after fading in
    }
}