using UnityEngine;

public class ThirdPersonCamera : MonoBehaviour
{
    [Header("따라갈 대상")]
    public Transform target;   // Player 루트 오브젝트를 넣기

    [Header("3인칭 카메라 위치")]
    public float distance = 6f; // 캐릭터 뒤쪽 거리
    public float height = 3f;   // 캐릭터 위쪽 높이

    [Header("부드러운 따라가기")]
    public float followSmooth = 8f; // 클수록 빠르게 따라감

    [Header("카메라 각도")]
    public float lookHeight = 1.5f; // 캐릭터의 어느 높이를 바라볼지

    void LateUpdate()
    {
        if (!target) return;

        // Player의 뒤쪽 방향 계산
        Vector3 backDirection = -target.forward;

        // 목표 카메라 위치 = Player 뒤쪽 + 위쪽
        Vector3 desiredPosition =
            target.position +
            backDirection * distance +
            Vector3.up * height;

        // 부드럽게 위치 이동
        transform.position = Vector3.Lerp(
            transform.position,
            desiredPosition,
            followSmooth * Time.deltaTime
        );

        // 카메라가 바라볼 위치
        Vector3 lookTarget = target.position + Vector3.up * lookHeight;

        // 카메라가 Player를 바라보게 회전
        transform.LookAt(lookTarget);
    }
}