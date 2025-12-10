using UnityEngine;

public class CommandController : MonoBehaviour
{
    [Header("참조 객체")]
    public UDPReceiver udp;                // Network 오브젝트에 붙어 있는 UDPReceiver
    public Transform player;               // 제어할 대상 (캐릭터, Cube 등)
    public Renderer playerRenderer;        // 색깔로 상태를 표시하기 위한 Renderer

    [Header("이동 설정")]
    public float moveSpeed = 4f;           // 기본 이동 속도
    public bool moveByCamera = false;      // true이면 카메라 방향 기준으로 이동

    [Header("물리 이동 옵션")]
    public bool usePhysicsMove = true;     // Rigidbody 기반 물리 이동을 사용할지 여부
    public bool makeRigidbodyKinematic = false; // 물리 이동을 쓰지 않을 때는 Kinematic으로 둘지 여부

    [Header("점프 설정")]
    public float jumpForce = 8f;           // 점프 힘
    public float jumpCooldown = 0.25f;     // 점프 쿨타임(초 단위)
    public LayerMask groundMask;           // 바닥 레이어 (Ground 레이어 지정)
    public float groundCheckOffsetY = 0.1f;// 바닥 체크용 Ray 출발 위치 Y 오프셋
    public float groundCheckDist = 0.6f;   // 바닥 체크 Ray 길이

    [Header("융합 정책 (손/음성 우선순위)")]
    public float gesturePriorityWindow = 0.8f; // 최근 손 제스처 이동 명령이 우선되는 시간(초)
    public float actionCooldown = 0.2f;        // 제스처 액션(ATTACK/DEFEND) 쿨타임 (음성은 거의 쿨타임 없음)

    [Header("무기 / 방패 비주얼")]
    public Renderer swordRenderer;             // ATTACK 시 강조할 검
    public Renderer shieldRenderer;            // DEFEND 시 강조할 방패
    public float actionFlashTime = 0.2f;       // 공격/방어 색상 유지 시간

    [Header("디버그 옵션")]
    public bool logCmd = true;            // 수신된 명령 로그 출력 여부
    public bool logMove = false;          // 실제 이동 벡터 로그 출력 여부

    // -------- 내부 상태 --------
    private string lastCmd = "";          // 마지막으로 처리한 명령
    private Vector3 moveDir = Vector3.zero; // 현재 이동 방향 (FORWARD/BACKWARD/LEFT/RIGHT)
    private Rigidbody rb;
    private Material cachedMat;
    private bool jumping = false;         // 점프 중인지 여부
    private float lastJumpTime = -999f;   // 마지막 점프 시간

    // 융합 관련 상태 (손/음성)
    private float lastGestureMoveTime = -999f; // 마지막 손 제스처 이동 명령 시간
    private float lastActionTime = -999f;      // 마지막 액션(ATTACK/DEFEND) 시점

    // 검/방패 기본 색깔 및 유지 시간
    private Color swordDefaultColor;
    private Color shieldDefaultColor;
    private float swordColorUntil = 0f;
    private float shieldColorUntil = 0f;

    void Awake()
    {
        // player가 지정되지 않았다면 자기 자신을 사용
        if (!player) player = transform;

        rb = player.GetComponent<Rigidbody>();
        if (playerRenderer) cachedMat = playerRenderer.material;

        // 검/방패 기본색 저장
        if (swordRenderer)
            swordDefaultColor = swordRenderer.material.color;
        if (shieldRenderer)
            shieldDefaultColor = shieldRenderer.material.color;

        if (rb)
        {
            // 캐릭터가 넘어지지 않도록 회전 고정
            rb.constraints = RigidbodyConstraints.FreezeRotation;

            // 이동 방식에 따른 Rigidbody 설정
            if (!usePhysicsMove && makeRigidbodyKinematic)
            {
                rb.isKinematic = true;
            }
            else if (usePhysicsMove)
            {
                rb.isKinematic = false;
            }
        }
    }

    void Update()
    {
        // 로컬 테스트용: 스페이스바로 점프 테스트
        if (Input.GetKeyDown(KeyCode.Space))
        {
            TryJump();
        }

        // UDP 또는 player 참조가 없으면 이동 처리만 수행
        if (!udp || !player) goto MOVE_UPDATE;

        // UDPReceiver에서 마지막으로 받은 문자열
        string s = udp.data;

        // [ ... ] 로 시작하는 JSON 로그 등은 무시
        if (!string.IsNullOrEmpty(s) && s.Length > 0 && s[0] != '[')
        {
            // 형식: "G:FORWARD" / "V:ATTACK" / 그 외
            string src = (s.Length > 2 && s[1] == ':') ? s.Substring(0, 1) : "?"; // "G"(손) / "V"(음성) / "?"
            string cmd = (s.Length > 2 && s[1] == ':') ? s.Substring(2).ToUpper() : s.Trim().ToUpper();

            if (logCmd) Debug.Log($"[CMD] src={src} cmd={cmd}");

            // ===== 이동 계열 명령 (FORWARD/BACKWARD/LEFT/RIGHT/STOP) =====
            if (cmd == "FORWARD" || cmd == "BACKWARD" || cmd == "LEFT" || cmd == "RIGHT" || cmd == "STOP")
            {
                // 손 제스처 이동이면 "최근 제스처 이동 시간" 업데이트
                if (src == "G") lastGestureMoveTime = Time.time;

                // gesturePriorityWindow 동안은 손 제스처 이동 명령을 우선시
                bool gesturePreferred = (Time.time - lastGestureMoveTime) <= gesturePriorityWindow;

                // 손에서 왔거나, 손 우선 시간이 끝난 뒤의 음성 명령이라면 이동 방향 갱신
                if (src == "G" || !gesturePreferred || src == "?")
                {
                    if (cmd != lastCmd)
                    {
                        lastCmd = cmd;
                        ApplyCommand(cmd);   // 여기서 moveDir / 몸 색상 변경
                    }
                }
            }
            // ===== 액션 계열 명령 (ATTACK / DEFEND / JUMP) =====
            else if (cmd == "ATTACK" || cmd == "DEFEND" || cmd == "JUMP")
            {
                bool isVoice = (src == "V");

                if (cmd == "JUMP")
                {
                    TryJump();
                }
                else
                {
                    // 음성 명령은 거의 쿨타임 없이 허용, 제스처 액션은 쿨타임 적용
                    if (isVoice || Time.time - lastActionTime >= actionCooldown)
                    {
                        lastCmd = cmd;
                        ApplyCommand(cmd);                      // 여기서 검/방패 이펙트
                        lastActionTime = isVoice
                            ? Time.time - (actionCooldown - 0.05f)
                            : Time.time;
                    }
                }
            }
        }

    // ===== 실제 이동 처리 =====
    MOVE_UPDATE:
        // 비물리 이동 모드: Transform.Translate 사용
        if (!usePhysicsMove && moveDir != Vector3.zero)
        {
            Vector3 dir = ResolveMoveDir(moveDir);
            player.Translate(dir * moveSpeed * Time.deltaTime, Space.World);
            if (logMove) Debug.Log($"[MOVE][Translate] dir={dir} v={moveSpeed}");
        }

        // 검/방패 색상 상태 업데이트
        UpdateActionVisuals();
    }

    void FixedUpdate()
    {
        // 물리 이동 모드일 때만 Rigidbody.MovePosition 사용
        if (!usePhysicsMove || moveDir == Vector3.zero || !rb) return;

        Vector3 dir = ResolveMoveDir(moveDir);
        Vector3 next = player.position + dir * moveSpeed * Time.fixedDeltaTime;
        rb.MovePosition(next);
        if (logMove) Debug.Log($"[MOVE][Physics] dir={dir} v={moveSpeed}");
    }

    // ===== 점프 시도 (쿨타임 + 바닥 체크 포함) =====
    void TryJump()
    {
        if (!rb) return;
        if (Time.time - lastJumpTime < jumpCooldown) return;
        if (!IsGrounded()) return;

        if (usePhysicsMove) rb.isKinematic = false;

        rb.AddForce(Vector3.up * jumpForce, ForceMode.Impulse);
        lastJumpTime = Time.time;
        jumping = true;
        Tint(Color.yellow);
        if (logCmd) Debug.Log("[JUMP] Jump triggered");
    }

    // 바닥에 붙어 있는지 체크 (Raycast 사용)
    bool IsGrounded()
    {
        Vector3 origin = player.position + Vector3.up * groundCheckOffsetY;
        bool hit = Physics.Raycast(
            origin,
            Vector3.down,
            groundCheckDist,
            groundMask,
            QueryTriggerInteraction.Ignore
        );

        jumping = !hit ? jumping : false;
        return hit;
    }

    Vector3 ResolveMoveDir(Vector3 raw)
    {
        if (moveByCamera && Camera.main)
        {
            return Camera.main.transform.TransformDirection(raw).normalized.WithY(0f).normalized;
        }
        return raw;
    }

    // ===== 명령 적용 (이동/액션/색상) =====
    void ApplyCommand(string cmd)
    {
        switch (cmd)
        {
            // ---- 지속 이동 계열 ----
            case "FORWARD":
                moveDir = Vector3.forward;
                Tint(Color.green);       // 몸을 초록색
                break;
            case "BACKWARD":
                moveDir = Vector3.back;
                Tint(Color.green);
                break;
            case "LEFT":
                moveDir = Vector3.left;
                Tint(Color.green);
                break;
            case "RIGHT":
                moveDir = Vector3.right;
                Tint(Color.green);
                break;

            // STOP: 이동을 멈추고 기본 색으로
            case "STOP":
                moveDir = Vector3.zero;
                Tint(Color.white);
                break;

            // ---- 액션 계열 (이동은 유지, 검/방패만 색상 강조) ----
            case "ATTACK":
                TriggerAttackVisual();
                break;
            case "DEFEND":
                TriggerDefendVisual();
                break;

            case "JUMP":
                TryJump();
                break;

            default:
                if (cmd.Contains("_"))
                {
                    var parts = cmd.Split('_');
                    foreach (var p in parts)
                    {
                        ApplyCommand(p);
                    }
                }
                break;
        }
    }

    // 본체 색상 바꾸기 (이동 상태 시각화)
    void Tint(Color c)
    {
        if (!playerRenderer) return;
        if (cachedMat == null) cachedMat = playerRenderer.material;
        cachedMat.color = c;
    }

    // ATTACK: 검 빨간색
    void TriggerAttackVisual()
    {
        if (!swordRenderer) return;
        var m = swordRenderer.material;
        m.color = Color.red;
        swordColorUntil = Time.time + actionFlashTime;
    }

    // DEFEND: 방패 파란색
    void TriggerDefendVisual()
    {
        if (!shieldRenderer) return;
        var m = shieldRenderer.material;
        m.color = Color.blue;
        shieldColorUntil = Time.time + actionFlashTime;
    }

    // 매 프레임 검/방패 색상 복원
    void UpdateActionVisuals()
    {
        if (swordRenderer && Time.time > swordColorUntil)
        {
            swordRenderer.material.color = swordDefaultColor;
        }
        if (shieldRenderer && Time.time > shieldColorUntil)
        {
            shieldRenderer.material.color = shieldDefaultColor;
        }
    }
}

// Vector3 유틸 확장 메서드 (Y 값만 바꾸는 헬퍼)
public static class VecExt
{
    public static Vector3 WithY(this Vector3 v, float y)
    {
        v.y = y;
        return v;
    }
}
