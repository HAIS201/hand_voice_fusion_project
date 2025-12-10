using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

public class UDPReceiver : MonoBehaviour
{
    [Header("UDP 설정")]
    public int listenPort = 5052;

    [Header("디버그")]
    public bool logPackets = false;

    // 다른 스크립트에서 읽을 수 있는 가장 최근 수신 문자열
    [HideInInspector] public string data = "";

    private UdpClient client;
    private Thread recvThread;
    private volatile bool running;

    void Start()
    {
        try
        {
            // 지정한 포트로 UDP 소켓 바인딩
            client = new UdpClient(listenPort);
            running = true;

            // 수신 전용 스레드 시작 (백그라운드)
            recvThread = new Thread(RecvLoop) { IsBackground = true };
            recvThread.Start();

            Debug.Log($"[UDPReceiver] Listening on UDP {listenPort}");
        }
        catch (SocketException e)
        {
            Debug.LogError($"[UDPReceiver] Port bind failed: {e.Message}");
        }
    }

    void RecvLoop()
    {
        IPEndPoint any = new IPEndPoint(IPAddress.Any, 0);

        while (running)
        {
            try
            {
                // 블로킹 방식으로 패킷 수신
                byte[] bytes = client.Receive(ref any);
                string msg = Encoding.UTF8.GetString(bytes);

                // 가장 최근에 받은 메시지를 그대로 보관
                data = msg;

                if (logPackets)
                    Debug.Log($"[UDPReceiver] {msg}");
            }
            catch (SocketException)
            {
                // 소켓이 닫히는 타이밍에 발생할 수 있는 예외는 무시
            }
            catch
            {
                // 일시적인 기타 예외도 조용히 무시
            }
        }
    }

    void OnDestroy()
    {
        // 스레드 루프 종료 플래그
        running = false;

        // 소켓 및 스레드 정리
        try { client?.Close(); } catch { }
        try { recvThread?.Join(100); } catch { }
    }
}
