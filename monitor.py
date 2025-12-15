import psutil
import time
import subprocess
from rich.console import Console
from rich.table import Table
from rich.live import Live

console = Console()

PROCESSES = [
    "synexs_flask_dashboard.py",
    "synexs_core_loop2.0.py"
]

def get_live_process(name):
    try:
        for p in psutil.process_iter(["pid", "name", "cmdline", "cpu_percent", "memory_info", "create_time"]):
            if name in " ".join(p.info["cmdline"] or []) and "python" in " ".join(p.info["cmdline"] or []):
                return "RUNNING", f"{p.info['cpu_percent']:.1f}", f"{p.info['memory_info'].rss / 1024 ** 2:.1f}MB", f"{int((time.time() - p.info['create_time']) // 3600)}h {int(((time.time() - p.info['create_time']) % 3600) // 60)}m {int((time.time() - p.info['create_time']) % 60)}s", str(p.info["pid"])
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        pass
    return "DOWN", "0.0", "0.0MB", "Never", "-"

def make_dashboard():
    table = Table(title="SYNEXS VPS MONITOR - VAN EDITION", show_header=True, header_style="bold cyan")
    table.add_column("Process", style="magenta")
    table.add_column("Status", justify="center")
    table.add_column("CPU %", style="yellow")
    table.add_column("RAM", style="green")
    table.add_column("Uptime", style="dim")
    table.add_column("PID", style="dim")

    for proc in PROCESSES:
        status, cpu, ram, uptime, pid = get_live_process(proc)
        color = "green" if status == "RUNNING" else "red" if status == "DOWN" else "yellow"
        table.add_row(proc, f"[{color}]{status}[/{color}]", cpu, ram, uptime, pid)

    try:
        cpu_total = psutil.cpu_percent(interval=1)
        ram_total = psutil.virtual_memory().percent
        disk = psutil.disk_usage("/").percent
        uptime_sys = subprocess.getoutput("uptime -p").replace("up ", "")
        table.add_row("", "", "", "", "", "")
        table.add_row("SYSTEM", "", f"{cpu_total}%", f"{ram_total}%", uptime_sys, "")
        table.add_row("", "", "", "", f"Disk: {disk}%", "")
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, subprocess.CalledProcessError):
        table.add_row("SYSTEM", "", "N/A", "N/A", "N/A", "")
        table.add_row("", "", "", "", "N/A", "")

    return table

if __name__ == "__main__":
    try:
        with Live(make_dashboard(), refresh_per_second=1, console=console) as live:
            while True:
                live.update(make_dashboard())
                time.sleep(2)
    except (KeyboardInterrupt, Exception) as e:
        console.print(f"Unexpected error: {e}")
        raise e