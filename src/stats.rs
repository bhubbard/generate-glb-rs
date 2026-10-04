use colored::Colorize;
use std::time::Instant;
use sysinfo::{Pid, System};

/// Performance and resource statistics tracker matching generate.py.
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    pub start_time: Instant,
    pub model_load_time: f32,
    pub generation_time: f32,
    pub export_time: f32,
    pub total_time: f32,
    pub initial_memory_mb: f32,
    pub peak_memory_mb: f32,
    pub cpu_usage: f32,
    pub gpu_device: Option<String>,
}

impl Default for PerformanceStats {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceStats {
    pub fn new() -> Self {
        let mut sys = System::new_all();
        sys.refresh_all();
        let initial_mem = get_process_memory_mb(&sys);
        let gpu = detect_gpu();

        Self {
            start_time: Instant::now(),
            model_load_time: 0.0,
            generation_time: 0.0,
            export_time: 0.0,
            total_time: 0.0,
            initial_memory_mb: initial_mem,
            peak_memory_mb: initial_mem,
            cpu_usage: 0.0,
            gpu_device: gpu,
        }
    }

    pub fn update_memory(&mut self) {
        let mut sys = System::new_all();
        sys.refresh_processes(sysinfo::ProcessesToUpdate::All, true);
        let mem = get_process_memory_mb(&sys);
        self.peak_memory_mb = self.peak_memory_mb.max(mem);
    }

    pub fn finish(&mut self) {
        self.total_time = self.start_time.elapsed().as_secs_f32();
        let mut sys = System::new_all();
        sys.refresh_cpu_all();
        self.cpu_usage = sys.global_cpu_usage();
        self.update_memory();
    }

    pub fn print_report(&self) {
        println!("\n{}", "Performance Statistics:".bold());
        println!("{}", "--------------------------------------------------".dimmed());
        println!("Total Processing Time: {:.2} seconds", self.total_time);
        println!("├─ Model Load Time: {:.2} seconds", self.model_load_time);
        println!("├─ Generation Time: {:.2} seconds", self.generation_time);
        println!("└─ Export Time:     {:.2} seconds", self.export_time);

        println!("\n{}", "Memory Usage:".bold());
        println!("├─ Initial: {:.1} MB", self.initial_memory_mb);
        println!("├─ Peak:    {:.1} MB", self.peak_memory_mb);
        let delta = (self.peak_memory_mb - self.initial_memory_mb).max(0.0);
        println!("└─ Delta:   {:.1} MB", delta);

        println!("\nCPU Usage: {:.1}%", self.cpu_usage);

        if let Some(ref gpu) = self.gpu_device {
            println!("\n{}", "GPU Statistics:".bold());
            println!("├─ Device: {}", gpu.green());
            println!("└─ Status: {}", "Active and available".green());
        }
    }
}

fn get_process_memory_mb(sys: &System) -> f32 {
    let pid = Pid::from_u32(std::process::id());
    if let Some(process) = sys.process(pid) {
        (process.memory() as f32) / (1024.0 * 1024.0)
    } else {
        0.0
    }
}

fn detect_gpu() -> Option<String> {
    #[cfg(target_os = "macos")]
    {
        Some("Apple Metal (MPS)".to_string())
    }
    #[cfg(not(target_os = "macos"))]
    {
        if std::path::Path::new("/dev/nvidia0").exists() {
            Some("NVIDIA CUDA".to_string())
        } else {
            None
        }
    }
}
