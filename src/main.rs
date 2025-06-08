use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::f32::consts::PI;

struct Oscillator {
    phase: f32,
    frequency: f32,
    amplitude: f32,
}

impl Oscillator {
    fn new(frequency: f32, amplitude: f32) -> Self {
        Self { phase: 0.0, frequency, amplitude }
    }
    
    fn next_sample(&mut self, sample_rate: f32) -> f32 {
        let sample = (self.phase * 2.0 * PI).sin() * self.amplitude;
        self.phase += self.frequency / sample_rate;
        self.phase %= 1.0;
        sample
    }
}

fn main() {
    let host = cpal::default_host();
    let device = host.default_output_device().expect("no output device");
    let config = device.default_output_config().expect("no default config");
    let sample_rate = config.sample_rate().0 as f32;
    
    let mut oscillators = vec![
        Oscillator::new(110.0, 0.1),
        Oscillator::new(330.0, 0.05),
        Oscillator::new(550.0, 0.03),
    ];
    
    let stream = device.build_output_stream(
        &config.into(),
        move |data: &mut [f32], _| {
            for sample in data {
                *sample = oscillators.iter_mut()
                    .map(|osc| osc.next_sample(sample_rate))
                    .sum();
            }
        },
        |err| eprintln!("Error: {}", err),
        None  // No timestamp needed
    ).expect("failed to build stream");
    
    stream.play().expect("failed to play");
    std::io::stdin().read_line(&mut String::new()).unwrap();
}