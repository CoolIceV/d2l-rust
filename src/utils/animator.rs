use std::{thread, time::Duration, vec};

use minifb::{Window, WindowOptions, Scale};
use plotters::prelude::*;
use std::collections::HashMap;

const COLORS1: [RGBColor;3] = [RED, BLUE, GREEN];

pub struct Animator {
    lines: HashMap<String, Vec<(f64, f64)>>,
    label: HashMap<String, usize>,
    window: Window,
}

impl Animator {
    pub fn new(labels: &[&str]) -> Self {
        Animator {
            lines: labels.iter().map(|&s| (s.to_string(), vec![])).collect(),
            label: labels.iter().enumerate().map(|(i, &s)| (s.to_string(), i)).collect(),
            window: Window::new(
                "Real-time Plot",
                600,
                400,
                WindowOptions {
                    resize: true,
                    // scale: Scale::X2,
                    ..WindowOptions::default()
                },
            ).unwrap(),
        }
    }

    pub fn add_point(&mut self, line: &str, x: f64, y: f64) {
        if let Some(line) = self.lines.get_mut(line) {
            line.push((x, y));
        }
    }

    pub fn draw(&mut self) {
        let mut buf = vec![0; 300 * 200 * 3];
        let slice_buf: &mut [u8] = &mut buf;
        {
            let root = BitMapBackend::with_buffer(slice_buf, (300, 200)).into_drawing_area();

            root.fill(&WHITE).unwrap();

            let mut chart = ChartBuilder::on(&root)
                .caption("Animator", ("sans-serif", 12).into_font())
                .margin(5)
                .x_label_area_size(30)
                .y_label_area_size(30)
                .build_cartesian_2d(0f32..(1000 as f32), 0f32..1.)
                .unwrap();

            chart.configure_mesh().draw().unwrap();
            for (label, line) in self.lines.iter() {
                let style1 = &COLORS1[self.label.get(label).unwrap() % COLORS1.len()];
                chart.draw_series(
                    LineSeries::new(
                        line.iter().map(|&(x, y)| (x as f32, y as f32)), 
                        style1)).unwrap()
                        .label(label)
                        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &style1.clone()));
            
            }            
           

            chart
                .configure_series_labels()
                .background_style(&WHITE.mix(0.8))
                .border_style(&BLACK)
                .draw()
                .unwrap();

            root.present().unwrap();
        }

        let u32_slice: Vec<u32> = slice_buf
            .chunks_exact(3) // Group every 3 bytes
            .map(|chunk| {
                let value = (chunk[0] as u32) << 16 | (chunk[1] as u32) << 8 | (chunk[2] as u32);
                value
            })
            .collect();
        // Update the window with the buffer content
        self.window
            .update_with_buffer(&u32_slice, 300, 200)
            .unwrap();
    }
}