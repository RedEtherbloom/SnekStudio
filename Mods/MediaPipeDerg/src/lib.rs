// We may mish to remove this once we decided a) on a name and b) a proper structure for rust code to live in SnekStudio
#![allow(non_snake_case)]
use godot::classes::{ISprite2D, Sprite2D};
use godot::prelude::*;

pub mod hand_detection;
pub mod models;

struct MediaPipeDergExtension;

#[gdextension]
unsafe impl ExtensionLibrary for MediaPipeDergExtension {}

#[derive(GodotClass)]
#[class(base=Sprite2D)]
struct ExampleTest {
    speed: f64,
    angular_speed: f64,

    base: Base<Sprite2D>,
}

#[godot_api]
impl ISprite2D for ExampleTest {
    fn init(base: Base<Sprite2D>) -> Self {
        godot_print!("I live! Rawr >w> I'm a dregon. Fear me!");

        Self {
            speed: 400.0,
            angular_speed: std::f64::consts::PI,
            base,
        }
    }

    fn physics_process(&mut self, delta: f64) {
        let radians = (self.angular_speed * delta) as f32;
        godot_print!("New radians: {radians}");
        self.base_mut().rotate(radians);
    }
}
