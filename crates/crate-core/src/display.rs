// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Screens and input as an app sees them: frames out of the machine, keys and
//! pointer positions into it.

/// A rectangle in pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Rect {
    pub const fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Rect {
            x,
            y,
            width,
            height,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    fn right(&self) -> u64 {
        self.x as u64 + self.width as u64
    }

    fn bottom(&self) -> u64 {
        self.y as u64 + self.height as u64
    }

    /// The smallest rectangle holding both. An empty one adds nothing.
    pub fn union(&self, other: &Rect) -> Rect {
        if self.is_empty() {
            return *other;
        }
        if other.is_empty() {
            return *self;
        }
        let x = self.x.min(other.x);
        let y = self.y.min(other.y);
        let right = self.right().max(other.right());
        let bottom = self.bottom().max(other.bottom());
        Rect::new(x, y, (right - x as u64) as u32, (bottom - y as u64) as u32)
    }

    /// The part inside both; empty if they do not meet.
    pub fn intersect(&self, other: &Rect) -> Rect {
        let x = self.x.max(other.x);
        let y = self.y.max(other.y);
        let right = self.right().min(other.right());
        let bottom = self.bottom().min(other.bottom());
        if right <= x as u64 || bottom <= y as u64 {
            return Rect::default();
        }
        Rect::new(x, y, (right - x as u64) as u32, (bottom - y as u64) as u32)
    }

    /// The part inside a `width` x `height` screen.
    pub fn clip(&self, width: u32, height: u32) -> Rect {
        self.intersect(&Rect::new(0, 0, width, height))
    }
}

/// A whole screen and the part of it that changed since the previous frame.
pub struct Frame<'a> {
    pub width: u32,
    pub height: u32,
    /// Bytes per row.
    pub stride: u32,
    /// Four bytes a pixel: red, green, blue, unused.
    pub pixels: &'a [u8],
    pub dirty: Rect,
}

/// A size the guest is offered for its screen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mode {
    pub width: u32,
    pub height: u32,
}

impl Mode {
    pub const fn new(width: u32, height: u32) -> Self {
        Mode { width, height }
    }
}

/// Where a display's frames go.
pub trait DisplaySink: Send + Sync {
    /// Whether anyone shows the display now. A frame costs a copy out of the
    /// GPU, so none is made while this is false.
    fn active(&self) -> bool;
    /// A new frame, on the GPU's thread. Its pixels are valid only during the
    /// call.
    fn frame(&self, frame: &Frame);
}

/// What a viewer can ask of a display.
pub trait DisplayControl: Send + Sync {
    /// Asks for a whole frame soon, whether or not the guest draws: for a
    /// viewer that has just become active and has nothing to show.
    fn refresh(&self);
}

/// The buttons of `InputSink::pointer`.
pub const BUTTON_LEFT: u32 = 1 << 0;
pub const BUTTON_RIGHT: u32 = 1 << 1;
pub const BUTTON_MIDDLE: u32 = 1 << 2;

/// The largest pointer coordinate: 0 to this spans the screen.
pub const POINTER_MAX: u32 = 32767;

/// Input for the guest's keyboard, absolute tablet and relative mouse.
pub trait InputSink: Send + Sync {
    /// `code` is a Linux input event code (KEY_A is 30).
    fn key(&self, code: u16, down: bool);
    /// An absolute position, 0 to `POINTER_MAX` across the screen in each
    /// direction, and the buttons held (`BUTTON_*`).
    fn pointer(&self, x: u32, y: u32, buttons: u32);
    /// Wheel notches: positive `dy` down, positive `dx` right.
    fn scroll(&self, dx: i32, dy: i32);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rectangles() {
        let a = Rect::new(10, 10, 20, 20);
        let b = Rect::new(25, 0, 10, 15);
        assert_eq!(a.union(&b), Rect::new(10, 0, 25, 30));
        assert_eq!(a.intersect(&b), Rect::new(25, 10, 5, 5));
        assert!(a.intersect(&Rect::new(40, 40, 1, 1)).is_empty());
        assert_eq!(Rect::default().union(&b), b);
        assert_eq!(a.clip(15, 100), Rect::new(10, 10, 5, 20));
        assert!(Rect::new(u32::MAX, 0, u32::MAX, 1).clip(8, 8).is_empty());
    }
}
