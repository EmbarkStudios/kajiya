use kajiya_simple::Vec3;

#[derive(Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct Sequence {
    items: Vec<SequenceItem>,
}

#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemOption<T> {
    item: T,
    pub is_some: bool,
}

impl<T> MemOption<T> {
    pub fn new(item: T) -> Self {
        Self {
            item,
            is_some: true,
        }
    }

    pub fn as_option(&self) -> Option<T>
    where
        T: Copy,
    {
        self.is_some.then_some(self.item)
    }

    pub fn unwrap_or(&self, other: T) -> T
    where
        T: Copy,
    {
        if self.is_some {
            self.item
        } else {
            other
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SequenceValue {
    pub camera_position: MemOption<Vec3>,
    pub camera_direction: MemOption<Vec3>,
    pub towards_sun: MemOption<Vec3>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SequenceFullValue {
    pub camera_position: Vec3,
    pub camera_direction: Vec3,
    pub towards_sun: Vec3,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SequenceItem {
    pub t: f32,
    pub value: SequenceValue,
}

impl SequenceItem {
    pub fn new(t: f32, value: SequenceValue) -> Self {
        Self { t, value }
    }
}

pub struct SequenceItemMut<'a> {
    pub value: &'a mut SequenceValue,
    pub duration: f32,
}

impl Sequence {
    pub fn add_keyframe(&mut self, after: Option<usize>, value: SequenceValue) {
        let idx = after.map_or_else(|| self.items.len(), |idx| idx + 1);

        let t_delta = 1.0;
        let prev_t = self
            .items
            .get(idx.saturating_sub(1))
            .map_or(-t_delta, |k| k.t);

        // Insert with the same `t` as the previous one, then shift after.
        self.items.insert(idx, SequenceItem::new(prev_t, value));

        self.apply_t_delta_from_index(idx, t_delta);
    }

    fn apply_t_delta_from_index(&mut self, start: usize, t_delta: f32) {
        for k in &mut self.items[start..] {
            k.t += t_delta;
        }
    }

    pub fn to_playback(&self) -> CameraPlaybackSequence {
        CameraPlaybackSequence {
            duration: self.items.last().map_or(0.0, |item| item.t),
            camera_position_spline: splines::Spline::from_iter(self.items.iter().filter_map(|k| {
                Some(splines::Key::new(
                    k.t,
                    k.value.camera_position.as_option()?,
                    splines::Interpolation::CatmullRom,
                ))
            })),
            camera_direction_spline: splines::Spline::from_iter(self.items.iter().filter_map(
                |k| {
                    Some(splines::Key::new(
                        k.t,
                        k.value.camera_direction.as_option()?,
                        splines::Interpolation::CatmullRom,
                    ))
                },
            )),
            towards_sun_spline: splines::Spline::from_iter(self.items.iter().filter_map(|k| {
                Some(splines::Key::new(
                    k.t,
                    k.value.towards_sun.as_option()?,
                    splines::Interpolation::CatmullRom,
                ))
            })),
        }
    }

    pub fn get_item(&self, i: usize) -> Option<&SequenceItem> {
        self.items.get(i)
    }

    pub fn delete_key(&mut self, i: usize) {
        let t_delta = self
            .items
            .get(i + 1)
            .map_or(0.0, |next| self.items[i].t - next.t);

        self.items.remove(i);

        self.apply_t_delta_from_index(i, t_delta);
    }

    pub fn each_key(&mut self, mut callback: impl FnMut(usize, &mut SequenceItemMut)) {
        let len = self.items.len();
        for i in 0..len {
            let duration = self
                .items
                .get(i + 1)
                .map_or(1.0, |p_next| p_next.t - self.items[i].t);

            let item = self.items.get_mut(i).unwrap();

            let mut item = SequenceItemMut {
                value: &mut item.value,
                duration,
            };

            callback(i, &mut item);

            item.duration = item.duration.max(0.01);

            if item.duration != duration {
                let shift = item.duration - duration;
                self.apply_t_delta_from_index(i + 1, shift);
            }
        }
    }
}

pub struct CameraPlaybackSequence {
    duration: f32,
    camera_position_spline: splines::Spline<f32, Vec3>,
    camera_direction_spline: splines::Spline<f32, Vec3>,
    towards_sun_spline: splines::Spline<f32, Vec3>,
}

impl CameraPlaybackSequence {
    pub fn sample(&mut self, t: f32) -> Option<SequenceFullValue> {
        if t > self.duration {
            return None;
        }

        let camera_position = self.camera_position_spline.clamped_sample(t)?;
        let camera_direction = self.camera_direction_spline.clamped_sample(t)?;
        let towards_sun = self.towards_sun_spline.clamped_sample(t)?;

        Some(SequenceFullValue {
            camera_position,
            camera_direction,
            towards_sun,
        })
    }
}
