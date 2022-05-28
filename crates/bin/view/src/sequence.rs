use kajiya_simple::Vec3;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Key<T> {
    t: f32,
    value: T,
}

impl<T> Key<T> {
    pub fn new(t: f32, value: T) -> Self {
        Self { t, value }
    }
}

#[derive(Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct CameraSequence {
    pub position_spline: Vec<Key<Vec3>>, // TODO: remove pub
    pub rotation_spline: Vec<Key<Vec3>>, // TODO: remove pub
}

pub struct CameraSequenceKey {
    pub position: Vec3,
    pub rotation: Vec3,
    pub duration: f32,
}

impl CameraSequence {
    pub fn add_keyframe(&mut self, after: Option<usize>, position: Vec3, rotation: Vec3) {
        let idx = after.map_or_else(|| self.position_spline.len(), |idx| idx + 1);

        let t_delta = 1.0;
        let prev_t = self
            .position_spline
            .get(idx.saturating_sub(1))
            .map_or(-t_delta, |k| k.t);

        // Insert with the same `t` as the previous one, then shift after.
        self.position_spline.insert(idx, Key::new(prev_t, position));
        self.rotation_spline.insert(idx, Key::new(prev_t, rotation));

        self.apply_t_delta_from_index(idx, t_delta);
    }

    fn apply_t_delta_from_index(&mut self, start: usize, t_delta: f32) {
        for k in &mut self.position_spline[start..] {
            k.t += t_delta;
        }
        for k in &mut self.rotation_spline[start..] {
            k.t += t_delta;
        }
    }

    pub fn to_playback(&self) -> CameraPlaybackSequence {
        CameraPlaybackSequence {
            position_spline: splines::Spline::from_iter(
                self.position_spline
                    .iter()
                    .map(|k| splines::Key::new(k.t, k.value, splines::Interpolation::CatmullRom)),
            ),
            rotation_spline: splines::Spline::from_iter(
                self.rotation_spline
                    .iter()
                    .map(|k| splines::Key::new(k.t, k.value, splines::Interpolation::CatmullRom)),
            ),
        }
    }

    pub fn get_value(&self, i: usize) -> Option<(f32, Vec3, Vec3)> {
        let p_key = self.position_spline.get(i)?;
        let r_key = self.rotation_spline.get(i)?;
        Some((p_key.t, p_key.value, r_key.value))
    }

    pub fn delete_key(&mut self, i: usize) {
        let t_delta = self
            .position_spline
            .get(i + 1)
            .map_or(0.0, |next| self.position_spline[i].t - next.t);

        self.position_spline.remove(i);
        self.rotation_spline.remove(i);

        self.apply_t_delta_from_index(i, t_delta);
    }

    pub fn each_key(&mut self, mut callback: impl FnMut(usize, &mut CameraSequenceKey)) {
        let len = self.position_spline.len();
        for i in 0..len {
            let duration = self
                .position_spline
                .get(i + 1)
                .map_or(1.0, |p_next| p_next.t - self.position_spline[i].t);

            let p_key = self.position_spline.get_mut(i).unwrap();
            let r_key = self.rotation_spline.get_mut(i).unwrap();

            let mut key = CameraSequenceKey {
                position: p_key.value,
                rotation: r_key.value,
                duration,
            };

            callback(i, &mut key);
            key.duration = key.duration.max(0.01);

            p_key.value = key.position;
            r_key.value = key.rotation;

            if key.duration != duration {
                let shift = key.duration - duration;
                self.apply_t_delta_from_index(i + 1, shift);
            }
        }
    }
}

pub struct CameraPlaybackSequence {
    position_spline: splines::Spline<f32, Vec3>,
    rotation_spline: splines::Spline<f32, Vec3>,
}

impl CameraPlaybackSequence {
    pub fn sample(&mut self, t: f32) -> Option<(Vec3, Vec3)> {
        let position = self.position_spline.sample(t)?;
        let direction = self.rotation_spline.sample(t)?;
        Some((position, direction))
    }
}
