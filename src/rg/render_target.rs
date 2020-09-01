/*use super::{GpuRt, Image, Ref};

pub struct ColorAttachment {
    pub texture: Ref<Image, GpuRt>,
    pub info: RenderTargetInfo,
}

impl From<(Ref<Image, GpuRt>, RenderTargetInfo)> for ColorAttachment {
    fn from(v: (Ref<Image, GpuRt>, RenderTargetInfo)) -> Self {
        Self {
            texture: v.0,
            info: v.1,
        }
    }
}

pub type ColorAttachments = [Option<ColorAttachment>; MAX_RENDER_TARGET_COUNT];

pub struct RenderTarget {
    pub(crate) color: ColorAttachments,
}

pub trait IntoColorAttachments {
    fn into_color_attachments(self) -> ColorAttachments;
}

macro_rules! impl_into_color_attachments {
    ($count:expr, $($elems:ident),*) => {
        impl IntoColorAttachments for [(Ref<Image, GpuRt>, RenderTargetInfo); $count] {
            #[allow(unused_assignments)]
            fn into_color_attachments(self) -> ColorAttachments {
                let mut color: [Option<ColorAttachment>; MAX_RENDER_TARGET_COUNT] =
                    array_init::array_init(|_| None);

                let [$($elems),*] = self;
                let mut i = 0;
                $(
                    color[i] = Some($elems.into());
                    i += 1;
                )*

                color
            }
        }
    };
}

impl_into_color_attachments! {1, e1}
impl_into_color_attachments! {2, e1, e2}
impl_into_color_attachments! {3, e1, e2, e3}
impl_into_color_attachments! {4, e1, e2, e3, e4}
impl_into_color_attachments! {5, e1, e2, e3, e4, e5}
impl_into_color_attachments! {6, e1, e2, e3, e4, e5, e6}
impl_into_color_attachments! {7, e1, e2, e3, e4, e5, e6, e7}
impl_into_color_attachments! {8, e1, e2, e3, e4, e5, e6, e7, e8}

impl RenderTarget {
    pub fn new(color: impl IntoColorAttachments) -> Self {
        Self {
            color: color.into_color_attachments(),
        }
    }
}

impl RenderTarget {
    pub fn to_draw_state(self: &RenderTarget) -> RenderDrawState {
        let [width, height] = self.color[0]
            .as_ref()
            .expect("render target color attachment")
            .texture
            .desc()
            .dims();

        RenderDrawState {
            viewport: Some(RenderViewportRect {
                x: 0.0,
                y: 0.0,
                width: width as f32,
                height: height as f32,
                min_z: 0.0,
                max_z: 1.0,
            }),
            scissor: Some(RenderScissorRect {
                x: 0,
                y: 0,
                width: width as i32,
                height: height as i32,
            }),
            stencil_ref: 0,
        }
    }
}
*/
