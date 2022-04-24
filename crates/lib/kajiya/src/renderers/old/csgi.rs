// Cone sweep global illumination prototype

use std::sync::Arc;

use glam::{IVec4, Vec3};
use kajiya_backend::{
    ash::vk,
    vk_sync::AccessType,
    vulkan::{
        image::*,
        ray_tracing::RayTracingAcceleration,
        shader::{
            PipelineShaderDesc, RasterPipelineDesc, RenderPass, ShaderPipelineStage, ShaderSource,
        },
    },
};
use kajiya_rg::{
    self as rg, BindRgRef, GetOrCreateTemporal, IntoRenderPassPipelineBinding, SimpleRenderPass,
};

use rust_shaders_shared::frame_constants::GiCascadeConstants;

use super::GbufferDepth;

// VOLUME_DIMS and CASCADE_COUNT must match GPU code.
// Seach token: d4109bba-438f-425e-8667-19e591be9a56
pub const VOLUME_DIMS: u32 = 64;
pub const CASCADE_COUNT: usize = 1;
const SCROLL_CASCADES: bool = false;
const VOLUME_WORLD_SCALE_MULT: f32 = 1.0;
//pub const VOLUME_DIMS: u32 = 40;
//pub const CASCADE_COUNT: usize = 4;
//const SCROLL_CASCADES: bool = true;
//const VOLUME_WORLD_SCALE_MULT: f32 = 0.5;

// Search token: b518ed19-c715-4cc7-9bc7-e0dbbca3e037
const CASCADE_EXP_SCALING_RATIO: f32 = 2.0;

pub struct CsgiRenderer {
    pub trace_subdiv: i32,
    pub neighbors_per_frame: i32,
    frame_idx: u32,
    cur_scroll: [CascadeScroll; CASCADE_COUNT],
    prev_scroll: [CascadeScroll; CASCADE_COUNT],
}

impl Default for CsgiRenderer {
    fn default() -> Self {
        Self {
            trace_subdiv: 3,
            neighbors_per_frame: 2,
            frame_idx: 0,
            cur_scroll: Default::default(),
            prev_scroll: Default::default(),
        }
    }
}

pub struct CsgiVolume {
    pub direct: [rg::Handle<Image>; CASCADE_COUNT],
    pub indirect: [rg::Handle<Image>; CASCADE_COUNT],
    pub subray_indirect: [rg::Handle<Image>; CASCADE_COUNT],
    pub opacity: [rg::Handle<Image>; CASCADE_COUNT],
}

#[derive(Default, Clone, Copy)]
struct CascadeScroll {
    scroll: [i32; 4],
}

impl CascadeScroll {
    fn volume_scroll_offset_from(&self, other: &Self) -> [i32; 4] {
        array_init::array_init(|i| {
            (self.scroll[i] - other.scroll[i]).clamp(-(VOLUME_DIMS as i32), VOLUME_DIMS as i32)
        })
    }
}

impl CsgiRenderer {
    pub fn volume_size(gi_scale: f32) -> f32 {
        12.0 * VOLUME_WORLD_SCALE_MULT * gi_scale
    }

    pub fn voxel_size(gi_scale: f32) -> f32 {
        Self::volume_size(gi_scale) / VOLUME_DIMS as f32
    }

    fn grid_center(focus_position: &Vec3, cascade_scale: f32) -> Vec3 {
        ((*focus_position) / Self::voxel_size(cascade_scale)).round()
            * Self::voxel_size(cascade_scale)
    }

    pub fn update_eye_position(&mut self, eye_position: &Vec3, gi_scale: f32) {
        // The grid is shifted, so the _center_ and not _origin_ of cascade 0 is at origin.
        // This moves the seams in the origin-centered grid away from origin.
        // Must match GPU code. Search token: 3e7ddeec-afbb-44e4-8b75-b54276c6df2b
        let grid_offset_world_space: f32 = -Self::voxel_size(gi_scale) * (VOLUME_DIMS as f32 / 2.0);

        // Transform the eye position to the grid-local space
        let eye_position = *eye_position - grid_offset_world_space;

        let cascade_scales: [f32; CASCADE_COUNT] =
            array_init::array_init(|i| CASCADE_EXP_SCALING_RATIO.powi(i as i32));

        let gi_volume_scroll: [CascadeScroll; CASCADE_COUNT] =
            array_init::array_init(|cascade_i| {
                //(round(get_eye_position() / CSGI_VOXEL_SIZE) * CSGI_VOXEL_SIZE)
                let cascade_scale = gi_scale * cascade_scales[cascade_i];

                let voxel_size = Self::voxel_size(cascade_scale);

                let gi_grid_center = Self::grid_center(&eye_position, cascade_scale);
                let gi_volume_scroll: Vec3 = gi_grid_center / voxel_size;

                CascadeScroll {
                    scroll: if SCROLL_CASCADES {
                        let gi_volume_scroll: Vec3 =
                            gi_volume_scroll + gi_volume_scroll.signum() * 0.5;
                        [
                            gi_volume_scroll.x as i32
                                - crate::renderers::csgi::VOLUME_DIMS as i32 / 2,
                            gi_volume_scroll.y as i32
                                - crate::renderers::csgi::VOLUME_DIMS as i32 / 2,
                            gi_volume_scroll.z as i32
                                - crate::renderers::csgi::VOLUME_DIMS as i32 / 2,
                            0,
                        ]
                    } else {
                        [0i32; 4]
                    },
                }
            });

        self.prev_scroll = self.cur_scroll;
        self.cur_scroll = gi_volume_scroll;
    }

    pub fn constants(&self, gi_scale: f32) -> [GiCascadeConstants; CASCADE_COUNT] {
        let cascade_scales: [f32; CASCADE_COUNT] =
            array_init::array_init(|i| CASCADE_EXP_SCALING_RATIO.powi(i as i32));

        array_init::array_init(|cascade_i| {
            let cascade_scale = gi_scale * cascade_scales[cascade_i];
            let volume_size = Self::volume_size(cascade_scale);
            let voxel_size = Self::voxel_size(cascade_scale);

            let mut gi_volume_scroll_frac = [0i32; 4];
            let mut gi_volume_scroll_int = [0i32; 4];

            {
                let gi_volume_scroll = self.cur_scroll[cascade_i].scroll;

                for (k, i) in gi_volume_scroll.iter().copied().enumerate() {
                    let g = i.wrapping_div_euclid(crate::renderers::csgi::VOLUME_DIMS as i32)
                        * crate::renderers::csgi::VOLUME_DIMS as i32;
                    let r = i - g;
                    gi_volume_scroll_frac[k] = r;
                    gi_volume_scroll_int[k] = g;
                }
            };

            //dbg!(gi_volume_scroll_frac, gi_volume_scroll_int);

            let voxels_scrolled_this_frame: IVec4 = self.cur_scroll[cascade_i]
                .volume_scroll_offset_from(&self.prev_scroll[cascade_i])
                .into();

            GiCascadeConstants {
                scroll_frac: gi_volume_scroll_frac.into(),
                scroll_int: gi_volume_scroll_int.into(),
                voxels_scrolled_this_frame,
                volume_size,
                voxel_size,
                ..Default::default()
            }
        })
    }

    pub fn render(
        &mut self,
        _eye_position: Vec3,
        rg: &mut rg::TemporalRenderGraph,
        sky_cube: &rg::Handle<Image>,
        bindless_descriptor_set: vk::DescriptorSet,
        tlas: &rg::Handle<RayTracingAcceleration>,
    ) -> CsgiVolume {
        let CsgiVolume {
            direct: mut direct_cascades,
            indirect: mut indirect_combined_cascades,
            subray_indirect: mut indirect_cascades,
            opacity: mut opacity_cascades,
        } = self.create_volume(rg);

        // Stagger cascade updates over frames
        //let cascade_update_mask = 1usize << (self.frame_idx as usize % CASCADE_COUNT);
        let cascade_update_mask = !0usize;

        // Advance quantum values as necessary (frame index slowed down to account for the stager).
        let quantum_idx: u32 = self.frame_idx / CASCADE_COUNT as u32;

        let sweep_vx_count = VOLUME_DIMS >> self.trace_subdiv.clamp(0, 5);

        // Note: The sweep shaders should be dispatched as individual slices, and synchronized
        // in-between with barries (no cache flushes), so that propagation can happen without race conditions.
        // At the very least, it could be dispatched as one slice, but looping inside the shaders would use atomic
        // work stealing instead of slice iteration.
        //
        // Either is going to leave the GPU quite under-utilized though, and would be a nice fit for async compute.

        for cascade_i in 0..CASCADE_COUNT {
            if 0 == ((1 << cascade_i) & cascade_update_mask) {
                continue;
            }

            SimpleRenderPass::new_compute(
                rg.add_pass("csgi decay"),
                "/shaders/csgi/decay_volume.hlsl",
            )
            .write(&mut direct_cascades[cascade_i])
            .write(&mut indirect_combined_cascades[cascade_i])
            .write(&mut indirect_cascades[cascade_i])
            .constants(cascade_i as u32)
            .dispatch([
                VOLUME_DIMS * CARDINAL_DIRECTION_COUNT as u32,
                VOLUME_DIMS as u32,
                VOLUME_DIMS,
            ]);

            SimpleRenderPass::new_rt(
                rg.add_pass("csgi trace"),
                ShaderSource::hlsl("/shaders/csgi/trace_volume.rgen.hlsl"),
                [
                    ShaderSource::hlsl("/shaders/rt/gbuffer.rmiss.hlsl"),
                    ShaderSource::hlsl("/shaders/rt/shadow.rmiss.hlsl"),
                ],
                [ShaderSource::hlsl("/shaders/rt/gbuffer.rchit.hlsl")],
            )
            .read_array(&indirect_combined_cascades)
            .read(sky_cube)
            .write(&mut direct_cascades[cascade_i])
            .constants((sweep_vx_count, cascade_i as u32, quantum_idx))
            .raw_descriptor_set(1, bindless_descriptor_set)
            .trace_rays(
                tlas,
                [
                    VOLUME_DIMS * CARDINAL_DIRECTION_COUNT as u32,
                    VOLUME_DIMS as u32,
                    VOLUME_DIMS / sweep_vx_count,
                ],
            );
        }

        /*let mut direct_opacity = rg.create(ImageDesc::new_3d(
            vk::Format::R8_UNORM,
            [VOLUME_DIMS, VOLUME_DIMS, VOLUME_DIMS],
        ));*/

        let null_cascade = rg.create(
            ImageDesc::new_3d(vk::Format::B10G11R11_UFLOAT_PACK32, [1, 1, 1])
                .usage(vk::ImageUsageFlags::SAMPLED),
        );

        for cascade_i in (0..CASCADE_COUNT).rev() {
            if 0 == ((1 << cascade_i) & cascade_update_mask) {
                continue;
            }

            let direct_opacity = &mut opacity_cascades[cascade_i];

            SimpleRenderPass::new_compute(
                rg.add_pass("csgi direct opacity sum"),
                "/shaders/csgi/direct_opacity_sum.hlsl",
            )
            .read(&direct_cascades[cascade_i])
            .write(direct_opacity)
            .dispatch([VOLUME_DIMS, VOLUME_DIMS, VOLUME_DIMS]);

            let (inner_indirect_cascades, outer_indirect_cascades) =
                indirect_cascades.split_at_mut(cascade_i + 1);

            let outer_cascade_subray_indirect =
                outer_indirect_cascades.first().unwrap_or(&null_cascade);
            let indirect_cascade = inner_indirect_cascades.last_mut().unwrap();

            SimpleRenderPass::new_compute(
                rg.add_pass("csgi diagonal sweep"),
                "/shaders/csgi/diagonal_sweep_volume.hlsl",
            )
            .read(&direct_cascades[cascade_i])
            .read(sky_cube)
            .read(direct_opacity)
            .read(outer_cascade_subray_indirect)
            .write(indirect_cascade)
            .write(&mut indirect_combined_cascades[cascade_i])
            .constants((cascade_i as u32, quantum_idx))
            .dispatch([VOLUME_DIMS, VOLUME_DIMS, DIAGONAL_DIRECTION_COUNT as u32]);

            SimpleRenderPass::new_compute(
                rg.add_pass("csgi sweep"),
                "/shaders/csgi/sweep_volume.hlsl",
            )
            .read(&direct_cascades[cascade_i])
            .read(sky_cube)
            .read(direct_opacity)
            .read(outer_cascade_subray_indirect)
            .write(indirect_cascade)
            .write(&mut indirect_combined_cascades[cascade_i])
            .constants((cascade_i as u32, quantum_idx))
            .dispatch([VOLUME_DIMS, VOLUME_DIMS, CARDINAL_DIRECTION_COUNT as u32]);
        }

        /*SimpleRenderPass::new_compute(
            rg.add_pass("csgi subray combine"),
            "/shaders/csgi/subray_combine.hlsl",
        )
        .read(&mut indirect_cascade0)
        .read(&direct_cascade0)
        .read(&direct_opacity_cascade0)
        .write(&mut indirect_cascade_combined0)
        .dispatch([VOLUME_DIMS * (TRACE_COUNT as u32), VOLUME_DIMS, VOLUME_DIMS]);*/

        self.frame_idx += 1;

        CsgiVolume {
            direct: direct_cascades,
            indirect: indirect_combined_cascades,
            subray_indirect: indirect_cascades,
            opacity: opacity_cascades,
        }
    }

    pub fn create_volume(&self, rg: &mut rg::TemporalRenderGraph) -> CsgiVolume {
        self.create_volume_with_dimensions(
            rg,
            [
                VOLUME_DIMS * CARDINAL_DIRECTION_COUNT as u32,
                VOLUME_DIMS,
                VOLUME_DIMS,
            ],
            [
                VOLUME_DIMS * TOTAL_SUBRAY_COUNT as u32,
                VOLUME_DIMS,
                VOLUME_DIMS,
            ],
            [
                VOLUME_DIMS * TOTAL_DIRECTION_COUNT as u32,
                VOLUME_DIMS,
                VOLUME_DIMS,
            ],
            [VOLUME_DIMS, VOLUME_DIMS, VOLUME_DIMS],
        )
    }

    pub fn create_dummy_volume(&self, rg: &mut rg::TemporalRenderGraph) -> CsgiVolume {
        self.create_volume_with_dimensions(rg, [1; 3], [1; 3], [1; 3], [1; 3])
    }

    pub(crate) fn create_volume_with_dimensions(
        &self,
        rg: &mut rg::TemporalRenderGraph,
        direct_cascade_dimensions: [u32; 3],
        indirect_cascade_dimensions: [u32; 3],
        indirect_combined_cascade_dimensions: [u32; 3],
        opacity_cascade_dimensions: [u32; 3],
    ) -> CsgiVolume {
        let direct_cascades: [rg::Handle<Image>; CASCADE_COUNT] = array_init::array_init(|i| {
            rg.get_or_create_temporal(
                format!("csgi.direct_cascade{}", i),
                ImageDesc::new_3d(
                    //vk::Format::B10G11R11_UFLOAT_PACK32,
                    vk::Format::R16G16B16A16_SFLOAT,
                    direct_cascade_dimensions,
                )
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE),
            )
            .unwrap()
        });

        let indirect_cascades: [rg::Handle<Image>; CASCADE_COUNT] = array_init::array_init(|i| {
            rg.get_or_create_temporal(
                format!("csgi.indirect_cascade{}", i),
                ImageDesc::new_3d(
                    vk::Format::B10G11R11_UFLOAT_PACK32,
                    //vk::Format::R16G16B16A16_SFLOAT,
                    indirect_cascade_dimensions,
                )
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE),
            )
            .unwrap()
        });

        let indirect_combined_cascades: [rg::Handle<Image>; CASCADE_COUNT] =
            array_init::array_init(|i| {
                rg.get_or_create_temporal(
                    format!("csgi.indirect_cascade_combined{}", i),
                    ImageDesc::new_3d(
                        vk::Format::B10G11R11_UFLOAT_PACK32,
                        //vk::Format::R16G16B16A16_SFLOAT,
                        indirect_combined_cascade_dimensions,
                    )
                    .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE),
                )
                .unwrap()
            });

        let opacity_cascades: [rg::Handle<Image>; CASCADE_COUNT] = array_init::array_init(|_| {
            rg.create(ImageDesc::new_3d(
                vk::Format::R8_UNORM,
                opacity_cascade_dimensions,
            ))
        });

        CsgiVolume {
            direct: direct_cascades,
            indirect: indirect_combined_cascades,
            subray_indirect: indirect_cascades,
            opacity: opacity_cascades,
        }
    }
}

impl CsgiVolume {
    pub fn debug_raster_voxel_grid(
        &self,
        rg: &mut rg::RenderGraph,
        render_pass: Arc<RenderPass>,
        gbuffer_depth: &mut GbufferDepth,
        velocity_img: &mut rg::Handle<Image>,
        cascade_idx: usize,
    ) {
        let mut pass = rg.add_pass("raster csgi voxels");

        let pipeline = pass.register_raster_pipeline(
            &[
                PipelineShaderDesc::builder(ShaderPipelineStage::Vertex)
                    .hlsl_source("/shaders/csgi/raster_voxels_vs.hlsl")
                    .build()
                    .unwrap(),
                PipelineShaderDesc::builder(ShaderPipelineStage::Pixel)
                    .hlsl_source("/shaders/csgi/raster_voxels_ps.hlsl")
                    .build()
                    .unwrap(),
            ],
            RasterPipelineDesc::builder()
                .render_pass(render_pass.clone())
                .face_cull(true),
        );

        let depth_ref = pass.raster(
            &mut gbuffer_depth.depth,
            AccessType::DepthAttachmentWriteStencilReadOnly,
        );

        let geometric_normal_ref = pass.raster(
            &mut gbuffer_depth.geometric_normal,
            AccessType::ColorAttachmentWrite,
        );
        let gbuffer_ref = pass.raster(&mut gbuffer_depth.gbuffer, AccessType::ColorAttachmentWrite);
        let velocity_ref = pass.raster(velocity_img, AccessType::ColorAttachmentWrite);

        let grid_ref = pass.read(
            &self.direct[cascade_idx],
            AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer,
        );

        pass.render(move |api| {
            let [width, height, _] = gbuffer_ref.desc().extent;

            api.begin_render_pass(
                &*render_pass,
                [width, height],
                &[
                    (geometric_normal_ref, &ImageViewDesc::default()),
                    (gbuffer_ref, &ImageViewDesc::default()),
                    (velocity_ref, &ImageViewDesc::default()),
                ],
                Some((
                    depth_ref,
                    &ImageViewDesc::builder()
                        .aspect_mask(vk::ImageAspectFlags::DEPTH)
                        .build()
                        .unwrap(),
                )),
            );

            api.set_default_view_and_scissor([width, height]);

            let constants_offset = api.dynamic_constants().push(&(cascade_idx as u32));
            let _ = api.bind_raster_pipeline(pipeline.into_binding().descriptor_set(
                0,
                &[
                    grid_ref.bind(),
                    rg::RenderPassBinding::DynamicConstants(constants_offset),
                ],
            ));

            unsafe {
                let raw_device = &api.device().raw;
                let cb = api.cb;

                raw_device.cmd_draw(
                    cb.raw,
                    // 6 verts (two triangles) per cube face
                    6 * CARDINAL_DIRECTION_COUNT as u32 * VOLUME_DIMS * VOLUME_DIMS * VOLUME_DIMS,
                    1,
                    0,
                    0,
                );
            }

            api.end_render_pass();
        });
    }

    pub fn fullscreen_debug_radiance(
        &self,
        rg: &mut rg::RenderGraph,
        output: &mut rg::Handle<Image>,
    ) {
        SimpleRenderPass::new_compute(
            rg.add_pass("csgi debug radiance"),
            "/shaders/csgi/debug_radiance.hlsl",
        )
        .read_array(&self.direct)
        .read_array(&self.indirect)
        .read_array(&self.subray_indirect)
        .write(output)
        .constants(output.desc().extent_inv_extent_2d())
        .dispatch(output.desc().extent);
    }
}

const CARDINAL_DIRECTION_COUNT: usize = 6;
const CARDINAL_SUBRAY_COUNT: usize = 5;

const DIAGONAL_DIRECTION_COUNT: usize = 8;
const DIAGONAL_SUBRAY_COUNT: usize = 3;

const TOTAL_DIRECTION_COUNT: usize = CARDINAL_DIRECTION_COUNT + DIAGONAL_DIRECTION_COUNT;
const TOTAL_SUBRAY_COUNT: usize = CARDINAL_DIRECTION_COUNT * CARDINAL_SUBRAY_COUNT
    + DIAGONAL_DIRECTION_COUNT * DIAGONAL_SUBRAY_COUNT;
