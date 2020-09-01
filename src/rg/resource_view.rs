//use std::sync::Arc;

pub mod srv {
    use super::super::resource::*;

    pub struct RgSrv {
        // TODO: non-texture
        pub rg_ref: Ref<Image, GpuSrv>,
    }

    pub fn texture_2d(rg_ref: Ref<Image, GpuSrv>) -> RgSrv {
        RgSrv {
            rg_ref: rg_ref.internal_clone(),
        }
    }
}

pub mod uav {
    use super::super::resource::*;

    pub struct RgUav {
        // TODO: non-texture
        pub rg_ref: Ref<Image, GpuUav>,
    }

    pub fn texture_2d(rg_ref: Ref<Image, GpuUav>) -> RgUav {
        RgUav { rg_ref }
    }
}

/*pub trait NamedShaderViews {
    fn named_views(
        &self,
        registry: &ResourceRegistry,
        srvs: &[(&'static str, srv::RgSrv)],
        uavs: &[(&'static str, uav::RgUav)],
    ) -> RenderResourceHandle;
}

impl NamedShaderViews for Arc<ComputePipeline> {
    fn named_views(
        &self,
        registry: &ResourceRegistry,
        srvs: &[(&'static str, srv::RgSrv)],
        uavs: &[(&'static str, uav::RgUav)],
    ) -> RenderResourceHandle {
        let mut resource_views = RenderShaderViewsDesc {
            shader_resource_views: vec![Default::default(); srvs.len()],
            unordered_access_views: vec![Default::default(); uavs.len()],
        };

        for (srv_name, srv) in srvs.into_iter() {
            let binding_idx = self
                .srvs
                .iter()
                .position(|name| name == srv_name)
                .expect(srv_name);

            // TODO: other binding types
            resource_views.shader_resource_views[binding_idx] = build::texture_2d(
                registry.resource(srv.rg_ref.internal_clone()).0,
                srv.rg_ref.desc().format,
                0,
                1,
                0,
                0.0f32,
            );
        }

        for (uav_name, uav) in uavs.into_iter() {
            let binding_idx = self
                .uavs
                .iter()
                .position(|name| name == uav_name)
                .expect(uav_name);

            // TODO: other binding types
            resource_views.unordered_access_views[binding_idx] = build::texture_2d_rw(
                registry.resource(uav.rg_ref.internal_clone()).0,
                uav.rg_ref.desc().format,
                0,
                0,
            );
        }

        // TODO: verify that all entries have been written to

        let resource_views_handle = registry
            .execution_params
            .handles
            .allocate_transient(RenderResourceType::ShaderViews);

        registry
            .execution_params
            .device
            .create_shader_views(
                resource_views_handle,
                &resource_views,
                "shader resource views".into(),
            )
            .unwrap();

        resource_views_handle
    }
}
*/
