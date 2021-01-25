pub mod rtr;
pub mod ssgi;
pub mod surfel_gi;

#[macro_export]
macro_rules! impl_renderer_temporal_logic {
    ($inst_type:tt, $($res_name:ident,)*) => {
        pub fn begin(&mut self, rg: &mut rg::RenderGraph) -> $inst_type {
            self.on_begin();

            $inst_type {
                $(
                    $res_name: rg.import_temporal(&mut self.$res_name),
                )*
            }
        }
        pub fn end(&mut self, rg: &mut rg::RenderGraph, inst: $inst_type) {
            $(
                rg.export_temporal(inst.$res_name, &mut self.$res_name, $crate::vk_sync::AccessType::Nothing);
            )*
        }

        pub fn retire(&mut self, rg: &rg::RetiredRenderGraph) {
            $(
                rg.retire_temporal(&mut self.$res_name);
            )*
        }
    };
}
