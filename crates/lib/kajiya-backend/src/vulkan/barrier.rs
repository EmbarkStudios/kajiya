use ash::vk;
use vk_sync::AccessType;

use super::device::Device;

pub struct ImageBarrier {
    image: vk::Image,
    prev_access: vk_sync::AccessType,
    next_access: vk_sync::AccessType,
    aspect_mask: vk::ImageAspectFlags,
    discard: bool,
}

pub fn record_image_barrier(device: &Device, cb: vk::CommandBuffer, barrier: ImageBarrier) {
    let range = vk::ImageSubresourceRange {
        aspect_mask: barrier.aspect_mask,
        base_mip_level: 0,
        level_count: vk::REMAINING_MIP_LEVELS,
        base_array_layer: 0,
        layer_count: vk::REMAINING_ARRAY_LAYERS,
    };

    vk_sync::cmd::pipeline_barrier(
        device.raw.fp_v1_0(),
        cb,
        None,
        &[],
        &[vk_sync::ImageBarrier {
            previous_accesses: &[barrier.prev_access],
            next_accesses: &[barrier.next_access],
            previous_layout: vk_sync::ImageLayout::Optimal,
            next_layout: vk_sync::ImageLayout::Optimal,
            discard_contents: barrier.discard,
            src_queue_family_index: device.universal_queue.family.index,
            dst_queue_family_index: device.universal_queue.family.index,
            image: barrier.image,
            range,
        }],
    );
}

impl ImageBarrier {
    pub fn new(
        image: vk::Image,
        prev_access: vk_sync::AccessType,
        next_access: vk_sync::AccessType,
        aspect_mask: vk::ImageAspectFlags,
    ) -> Self {
        Self {
            image,
            prev_access,
            next_access,
            discard: false,
            aspect_mask,
        }
    }

    pub fn with_discard(mut self, discard: bool) -> Self {
        self.discard = discard;
        self
    }
}

// From vk_sync
pub struct AccessInfo {
    pub stage_mask: vk::PipelineStageFlags,
    pub access_mask: vk::AccessFlags,
    pub image_layout: vk::ImageLayout,
}

pub fn get_access_info(access_type: AccessType) -> AccessInfo {
    match access_type {
        AccessType::Nothing => AccessInfo {
            stage_mask: vk::PipelineStageFlags::empty(),
            access_mask: vk::AccessFlags::empty(),
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::CommandBufferReadNVX => AccessInfo {
            stage_mask: vk::PipelineStageFlags::COMMAND_PREPROCESS_NV,
            access_mask: vk::AccessFlags::COMMAND_PREPROCESS_READ_NV,
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::IndirectBuffer => AccessInfo {
            stage_mask: vk::PipelineStageFlags::DRAW_INDIRECT,
            access_mask: vk::AccessFlags::INDIRECT_COMMAND_READ,
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::IndexBuffer => AccessInfo {
            stage_mask: vk::PipelineStageFlags::VERTEX_INPUT,
            access_mask: vk::AccessFlags::INDEX_READ,
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::VertexBuffer => AccessInfo {
            stage_mask: vk::PipelineStageFlags::VERTEX_INPUT,
            access_mask: vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::VertexShaderReadUniformBuffer => AccessInfo {
            stage_mask: vk::PipelineStageFlags::VERTEX_SHADER,
            access_mask: vk::AccessFlags::SHADER_READ,
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::VertexShaderReadSampledImageOrUniformTexelBuffer => AccessInfo {
            stage_mask: vk::PipelineStageFlags::VERTEX_SHADER,
            access_mask: vk::AccessFlags::SHADER_READ,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        },
        AccessType::VertexShaderReadOther => AccessInfo {
            stage_mask: vk::PipelineStageFlags::VERTEX_SHADER,
            access_mask: vk::AccessFlags::SHADER_READ,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::TessellationControlShaderReadUniformBuffer => AccessInfo {
            stage_mask: vk::PipelineStageFlags::TESSELLATION_CONTROL_SHADER,
            access_mask: vk::AccessFlags::UNIFORM_READ,
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::TessellationControlShaderReadSampledImageOrUniformTexelBuffer => AccessInfo {
            stage_mask: vk::PipelineStageFlags::TESSELLATION_CONTROL_SHADER,
            access_mask: vk::AccessFlags::SHADER_READ,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        },
        AccessType::TessellationControlShaderReadOther => AccessInfo {
            stage_mask: vk::PipelineStageFlags::TESSELLATION_CONTROL_SHADER,
            access_mask: vk::AccessFlags::SHADER_READ,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::TessellationEvaluationShaderReadUniformBuffer => AccessInfo {
            stage_mask: vk::PipelineStageFlags::TESSELLATION_EVALUATION_SHADER,
            access_mask: vk::AccessFlags::UNIFORM_READ,
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::TessellationEvaluationShaderReadSampledImageOrUniformTexelBuffer => {
            AccessInfo {
                stage_mask: vk::PipelineStageFlags::TESSELLATION_EVALUATION_SHADER,
                access_mask: vk::AccessFlags::SHADER_READ,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            }
        }
        AccessType::TessellationEvaluationShaderReadOther => AccessInfo {
            stage_mask: vk::PipelineStageFlags::TESSELLATION_EVALUATION_SHADER,
            access_mask: vk::AccessFlags::SHADER_READ,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::GeometryShaderReadUniformBuffer => AccessInfo {
            stage_mask: vk::PipelineStageFlags::GEOMETRY_SHADER,
            access_mask: vk::AccessFlags::UNIFORM_READ,
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::GeometryShaderReadSampledImageOrUniformTexelBuffer => AccessInfo {
            stage_mask: vk::PipelineStageFlags::GEOMETRY_SHADER,
            access_mask: vk::AccessFlags::SHADER_READ,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        },
        AccessType::GeometryShaderReadOther => AccessInfo {
            stage_mask: vk::PipelineStageFlags::GEOMETRY_SHADER,
            access_mask: vk::AccessFlags::SHADER_READ,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::FragmentShaderReadUniformBuffer => AccessInfo {
            stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
            access_mask: vk::AccessFlags::UNIFORM_READ,
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer => AccessInfo {
            stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
            access_mask: vk::AccessFlags::SHADER_READ,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        },
        AccessType::FragmentShaderReadColorInputAttachment => AccessInfo {
            stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
            access_mask: vk::AccessFlags::INPUT_ATTACHMENT_READ,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        },
        AccessType::FragmentShaderReadDepthStencilInputAttachment => AccessInfo {
            stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
            access_mask: vk::AccessFlags::INPUT_ATTACHMENT_READ,
            image_layout: vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        },
        AccessType::FragmentShaderReadOther => AccessInfo {
            stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
            access_mask: vk::AccessFlags::SHADER_READ,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::ColorAttachmentRead => AccessInfo {
            stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ,
            image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        },
        AccessType::DepthStencilAttachmentRead => AccessInfo {
            stage_mask: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
            image_layout: vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        },
        AccessType::ComputeShaderReadUniformBuffer => AccessInfo {
            stage_mask: vk::PipelineStageFlags::COMPUTE_SHADER,
            access_mask: vk::AccessFlags::UNIFORM_READ,
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer => AccessInfo {
            stage_mask: vk::PipelineStageFlags::COMPUTE_SHADER,
            access_mask: vk::AccessFlags::SHADER_READ,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        },
        AccessType::ComputeShaderReadOther => AccessInfo {
            stage_mask: vk::PipelineStageFlags::COMPUTE_SHADER,
            access_mask: vk::AccessFlags::SHADER_READ,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::AnyShaderReadUniformBuffer => AccessInfo {
            stage_mask: vk::PipelineStageFlags::ALL_COMMANDS,
            access_mask: vk::AccessFlags::UNIFORM_READ,
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::AnyShaderReadUniformBufferOrVertexBuffer => AccessInfo {
            stage_mask: vk::PipelineStageFlags::ALL_COMMANDS,
            access_mask: vk::AccessFlags::UNIFORM_READ | vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer => AccessInfo {
            stage_mask: vk::PipelineStageFlags::ALL_COMMANDS,
            access_mask: vk::AccessFlags::SHADER_READ,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        },
        AccessType::AnyShaderReadOther => AccessInfo {
            stage_mask: vk::PipelineStageFlags::ALL_COMMANDS,
            access_mask: vk::AccessFlags::SHADER_READ,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::TransferRead => AccessInfo {
            stage_mask: vk::PipelineStageFlags::TRANSFER,
            access_mask: vk::AccessFlags::TRANSFER_READ,
            image_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        },
        AccessType::HostRead => AccessInfo {
            stage_mask: vk::PipelineStageFlags::HOST,
            access_mask: vk::AccessFlags::HOST_READ,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::Present => AccessInfo {
            stage_mask: vk::PipelineStageFlags::empty(),
            access_mask: vk::AccessFlags::empty(),
            image_layout: vk::ImageLayout::PRESENT_SRC_KHR,
        },
        AccessType::CommandBufferWriteNVX => AccessInfo {
            stage_mask: vk::PipelineStageFlags::COMMAND_PREPROCESS_NV,
            access_mask: vk::AccessFlags::COMMAND_PREPROCESS_WRITE_NV,
            image_layout: vk::ImageLayout::UNDEFINED,
        },
        AccessType::VertexShaderWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags::VERTEX_SHADER,
            access_mask: vk::AccessFlags::SHADER_WRITE,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::TessellationControlShaderWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags::TESSELLATION_CONTROL_SHADER,
            access_mask: vk::AccessFlags::SHADER_WRITE,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::TessellationEvaluationShaderWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags::TESSELLATION_EVALUATION_SHADER,
            access_mask: vk::AccessFlags::SHADER_WRITE,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::GeometryShaderWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags::GEOMETRY_SHADER,
            access_mask: vk::AccessFlags::SHADER_WRITE,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::FragmentShaderWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
            access_mask: vk::AccessFlags::SHADER_WRITE,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::ColorAttachmentWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        },
        AccessType::DepthStencilAttachmentWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            image_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        },
        AccessType::DepthAttachmentWriteStencilReadOnly => AccessInfo {
            stage_mask: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
            image_layout: vk::ImageLayout::DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL,
        },
        AccessType::StencilAttachmentWriteDepthReadOnly => AccessInfo {
            stage_mask: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
            image_layout: vk::ImageLayout::DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL,
        },
        AccessType::ComputeShaderWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags::COMPUTE_SHADER,
            access_mask: vk::AccessFlags::SHADER_WRITE,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::AnyShaderWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags::ALL_COMMANDS,
            access_mask: vk::AccessFlags::SHADER_WRITE,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::TransferWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags::TRANSFER,
            access_mask: vk::AccessFlags::TRANSFER_WRITE,
            image_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        },
        AccessType::HostWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags::HOST,
            access_mask: vk::AccessFlags::HOST_WRITE,
            image_layout: vk::ImageLayout::GENERAL,
        },
        AccessType::ColorAttachmentReadWrite => AccessInfo {
            stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
                | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        },
        AccessType::General => AccessInfo {
            stage_mask: vk::PipelineStageFlags::ALL_COMMANDS,
            access_mask: vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
            image_layout: vk::ImageLayout::GENERAL,
        },
    }
}

pub fn image_aspect_mask_from_format(format: vk::Format) -> vk::ImageAspectFlags {
    match format {
        vk::Format::D16_UNORM => vk::ImageAspectFlags::DEPTH,
        vk::Format::X8_D24_UNORM_PACK32 => vk::ImageAspectFlags::DEPTH,
        vk::Format::D32_SFLOAT => vk::ImageAspectFlags::DEPTH,
        vk::Format::S8_UINT => vk::ImageAspectFlags::STENCIL,
        vk::Format::D16_UNORM_S8_UINT => {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        }
        vk::Format::D24_UNORM_S8_UINT => {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        }
        vk::Format::D32_SFLOAT_S8_UINT => {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        }
        _ => vk::ImageAspectFlags::COLOR,
    }
}

// TODO: is access type relevant here at all?
pub fn image_aspect_mask_from_access_type_and_format(
    access_type: AccessType,
    format: vk::Format,
) -> Option<vk::ImageAspectFlags> {
    let image_layout = get_access_info(access_type).image_layout;

    match image_layout {
        vk::ImageLayout::GENERAL
        | vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        | vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        | vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
        | vk::ImageLayout::DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL
        | vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        | vk::ImageLayout::TRANSFER_SRC_OPTIMAL
        | vk::ImageLayout::TRANSFER_DST_OPTIMAL => Some(image_aspect_mask_from_format(format)),
        _ => {
            //println!("{:?}", image_layout);
            None
        }
    }

    /*let info = get_access_info(access_type);

    match info.image_layout {
        vk::ImageLayout::GENERAL => Some(image_aspect_mask_from_format(format)),
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => Some(image_aspect_mask_from_format(format)),
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL => Some(
            image_aspect_mask_from_format(format)
                & (vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL),
        ),
        vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL => Some(
            image_aspect_mask_from_format(format)
                & (vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL),
        ),
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => Some(image_aspect_mask_from_format(format)),
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL => Some(image_aspect_mask_from_format(format)),
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => Some(image_aspect_mask_from_format(format)),
        _ => None,
    }*/
}
