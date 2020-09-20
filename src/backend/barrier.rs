use ash::{version::DeviceV1_0, vk};
use vk_sync::AccessType;

pub struct ImageBarrier {
    image: vk::Image,
    prev_access: vk_sync::AccessType,
    next_access: vk_sync::AccessType,
    aspect_mask: vk::ImageAspectFlags,
    discard: bool,
}

pub fn record_image_barrier(device: &ash::Device, cb: vk::CommandBuffer, barrier: ImageBarrier) {
    let range = vk::ImageSubresourceRange {
        aspect_mask: barrier.aspect_mask,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };

    vk_sync::cmd::pipeline_barrier(
        device.fp_v1_0(),
        cb,
        None,
        &[],
        &[vk_sync::ImageBarrier {
            previous_accesses: &[barrier.prev_access],
            next_accesses: &[barrier.next_access],
            previous_layout: vk_sync::ImageLayout::Optimal,
            next_layout: vk_sync::ImageLayout::Optimal,
            discard_contents: barrier.discard,
            src_queue_family_index: 0,
            dst_queue_family_index: 0,
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
#[allow(dead_code)]
pub(crate) struct AccessInfo {
    pub(crate) stage_mask: ash::vk::PipelineStageFlags,
    pub(crate) access_mask: ash::vk::AccessFlags,
    pub(crate) image_layout: ash::vk::ImageLayout,
}

pub(crate) fn get_access_info(access_type: AccessType) -> AccessInfo {
    match access_type {
        AccessType::Nothing => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::empty(),
            access_mask: ash::vk::AccessFlags::empty(),
            image_layout: ash::vk::ImageLayout::UNDEFINED,
        },
        AccessType::CommandBufferReadNVX => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::COMMAND_PREPROCESS_NV,
            access_mask: ash::vk::AccessFlags::COMMAND_PREPROCESS_READ_NV,
            image_layout: ash::vk::ImageLayout::UNDEFINED,
        },
        AccessType::IndirectBuffer => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::DRAW_INDIRECT,
            access_mask: ash::vk::AccessFlags::INDIRECT_COMMAND_READ,
            image_layout: ash::vk::ImageLayout::UNDEFINED,
        },
        AccessType::IndexBuffer => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::VERTEX_INPUT,
            access_mask: ash::vk::AccessFlags::INDEX_READ,
            image_layout: ash::vk::ImageLayout::UNDEFINED,
        },
        AccessType::VertexBuffer => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::VERTEX_INPUT,
            access_mask: ash::vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
            image_layout: ash::vk::ImageLayout::UNDEFINED,
        },
        AccessType::VertexShaderReadUniformBuffer => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::VERTEX_SHADER,
            access_mask: ash::vk::AccessFlags::SHADER_READ,
            image_layout: ash::vk::ImageLayout::UNDEFINED,
        },
        AccessType::VertexShaderReadSampledImageOrUniformTexelBuffer => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::VERTEX_SHADER,
            access_mask: ash::vk::AccessFlags::SHADER_READ,
            image_layout: ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        },
        AccessType::VertexShaderReadOther => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::VERTEX_SHADER,
            access_mask: ash::vk::AccessFlags::SHADER_READ,
            image_layout: ash::vk::ImageLayout::GENERAL,
        },
        AccessType::TessellationControlShaderReadUniformBuffer => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::TESSELLATION_CONTROL_SHADER,
            access_mask: ash::vk::AccessFlags::UNIFORM_READ,
            image_layout: ash::vk::ImageLayout::UNDEFINED,
        },
        AccessType::TessellationControlShaderReadSampledImageOrUniformTexelBuffer => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::TESSELLATION_CONTROL_SHADER,
            access_mask: ash::vk::AccessFlags::SHADER_READ,
            image_layout: ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        },
        AccessType::TessellationControlShaderReadOther => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::TESSELLATION_CONTROL_SHADER,
            access_mask: ash::vk::AccessFlags::SHADER_READ,
            image_layout: ash::vk::ImageLayout::GENERAL,
        },
        AccessType::TessellationEvaluationShaderReadUniformBuffer => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::TESSELLATION_EVALUATION_SHADER,
            access_mask: ash::vk::AccessFlags::UNIFORM_READ,
            image_layout: ash::vk::ImageLayout::UNDEFINED,
        },
        AccessType::TessellationEvaluationShaderReadSampledImageOrUniformTexelBuffer => {
            AccessInfo {
                stage_mask: ash::vk::PipelineStageFlags::TESSELLATION_EVALUATION_SHADER,
                access_mask: ash::vk::AccessFlags::SHADER_READ,
                image_layout: ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            }
        }
        AccessType::TessellationEvaluationShaderReadOther => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::TESSELLATION_EVALUATION_SHADER,
            access_mask: ash::vk::AccessFlags::SHADER_READ,
            image_layout: ash::vk::ImageLayout::GENERAL,
        },
        AccessType::GeometryShaderReadUniformBuffer => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::GEOMETRY_SHADER,
            access_mask: ash::vk::AccessFlags::UNIFORM_READ,
            image_layout: ash::vk::ImageLayout::UNDEFINED,
        },
        AccessType::GeometryShaderReadSampledImageOrUniformTexelBuffer => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::GEOMETRY_SHADER,
            access_mask: ash::vk::AccessFlags::SHADER_READ,
            image_layout: ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        },
        AccessType::GeometryShaderReadOther => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::GEOMETRY_SHADER,
            access_mask: ash::vk::AccessFlags::SHADER_READ,
            image_layout: ash::vk::ImageLayout::GENERAL,
        },
        AccessType::FragmentShaderReadUniformBuffer => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::FRAGMENT_SHADER,
            access_mask: ash::vk::AccessFlags::UNIFORM_READ,
            image_layout: ash::vk::ImageLayout::UNDEFINED,
        },
        AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::FRAGMENT_SHADER,
            access_mask: ash::vk::AccessFlags::SHADER_READ,
            image_layout: ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        },
        AccessType::FragmentShaderReadColorInputAttachment => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::FRAGMENT_SHADER,
            access_mask: ash::vk::AccessFlags::INPUT_ATTACHMENT_READ,
            image_layout: ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        },
        AccessType::FragmentShaderReadDepthStencilInputAttachment => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::FRAGMENT_SHADER,
            access_mask: ash::vk::AccessFlags::INPUT_ATTACHMENT_READ,
            image_layout: ash::vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        },
        AccessType::FragmentShaderReadOther => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::FRAGMENT_SHADER,
            access_mask: ash::vk::AccessFlags::SHADER_READ,
            image_layout: ash::vk::ImageLayout::GENERAL,
        },
        AccessType::ColorAttachmentRead => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            access_mask: ash::vk::AccessFlags::COLOR_ATTACHMENT_READ,
            image_layout: ash::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        },
        AccessType::DepthStencilAttachmentRead => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | ash::vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            access_mask: ash::vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
            image_layout: ash::vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        },
        AccessType::ComputeShaderReadUniformBuffer => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::COMPUTE_SHADER,
            access_mask: ash::vk::AccessFlags::UNIFORM_READ,
            image_layout: ash::vk::ImageLayout::UNDEFINED,
        },
        AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::COMPUTE_SHADER,
            access_mask: ash::vk::AccessFlags::SHADER_READ,
            image_layout: ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        },
        AccessType::ComputeShaderReadOther => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::COMPUTE_SHADER,
            access_mask: ash::vk::AccessFlags::SHADER_READ,
            image_layout: ash::vk::ImageLayout::GENERAL,
        },
        AccessType::AnyShaderReadUniformBuffer => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::ALL_COMMANDS,
            access_mask: ash::vk::AccessFlags::UNIFORM_READ,
            image_layout: ash::vk::ImageLayout::UNDEFINED,
        },
        AccessType::AnyShaderReadUniformBufferOrVertexBuffer => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::ALL_COMMANDS,
            access_mask: ash::vk::AccessFlags::UNIFORM_READ
                | ash::vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
            image_layout: ash::vk::ImageLayout::UNDEFINED,
        },
        AccessType::AnyShaderReadSampledImageOrUniformTexelBuffer => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::ALL_COMMANDS,
            access_mask: ash::vk::AccessFlags::SHADER_READ,
            image_layout: ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        },
        AccessType::AnyShaderReadOther => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::ALL_COMMANDS,
            access_mask: ash::vk::AccessFlags::SHADER_READ,
            image_layout: ash::vk::ImageLayout::GENERAL,
        },
        AccessType::TransferRead => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::TRANSFER,
            access_mask: ash::vk::AccessFlags::TRANSFER_READ,
            image_layout: ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        },
        AccessType::HostRead => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::HOST,
            access_mask: ash::vk::AccessFlags::HOST_READ,
            image_layout: ash::vk::ImageLayout::GENERAL,
        },
        AccessType::Present => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::empty(),
            access_mask: ash::vk::AccessFlags::empty(),
            image_layout: ash::vk::ImageLayout::PRESENT_SRC_KHR,
        },
        AccessType::CommandBufferWriteNVX => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::COMMAND_PREPROCESS_NV,
            access_mask: ash::vk::AccessFlags::COMMAND_PREPROCESS_WRITE_NV,
            image_layout: ash::vk::ImageLayout::UNDEFINED,
        },
        AccessType::VertexShaderWrite => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::VERTEX_SHADER,
            access_mask: ash::vk::AccessFlags::SHADER_WRITE,
            image_layout: ash::vk::ImageLayout::GENERAL,
        },
        AccessType::TessellationControlShaderWrite => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::TESSELLATION_CONTROL_SHADER,
            access_mask: ash::vk::AccessFlags::SHADER_WRITE,
            image_layout: ash::vk::ImageLayout::GENERAL,
        },
        AccessType::TessellationEvaluationShaderWrite => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::TESSELLATION_EVALUATION_SHADER,
            access_mask: ash::vk::AccessFlags::SHADER_WRITE,
            image_layout: ash::vk::ImageLayout::GENERAL,
        },
        AccessType::GeometryShaderWrite => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::GEOMETRY_SHADER,
            access_mask: ash::vk::AccessFlags::SHADER_WRITE,
            image_layout: ash::vk::ImageLayout::GENERAL,
        },
        AccessType::FragmentShaderWrite => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::FRAGMENT_SHADER,
            access_mask: ash::vk::AccessFlags::SHADER_WRITE,
            image_layout: ash::vk::ImageLayout::GENERAL,
        },
        AccessType::ColorAttachmentWrite => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            access_mask: ash::vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            image_layout: ash::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        },
        AccessType::DepthStencilAttachmentWrite => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | ash::vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            access_mask: ash::vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            image_layout: ash::vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        },
        AccessType::DepthAttachmentWriteStencilReadOnly => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | ash::vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            access_mask: ash::vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
                | ash::vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
            image_layout: ash::vk::ImageLayout::DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL,
        },
        AccessType::StencilAttachmentWriteDepthReadOnly => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | ash::vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            access_mask: ash::vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
                | ash::vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
            image_layout: ash::vk::ImageLayout::DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL,
        },
        AccessType::ComputeShaderWrite => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::COMPUTE_SHADER,
            access_mask: ash::vk::AccessFlags::SHADER_WRITE,
            image_layout: ash::vk::ImageLayout::GENERAL,
        },
        AccessType::AnyShaderWrite => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::ALL_COMMANDS,
            access_mask: ash::vk::AccessFlags::SHADER_WRITE,
            image_layout: ash::vk::ImageLayout::GENERAL,
        },
        AccessType::TransferWrite => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::TRANSFER,
            access_mask: ash::vk::AccessFlags::TRANSFER_WRITE,
            image_layout: ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        },
        AccessType::HostWrite => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::HOST,
            access_mask: ash::vk::AccessFlags::HOST_WRITE,
            image_layout: ash::vk::ImageLayout::GENERAL,
        },
        AccessType::ColorAttachmentReadWrite => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            access_mask: ash::vk::AccessFlags::COLOR_ATTACHMENT_READ
                | ash::vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            image_layout: ash::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        },
        AccessType::General => AccessInfo {
            stage_mask: ash::vk::PipelineStageFlags::ALL_COMMANDS,
            access_mask: ash::vk::AccessFlags::MEMORY_READ | ash::vk::AccessFlags::MEMORY_WRITE,
            image_layout: ash::vk::ImageLayout::GENERAL,
        },
    }
}
