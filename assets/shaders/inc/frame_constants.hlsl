struct FrameConstants {
    uint frame_index;
};

[[vk::binding(0, 2)]] ConstantBuffer<FrameConstants> frame_constants;
