use thiserror::Error;

#[derive(Error, Debug)]
pub enum VkError {
    #[error("unknown data store error")]
    Generic(ash::vk::Result),
}
