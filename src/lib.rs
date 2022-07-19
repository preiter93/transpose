//! Fast Transposes of flattened Arrays
//!
//! # ToDo
//!
//! - Parallel Transposes
pub mod ej;
pub use ej::transpose_recursive;
pub use ej::transpose_tiled;
pub mod inplace;
pub mod outofplace;
pub use inplace::ip_transpose;
pub use outofplace::oop_transpose;
