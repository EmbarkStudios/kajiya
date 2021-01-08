#[allow(dead_code)]
pub fn into_byte_vec<T>(mut v: Vec<T>) -> Vec<u8>
where
    T: Copy,
{
    unsafe {
        let p = v.as_mut_ptr();
        let item_sizeof = std::mem::size_of::<T>();
        let len = v.len() * item_sizeof;
        let cap = v.capacity() * item_sizeof;
        std::mem::forget(v);
        Vec::from_raw_parts(p as *mut u8, len, cap)
    }
}

#[allow(dead_code)]
pub fn as_byte_slice<'a, T>(t: &'a T) -> &'a [u8]
where
    T: Copy,
{
    unsafe { std::slice::from_raw_parts(t as *const T as *mut u8, std::mem::size_of::<T>()) }
}
