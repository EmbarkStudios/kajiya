#![allow(dead_code)]

use parking_lot::Mutex;
use std::{
    cell::UnsafeCell,
    mem::MaybeUninit,
    sync::atomic::{AtomicI64, Ordering},
};

const PENDING_RESOURE_BLOCK_SIZE: usize = 64;

#[derive(Clone, Copy, Debug)]
pub struct ResourceHandle {
    index: u32,
    version: u16,
}

impl ResourceHandle {
    pub fn index(self) -> u32 {
        self.index
    }
}

pub(crate) struct ResourceStorage<T: Sized> {
    versions: Vec<u16>,
    current: Vec<UnsafeCell<MaybeUninit<T>>>,
    pending: Mutex<Vec<UnsafeCell<Vec<T>>>>,
    free_index_buffer: Vec<u32>,
    free_index_count: AtomicI64,
    pending_removal: Mutex<Vec<ResourceHandle>>,
}

impl<T: Sized> Default for ResourceStorage<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Sized> ResourceStorage<T> {
    pub fn new() -> Self {
        Self {
            versions: Default::default(),
            current: Default::default(),
            pending: Default::default(),
            free_index_buffer: Default::default(),
            free_index_count: AtomicI64::new(0),
            pending_removal: Default::default(),
        }
    }

    pub fn insert(&self, item: T) -> ResourceHandle {
        let free_idx_addr = self.free_index_count.fetch_sub(1, Ordering::Relaxed) - 1;
        if free_idx_addr >= 0 {
            unsafe {
                let index = *self.free_index_buffer.get_unchecked(free_idx_addr as usize) as usize;
                (*self.current[index].get()).as_mut_ptr().write(item);

                ResourceHandle {
                    index: index as _,
                    version: self.versions[index],
                }
            }
        } else {
            let mut pending = self.pending.lock();

            let block_count = pending.len();
            if block_count > 0 {
                let block_index = block_count - 1;
                unsafe {
                    let last = pending.get_unchecked(block_index);
                    let last = &mut *last.get();

                    let idx_within_block = last.len();
                    if idx_within_block < last.capacity() {
                        last.push(item);

                        return ResourceHandle {
                            index: (block_index * PENDING_RESOURE_BLOCK_SIZE + idx_within_block)
                                as _,
                            version: 0,
                        };
                    }
                }
            }

            let block_index = block_count;
            let idx_within_block = 0;

            let mut new_block = Vec::with_capacity(PENDING_RESOURE_BLOCK_SIZE);
            new_block.push(item);
            pending.push(UnsafeCell::new(new_block));

            return ResourceHandle {
                index: (block_index * PENDING_RESOURE_BLOCK_SIZE + idx_within_block) as _,
                version: 0,
            };
        }
    }

    pub fn get(&self, handle: ResourceHandle) -> &T {
        let index = handle.index as usize;
        if index < self.current.len() {
            unsafe {
                if *self.versions.get_unchecked(index) == handle.version {
                    &*(*self.current.get_unchecked(index).get()).as_ptr()
                } else {
                    panic!("Invalid resource handle: stale version");
                }
            }
        } else {
            assert!(
                handle.version == 0,
                "Invalid resource handle: version from the future?"
            );

            let pending = self.pending.lock();
            let block = index / PENDING_RESOURE_BLOCK_SIZE;
            let element = index % PENDING_RESOURE_BLOCK_SIZE;

            unsafe { &(&*(*pending)[block].get())[element] }
        }
    }

    pub fn contains(&self, handle: ResourceHandle) -> bool {
        let index = handle.index as usize;
        if index < self.current.len() {
            unsafe { *self.versions.get_unchecked(index) == handle.version }
        } else {
            if handle.version != 0 {
                false
            } else {
                let pending = self.pending.lock();
                let block = index / PENDING_RESOURE_BLOCK_SIZE;
                let element = index % PENDING_RESOURE_BLOCK_SIZE;

                if let Some(block) = pending.get(block) {
                    unsafe { element < (*block.get()).len() }
                } else {
                    false
                }
            }
        }
    }

    pub fn remove(&self, handle: ResourceHandle) {
        // Confirm the item exists
        let _ = self.get(handle);
        self.pending_removal.lock().push(handle);
    }

    pub fn maintain(&mut self) {
        let mut pending = self.pending.lock();
        let pending = &mut *pending;

        for block in pending.drain(..) {
            let block = block.into_inner();
            for item in block.into_iter() {
                self.current.push(UnsafeCell::new(MaybeUninit::new(item)));
                self.versions.push(0);
            }
        }

        self.free_index_buffer
            .truncate(self.free_index_count.load(Ordering::Relaxed).max(0) as usize);

        let mut pending_removal = self.pending_removal.lock();
        for to_remove in pending_removal.drain(..) {
            let index = to_remove.index as usize;

            // Check the version in order to avoid double removes
            if self.versions[index] == to_remove.version {
                self.versions[index] += 1;
                unsafe {
                    // Drop the item
                    std::ptr::drop_in_place((&mut *self.current[index].get()).as_mut_ptr());
                }

                self.free_index_buffer.push(index as _);
            }
        }

        self.free_index_count
            .store(self.free_index_buffer.len() as i64, Ordering::Relaxed);
    }
}

impl<T: Sized> Drop for ResourceStorage<T> {
    fn drop(&mut self) {
        self.maintain();

        // Mark all items with version 0
        for ver in self.versions.iter_mut() {
            *ver = 0;
        }

        // Mark dead items with version 1
        for idx in self.free_index_buffer.drain(..) {
            self.versions[idx as usize] = 1;
        }

        // All items which are still alive have version 0 now. Let's drop them now.
        for (ver, item) in self.versions.iter().zip(self.current.iter()) {
            if 0 == *ver {
                unsafe {
                    std::ptr::drop_in_place((&mut *item.get()).as_mut_ptr());
                }
            }
        }
    }
}

#[test]
fn basics() {
    #[derive(PartialEq, Eq)]
    struct Item {
        val: u32,
    }

    /*impl Drop for Item {
        fn drop(&mut self) {
            println!("Dropping {}", self.val);
        }
    }*/

    let mut storage = ResourceStorage::<Item>::new();

    let h0 = storage.insert(Item { val: 0 });
    let h1 = storage.insert(Item { val: 1 });
    let h2 = storage.insert(Item { val: 2 });

    assert!(storage.contains(h0));
    assert!(storage.contains(h1));
    assert!(storage.contains(h2));
    assert!(storage.get(h0).val == 0);
    assert!(storage.get(h1).val == 1);
    assert!(storage.get(h2).val == 2);
    storage.remove(h0);
    storage.maintain();
    assert!(!storage.contains(h0));
    assert!(storage.contains(h1));
    assert!(storage.contains(h2));
    assert!(storage.get(h1).val == 1);
    assert!(storage.get(h2).val == 2);

    storage.remove(h1);

    storage.maintain();
    assert!(!storage.contains(h1));

    let h3 = storage.insert(Item { val: 3 });
    assert!(h3.index <= 2);
}
