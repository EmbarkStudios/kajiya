use std::{borrow::BorrowMut, sync::Arc};

use parking_lot::RwLock;

type ReelCellInner<T> = Arc<RwLock<Option<T>>>;

pub struct ReelCell<T> {
    inner: ReelCellInner<T>,
}

impl<T> ReelCell<T> {
    pub fn new(value: T) -> Self {
        Self {
            inner: Arc::new(RwLock::new(Some(value))),
        }
    }

    pub fn try_take(&self) -> Option<ReelRef<T>> {
        let reel = self.inner.clone();
        let maybe_value = reel.write().take();
        maybe_value.map(|value| ReelRef {
            value: Some(value),
            reel,
        })
    }
}

pub struct ReelRef<T> {
    value: Option<T>,
    reel: ReelCellInner<T>,
}

impl<T> Drop for ReelRef<T> {
    fn drop(&mut self) {
        *self.reel.write() = Some(self.value.take().unwrap());
    }
}

impl<T> std::ops::Deref for ReelRef<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.value.as_ref().unwrap()
    }
}

impl<T> std::ops::DerefMut for ReelRef<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.value.as_mut().unwrap()
    }
}

#[test]
fn test() {
    let reel = ReelCell::new(123u32);

    {
        let foo = reel.try_take().unwrap();
        assert_eq!(*foo, 123u32);
        assert!(reel.try_take().is_none());
        drop(foo);
    }

    assert!(reel.try_take().is_some());
}
