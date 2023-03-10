use std::cell::UnsafeCell;

struct TempListInner<T> {
    payload: arrayvec::ArrayVec<[T; 8]>,
    next: Option<TempList<T>>,
}

impl<T> Default for TempListInner<T> {
    fn default() -> Self {
        Self {
            payload: Default::default(),
            next: None,
        }
    }
}

pub struct TempList<T>(UnsafeCell<Box<TempListInner<T>>>);

impl<T> Default for TempList<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> TempList<T> {
    pub fn new() -> Self {
        Self(UnsafeCell::new(Box::default()))
    }

    pub fn add(&self, item: T) -> &T {
        unsafe {
            let inner = &mut *self.0.get();
            if let Err(err) = inner.payload.try_push(item) {
                let mut new_payload = arrayvec::ArrayVec::new();
                new_payload.push(err.element());

                let mut new_node = Box::new(TempListInner {
                    payload: new_payload,
                    next: None,
                });

                std::mem::swap(&mut new_node, inner);
                inner.next = Some(TempList(UnsafeCell::new(new_node)));
                &inner.payload[0]
            } else {
                &inner.payload[inner.payload.len() - 1]
            }
        }
    }
}

#[test]
fn test_add() {
    let list = TempList::new();
    let mut refs: Vec<&u32> = Vec::new();
    const ITEM_COUNT: u32 = 1024;

    for i in 0..ITEM_COUNT {
        refs.push(list.add(i))
    }

    for i in 0..ITEM_COUNT {
        assert_eq!(i, *refs[i as usize]);
    }
}
