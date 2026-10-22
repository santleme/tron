//! Small, dependency-free native kernel for counting articulation vertices.
//!
//! The caller passes the already-computed reachable set as little-endian u64
//! limbs. The other head is removed to match `chamber_tree` semantics.

use std::slice;

const EMPTY: usize = usize::MAX;

#[no_mangle]
pub unsafe extern "C" fn circuit_articulation_count(
    words: *const u64,
    word_count: usize,
    width: usize,
    height: usize,
    start: usize,
    other: usize,
    result: *mut u32,
) -> i32 {
    if words.is_null() || result.is_null() || width == 0 || height == 0 {
        return 1;
    }
    let Some(size) = width.checked_mul(height) else { return 1 };
    let needed = size.saturating_add(63) / 64;
    if word_count != needed || start >= size || other >= size {
        return 1;
    }
    let packed = slice::from_raw_parts(words, word_count);
    let mut reachable = vec![false; size];
    for cell in 0..size {
        reachable[cell] = (packed[cell / 64] & (1u64 << (cell % 64))) != 0;
    }
    reachable[other] = false;
    if !reachable[start] {
        *result = 0;
        return 0;
    }

    let mut discovery = vec![0usize; size];
    let mut low = vec![0usize; size];
    let mut parent = vec![EMPTY; size];
    let mut children = vec![0usize; size];
    let mut articulation = vec![false; size];
    let mut clock = 1usize;
    discovery[start] = clock;
    low[start] = clock;
    // Frame stores the next U,R,D,L neighbor index, matching Python ordering.
    let mut stack = vec![(start, 0usize)];

    while let Some((cell, next_dir)) = stack.pop() {
        if next_dir < 4 {
            stack.push((cell, next_dir + 1));
            let x = cell % width;
            let y = cell / width;
            let neighbor = match next_dir {
                0 if y > 0 => Some(cell - width),
                1 if x + 1 < width => Some(cell + 1),
                2 if y + 1 < height => Some(cell + width),
                3 if x > 0 => Some(cell - 1),
                _ => None,
            };
            let Some(nxt) = neighbor else { continue };
            if !reachable[nxt] { continue; }
            if discovery[nxt] == 0 {
                parent[nxt] = cell;
                children[cell] += 1;
                clock += 1;
                discovery[nxt] = clock;
                low[nxt] = clock;
                stack.push((nxt, 0));
            } else if nxt != parent[cell] && discovery[nxt] < discovery[cell] {
                low[cell] = low[cell].min(discovery[nxt]);
            }
        } else {
            let ancestor = parent[cell];
            if ancestor == EMPTY {
                articulation[cell] = children[cell] > 1;
            } else {
                low[ancestor] = low[ancestor].min(low[cell]);
                if low[cell] >= discovery[ancestor] && parent[ancestor] != EMPTY {
                    articulation[ancestor] = true;
                }
            }
        }
    }
    *result = articulation.iter().filter(|&&point| point).count() as u32;
    0
}
