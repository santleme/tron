//! Small, dependency-free native territory kernels.
//!
//! Bitboards cross the FFI boundary as little-endian u64 limbs. The fused
//! evaluator mirrors `training.search.territory.evaluate` for live states.

use std::slice;

const EMPTY: usize = usize::MAX;
const MAX_CELLS: usize = 4096;

fn articulation_count(
    reachable: &[bool],
    width: usize,
    height: usize,
    start: usize,
    other: usize,
) -> u32 {
    let size = width * height;
    let mut reachable = reachable.to_vec();
    reachable[other] = false;
    if !reachable[start] {
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
            if !reachable[nxt] {
                continue;
            }
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
    articulation.iter().filter(|&&point| point).count() as u32
}

fn flood(occupied: &[bool], width: usize, height: usize, start: usize) -> Vec<bool> {
    let size = width * height;
    let mut seen = vec![false; size];
    if start >= size {
        return seen;
    }
    let mut queue = Vec::with_capacity(size);
    seen[start] = true;
    queue.push(start);
    let mut cursor = 0;
    while cursor < queue.len() {
        let cell = queue[cursor];
        cursor += 1;
        let x = cell % width;
        let y = cell / width;
        let neighbors = [
            if y > 0 { Some(cell - width) } else { None },
            if x + 1 < width { Some(cell + 1) } else { None },
            if y + 1 < height {
                Some(cell + width)
            } else {
                None
            },
            if x > 0 { Some(cell - 1) } else { None },
        ];
        for nxt in neighbors.into_iter().flatten() {
            if !seen[nxt] && !occupied[nxt] {
                seen[nxt] = true;
                queue.push(nxt);
            }
        }
    }
    seen
}

fn distances(
    occupied: &[bool],
    width: usize,
    height: usize,
    start: usize,
    other: usize,
) -> Vec<i32> {
    let size = width * height;
    let mut distance = vec![-1i32; size];
    let mut queue = Vec::with_capacity(size);
    let mut blocked = occupied.to_vec();
    blocked[start] = false;
    blocked[other] = true;
    distance[start] = 0;
    queue.push(start);
    let mut cursor = 0;
    while cursor < queue.len() {
        let cell = queue[cursor];
        cursor += 1;
        let x = cell % width;
        let y = cell / width;
        let neighbors = [
            if y > 0 { Some(cell - width) } else { None },
            if x + 1 < width { Some(cell + 1) } else { None },
            if y + 1 < height {
                Some(cell + width)
            } else {
                None
            },
            if x > 0 { Some(cell - 1) } else { None },
        ];
        for nxt in neighbors.into_iter().flatten() {
            if distance[nxt] < 0 && !blocked[nxt] {
                distance[nxt] = distance[cell] + 1;
                queue.push(nxt);
            }
        }
    }
    distance
}

fn legal_mobility(
    occupied: &[bool],
    width: usize,
    height: usize,
    head: usize,
    heading: usize,
) -> i32 {
    let x = head % width;
    let y = head / width;
    let destinations = [
        if y > 0 { Some(head - width) } else { None },
        if x + 1 < width { Some(head + 1) } else { None },
        if y + 1 < height {
            Some(head + width)
        } else {
            None
        },
        if x > 0 { Some(head - 1) } else { None },
    ];
    destinations
        .iter()
        .enumerate()
        .filter(|(action, dest)| {
            *action != (heading + 2) % 4 && dest.map_or(false, |cell| !occupied[cell])
        })
        .count() as i32
}

fn evaluate(
    occupied: &[bool],
    width: usize,
    height: usize,
    heads: [usize; 2],
    headings: [usize; 2],
) -> f64 {
    let regions = [
        flood(occupied, width, height, heads[0]),
        flood(occupied, width, height, heads[1]),
    ];
    let sizes = [
        regions[0].iter().filter(|&&cell| cell).count() as f64,
        regions[1].iter().filter(|&&cell| cell).count() as f64,
    ];
    let connected = regions[0].iter().zip(&regions[1]).any(|(&a, &b)| a && b);
    if !connected {
        let delta = (sizes[0] - sizes[1]) / (width * height).max(1) as f64;
        return (delta * 2.0).clamp(-0.95, 0.95);
    }

    let d0 = distances(occupied, width, height, heads[0], heads[1]);
    let d1 = distances(occupied, width, height, heads[1], heads[0]);
    let mut owned = [0i64; 2];
    for cell in 0..width * height {
        let playable = !occupied[cell] || cell == heads[0] || cell == heads[1];
        if !playable {
            continue;
        }
        let (a, b) = (d0[cell], d1[cell]);
        if (a < 0 && b < 0) || a == b {
            continue;
        } else if b < 0 || (a >= 0 && a < b) {
            owned[0] += 1;
        } else {
            owned[1] += 1;
        }
    }
    let mobility = legal_mobility(occupied, width, height, heads[0], headings[0])
        - legal_mobility(occupied, width, height, heads[1], headings[1]);
    let articulation = articulation_count(&regions[0], width, height, heads[0], heads[1]) as i32
        - articulation_count(&regions[1], width, height, heads[1], heads[0]) as i32;
    let scale = (width * height).max(1) as f64;
    ((owned[0] - owned[1]) as f64 / scale + 0.035 * mobility as f64 + 0.006 * articulation as f64)
        .clamp(-0.95, 0.95)
}

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
    let Some(size) = width.checked_mul(height) else {
        return 1;
    };
    let needed = size.saturating_add(63) / 64;
    if size > MAX_CELLS || word_count != needed || start >= size || other >= size {
        return 1;
    }
    let packed = slice::from_raw_parts(words, word_count);
    let mut reachable = vec![false; size];
    for cell in 0..size {
        reachable[cell] = (packed[cell / 64] & (1u64 << (cell % 64))) != 0;
    }
    *result = articulation_count(&reachable, width, height, start, other);
    0
}

/// Return territory value for player 0, or NaN for malformed arguments.
#[no_mangle]
pub unsafe extern "C" fn circuit_evaluate_value(
    words: *const u64,
    word_count: usize,
    width: usize,
    height: usize,
    head0: usize,
    head1: usize,
    heading0: usize,
    heading1: usize,
) -> f64 {
    if words.is_null() || width < 2 || height < 2 {
        return f64::NAN;
    }
    let Some(size) = width.checked_mul(height) else {
        return f64::NAN;
    };
    let needed = size.saturating_add(63) / 64;
    if size > MAX_CELLS
        || word_count != needed
        || head0 >= size
        || head1 >= size
        || head0 == head1
        || heading0 >= 4
        || heading1 >= 4
    {
        return f64::NAN;
    }
    let packed = slice::from_raw_parts(words, word_count);
    let mut occupied = vec![false; size];
    for cell in 0..size {
        occupied[cell] = (packed[cell / 64] & (1u64 << (cell % 64))) != 0;
    }
    if !occupied[head0] || !occupied[head1] {
        return f64::NAN;
    }
    evaluate(
        &occupied,
        width,
        height,
        [head0, head1],
        [heading0, heading1],
    )
}
