//! Small allocation-bounded fixed-depth simultaneous Tron planner for WASM.
//!
//! Input is a compact packet written by JS into `circuit_input_ptr()`:
//! [width, height, head0, head1, heading0, heading1, result, alive_bits,
//!  max_depth, 0, edge_budget:u32 LE, occupied:32 bytes].
//! Result codes: 0 ongoing, 1 player 0 win, 2 player 1 win, 3 draw.
//! Output f64 slots at `circuit_output_ptr()`:
//! value, policy0[4], policy1[4], legal_mask0, legal_mask1, selected_action,
//! completed_depth, transition_edges, leaf_evaluations.

use std::collections::HashMap;

const MAX_CELLS: usize = 256;
const WORDS: usize = MAX_CELLS / 64;
const INPUT_BYTES: usize = 14 + WORDS * 8;
const DELTAS: [(i32, i32); 4] = [(0, -1), (1, 0), (0, 1), (-1, 0)];

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct State {
    width: u8,
    height: u8,
    heads: [u16; 2],
    headings: [u8; 2],
    occupied: [u64; WORDS],
    alive: [bool; 2],
    result: u8,
}

impl State {
    fn size(self) -> usize { self.width as usize * self.height as usize }
    fn has(self, cell: usize) -> bool {
        self.occupied[cell / 64] & (1u64 << (cell % 64)) != 0
    }
    fn set(&mut self, cell: usize) { self.occupied[cell / 64] |= 1u64 << (cell % 64); }
}

#[derive(Clone, Copy)]
struct SearchOutput {
    value: f64,
    policy: [[f64; 4]; 2],
    masks: [u8; 2],
    action: i32,
    depth: usize,
    edges: usize,
    leaves: usize,
}

fn legal_mask(s: &State, player: usize) -> u8 {
    if s.result != 0 || !s.alive[player] { return 0; }
    let mut mask = 0;
    for action in 0..4 {
        if action == (s.headings[player] as usize + 2) % 4 { continue; }
        let cell = s.heads[player] as usize;
        let x = cell % s.width as usize;
        let y = cell / s.width as usize;
        let nx = x as i32 + DELTAS[action].0;
        let ny = y as i32 + DELTAS[action].1;
        if nx < 0 || ny < 0 || nx >= s.width as i32 || ny >= s.height as i32 { continue; }
        if !s.has(ny as usize * s.width as usize + nx as usize) { mask |= 1 << action; }
    }
    mask
}

fn action_list(s: &State, player: usize) -> Vec<i8> {
    let mask = legal_mask(s, player);
    let mut actions: Vec<i8> = (0..4).filter(|a| mask & (1 << a) != 0).map(|a| a as i8).collect();
    if actions.is_empty() { actions.push(-1); }
    actions
}

fn destination(s: &State, player: usize, action: i8) -> Option<usize> {
    if action < 0 || action > 3 { return None; }
    let cell = s.heads[player] as usize;
    let nx = (cell % s.width as usize) as i32 + DELTAS[action as usize].0;
    let ny = (cell / s.width as usize) as i32 + DELTAS[action as usize].1;
    if nx < 0 || ny < 0 || nx >= s.width as i32 || ny >= s.height as i32 { return None; }
    Some(ny as usize * s.width as usize + nx as usize)
}

fn transition(s: State, actions: [i8; 2]) -> State {
    if s.result != 0 { return s; }
    let mut dead = [false; 2];
    let mut dest = [None, None];
    for p in 0..2 {
        let a = actions[p];
        if !s.alive[p] || a < 0 || a > 3 { dead[p] = true; continue; }
        if a as usize == (s.headings[p] as usize + 2) % 4 { dead[p] = true; continue; }
        dest[p] = destination(&s, p, a);
        match dest[p] {
            None => dead[p] = true,
            Some(cell) if s.has(cell) => dead[p] = true,
            _ => (),
        }
    }
    if dest[0].is_some() && dest[0] == dest[1] { dead = [true, true]; }
    if dest[0] == Some(s.heads[1] as usize) && dest[1] == Some(s.heads[0] as usize) { dead = [true, true]; }
    let mut next = s;
    for p in 0..2 {
        if !dead[p] {
            if let Some(cell) = dest[p] {
                next.heads[p] = cell as u16;
                next.headings[p] = actions[p] as u8;
                next.set(cell);
            }
        }
        next.alive[p] = !dead[p];
    }
    next.result = match (dead[0], dead[1]) { (true, true) => 3, (true, false) => 2, (false, true) => 1, _ => 0 };
    next
}

fn flood(s: &State, start: usize) -> [bool; MAX_CELLS] {
    let mut seen = [false; MAX_CELLS];
    if start >= s.size() { return seen; }
    seen[start] = true;
    let mut queue = [0usize; MAX_CELLS];
    let mut head = 0; let mut tail = 1; queue[0] = start;
    while head < tail {
        let cell = queue[head]; head += 1;
        let x = cell % s.width as usize; let y = cell / s.width as usize;
        let ns = [(x as i32, y as i32 - 1), (x as i32 + 1, y as i32),
                  (x as i32, y as i32 + 1), (x as i32 - 1, y as i32)];
        for (nx, ny) in ns {
            if nx < 0 || ny < 0 || nx >= s.width as i32 || ny >= s.height as i32 { continue; }
            let n = ny as usize * s.width as usize + nx as usize;
            if !seen[n] && !s.has(n) { seen[n] = true; queue[tail] = n; tail += 1; }
        }
    }
    seen
}

fn distances(s: &State, player: usize) -> [i16; MAX_CELLS] {
    let mut d = [-1i16; MAX_CELLS];
    let start = s.heads[player] as usize;
    d[start] = 0;
    let mut queue = [0usize; MAX_CELLS]; let mut head = 0; let mut tail = 1; queue[0] = start;
    while head < tail {
        let cell = queue[head]; head += 1;
        let x = cell % s.width as usize; let y = cell / s.width as usize;
        let ns = [(x as i32, y as i32 - 1), (x as i32 + 1, y as i32),
                  (x as i32, y as i32 + 1), (x as i32 - 1, y as i32)];
        for (nx, ny) in ns {
            if nx < 0 || ny < 0 || nx >= s.width as i32 || ny >= s.height as i32 { continue; }
            let n = ny as usize * s.width as usize + nx as usize;
            if d[n] >= 0 || (s.has(n) && n != start) || n == s.heads[1-player] as usize { continue; }
            d[n] = d[cell] + 1; queue[tail] = n; tail += 1;
        }
    }
    d
}

fn articulation_count(s: &State, player: usize, reachable: &[bool; MAX_CELLS]) -> i32 {
    let start = s.heads[player] as usize;
    let excluded = s.heads[1-player] as usize;
    let mut disc = [0i16; MAX_CELLS]; let mut low = [0i16; MAX_CELLS]; let mut parent = [-1i16; MAX_CELLS];
    let mut children = [0u8; MAX_CELLS]; let mut art = [false; MAX_CELLS]; let mut clock = 0i16;
    fn dfs(v: usize, s: &State, reachable: &[bool; MAX_CELLS], excluded: usize,
           disc: &mut [i16; MAX_CELLS], low: &mut [i16; MAX_CELLS], parent: &mut [i16; MAX_CELLS],
           children: &mut [u8; MAX_CELLS], art: &mut [bool; MAX_CELLS], clock: &mut i16) {
        *clock += 1; disc[v] = *clock; low[v] = *clock;
        let x = v % s.width as usize; let y = v / s.width as usize;
        let ns = [(x as i32, y as i32 - 1), (x as i32 + 1, y as i32),
                  (x as i32, y as i32 + 1), (x as i32 - 1, y as i32)];
        for (nx, ny) in ns {
            if nx < 0 || ny < 0 || nx >= s.width as i32 || ny >= s.height as i32 { continue; }
            let n = ny as usize * s.width as usize + nx as usize;
            if !reachable[n] || n == excluded { continue; }
            if disc[n] == 0 {
                parent[n] = v as i16; children[v] += 1;
                dfs(n, s, reachable, excluded, disc, low, parent, children, art, clock);
                low[v] = low[v].min(low[n]);
                if parent[v] >= 0 && low[n] >= disc[v] { art[v] = true; }
            } else if n as i16 != parent[v] { low[v] = low[v].min(disc[n]); }
        }
        if parent[v] < 0 && children[v] > 1 { art[v] = true; }
    }
    if reachable[start] { dfs(start, s, reachable, excluded, &mut disc, &mut low, &mut parent, &mut children, &mut art, &mut clock); }
    art.iter().take(s.size()).filter(|x| **x).count() as i32
}

fn eval_with_regions(s: &State, regions: Option<([bool; MAX_CELLS], [bool; MAX_CELLS])>) -> f64 {
    match s.result { 3 => return 0.0, 1 => return 1.0, 2 => return -1.0, _ => () }
    let (r0, r1) = regions.unwrap_or_else(|| (
        flood(s, s.heads[0] as usize), flood(s, s.heads[1] as usize),
    ));
    let n0 = r0[..s.size()].iter().filter(|x| **x).count() as i32;
    let n1 = r1[..s.size()].iter().filter(|x| **x).count() as i32;
    let shared = (0..s.size()).any(|i| r0[i] && r1[i]);
    if !shared {
        return (((n0 - n1) as f64 / s.size() as f64) * 2.0).clamp(-0.95, 0.95);
    }
    let d0 = distances(s, 0); let d1 = distances(s, 1);
    let mut owned = [0i32; 3];
    for cell in 0..s.size() {
        if !( !s.has(cell) || cell == s.heads[0] as usize || cell == s.heads[1] as usize) { continue; }
        let (a,b) = (d0[cell],d1[cell]);
        if (a < 0 && b < 0) || a == b { owned[2] += 1; }
        else if b < 0 || (a >= 0 && a < b) { owned[0] += 1; }
        else { owned[1] += 1; }
    }
    let mobility = legal_mask(s,0).count_ones() as i32 - legal_mask(s,1).count_ones() as i32;
    let articulation = articulation_count(s, 0, &r0) - articulation_count(s, 1, &r1);
    (((owned[0]-owned[1]) as f64 / s.size() as f64) + 0.035 * mobility as f64 + 0.006 * articulation as f64).clamp(-0.95,0.95)
}

fn eval(s: &State) -> f64 { eval_with_regions(s, None) }

fn gaussian(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n=b.len(); let mut w: Vec<Vec<f64>>=(0..n).map(|i| { let mut r=a[i].clone(); r.push(b[i]); r }).collect();
    for c in 0..n {
        let mut pivot=c; for r in c+1..n { if w[r][c].abs()>w[pivot][c].abs(){pivot=r;} }
        if w[pivot][c].abs()<=1e-10{return None;} w.swap(c,pivot);
        let scale=w[c][c]; for j in c..=n {w[c][j]/=scale;}
        for r in 0..n {if r==c{continue;} let factor=w[r][c]; if factor!=0.0 {for j in c..=n {w[r][j]-=factor*w[c][j];}}}
    }
    Some((0..n).map(|i|w[i][n]).collect())
}

fn combinations(n: usize, k: usize, f: &mut impl FnMut(&[usize])) {
    fn rec(n:usize,k:usize,next:usize,cur:&mut Vec<usize>,f:&mut impl FnMut(&[usize])) {
        if cur.len()==k {f(cur); return;}
        let remaining=k-cur.len();
        for i in next..=n-remaining {cur.push(i); rec(n,k,i+1,cur,f); cur.pop();}
    }
    rec(n,k,0,&mut Vec::new(),f)
}

fn solve_side(pay: &[Vec<f64>], maximize: bool) -> Option<(Vec<f64>,f64)> {
    let (rows,cols)=(pay.len(),pay[0].len());
    let vars=if maximize {rows} else {cols};
    let mut constraints:Vec<(Vec<f64>,f64)>=Vec::new();
    if maximize {
        for i in 0..rows {let mut a=vec![0.0;rows+1];a[i]=1.0;constraints.push((a,0.0));}
        for j in 0..cols {let mut a=vec![0.0;rows+1];for i in 0..rows{a[i]=pay[i][j];}a[rows]=-1.0;constraints.push((a,0.0));}
    } else {
        for j in 0..cols {let mut a=vec![0.0;cols+1];a[j]=1.0;constraints.push((a,0.0));}
        for i in 0..rows {let mut a=vec![0.0;cols+1];for j in 0..cols{a[j]=-pay[i][j];}a[cols]=1.0;constraints.push((a,0.0));}
    }
    let mut best:Option<(Vec<f64>,f64)>=None;
    combinations(constraints.len(),vars,&mut |active| {
        let mut a=vec![vec![1.0;vars+1]]; a[0][vars]=0.0;
        let mut b=vec![1.0];
        for &ci in active {a.push(constraints[ci].0.clone());b.push(constraints[ci].1);}
        let Some(sol)=gaussian(&a,&b) else{return;};
        let probs=&sol[..vars]; let value=sol[vars];
        if probs.iter().any(|p|*p < -1e-8) || (probs.iter().sum::<f64>()-1.0).abs()>1e-7{return;}
        if maximize {
            for j in 0..cols {if (0..rows).map(|i|probs[i]*pay[i][j]).sum::<f64>()<value-1e-8{return;}}
        } else {
            for i in 0..rows {if (0..cols).map(|j|pay[i][j]*probs[j]).sum::<f64>()>value+1e-8{return;}}
        }
        if best.as_ref().map_or(true,|(_,v)|if maximize {value>*v+1e-8}else{value<*v-1e-8}) {
            let mut p:Vec<f64>=probs.iter().map(|x|x.max(0.0)).collect(); let sum=p.iter().sum::<f64>(); if sum>0.0{for x in &mut p{*x/=sum;}}
            best=Some((p,value));
        }
    });
    best
}

fn solve_matrix(pay: &[Vec<f64>]) -> Option<(f64,Vec<f64>,Vec<f64>)> {
    if pay.is_empty() || pay[0].is_empty() || pay.iter().any(|r|r.len()!=pay[0].len()){return None;}
    let (row,v0)=solve_side(pay,true)?; let (col,v1)=solve_side(pay,false)?;
    if (v0-v1).abs()>1e-6{return None;} Some(((v0+v1)*0.5,row,col))
}

struct Context { edges: usize, leaves: usize, budget: usize, cache: HashMap<(State,u8),f64> }
fn has_no_move(s:&State)->bool {legal_mask(s,0)==0||legal_mask(s,1)==0}
fn solve(s:State,depth:u8,c:&mut Context)->Result<f64,()> {
    if s.result!=0 {c.leaves+=1;return Ok(eval(&s));}
    if depth>0 {
        if let Some(v)=c.cache.get(&(s,depth)){return Ok(*v);}
    }
    let r0=flood(&s,s.heads[0] as usize);let r1=flood(&s,s.heads[1] as usize);
    let separated=!(0..s.size()).any(|i|r0[i]&&r1[i]);
    if depth==0||separated {
        if has_no_move(&s) {
            let a0=action_list(&s,0);let a1=action_list(&s,1);let mut matrix=Vec::new();
            for &x in &a0 {let mut row=Vec::new();for &y in &a1 {if c.edges>=c.budget{return Err(());}c.edges+=1;let n=transition(s,[x,y]);c.leaves+=1;row.push(eval(&n));}matrix.push(row);}
            return solve_matrix(&matrix).map(|x|x.0).ok_or(());
        }
        c.leaves+=1;return Ok(eval_with_regions(&s,Some((r0,r1))));
    }
    let a0=action_list(&s,0);let a1=action_list(&s,1);let mut matrix=Vec::new();
    for &x in &a0 {let mut row=Vec::new();for &y in &a1 {if c.edges>=c.budget{return Err(());}c.edges+=1;row.push(solve(transition(s,[x,y]),depth-1,c)?);}matrix.push(row);}
    let value=solve_matrix(&matrix).ok_or(())?.0;c.cache.insert((s,depth),value);Ok(value)
}

fn plan(s:State,max_depth:u8,budget:usize)->Option<SearchOutput>{
    let masks=[legal_mask(&s,0),legal_mask(&s,1)];
    if s.result!=0 {return Some(SearchOutput{value:eval(&s),policy:[[0.0;4];2],masks,action:-1,depth:0,edges:0,leaves:0});}
    let mut ctx=Context{edges:0,leaves:0,budget,cache:HashMap::new()};
    let mut best=eval(&s);let mut policies=[[0.0;4];2];let mut completed=0usize;let mut final_actions=action_list(&s,0);
    for player in 0..2 {
        let actions=action_list(&s,player);
        let legal_count=actions.iter().filter(|a|**a>=0).count();
        if legal_count>0 {for action in actions {if action>=0 {policies[player][action as usize]=1.0/legal_count as f64;}}}
    }
    for depth in 1..=max_depth {
        let before=ctx.edges;let a0=action_list(&s,0);let a1=action_list(&s,1);let mut matrix=Vec::new();let mut failed=false;
        'outer: for &x in &a0 {let mut row=Vec::new();for &y in &a1 {if ctx.edges>=ctx.budget{failed=true;break 'outer;}ctx.edges+=1;match solve(transition(s,[x,y]),depth-1,&mut ctx){Ok(v)=>row.push(v),Err(_)=>{failed=true;break 'outer;}}}matrix.push(row);}
        if failed {debug_assert!(ctx.edges>=before);break;}
        let (value,p0,p1)=solve_matrix(&matrix)?;best=value;completed=depth as usize;final_actions=if legal_mask(&s,0)==0{action_list(&s,0)}else{a0};
        for (i,&a) in final_actions.iter().enumerate(){if a>=0{policies[0][a as usize]=p0[i];}}
        for (i,&a) in a1.iter().enumerate(){if a>=0{policies[1][a as usize]=p1[i];}}
    }
    let action=if final_actions.iter().all(|a|*a<0){-1}else{final_actions.iter().copied().filter(|a|*a>=0).max_by(|&a,&b|policies[0][a as usize].total_cmp(&policies[0][b as usize]).then_with(||b.cmp(&a))).unwrap_or(-1) as i32};
    Some(SearchOutput{value:best,policy:policies,masks,action,depth:completed,edges:ctx.edges,leaves:ctx.leaves})
}

static mut INPUT: [u8; INPUT_BYTES] = [0; INPUT_BYTES];
static mut OUTPUT: [f64; 15] = [0.0; 15];
#[cfg(circuit_parity)]
static mut TRANSITION_OUTPUT: [u32; 14] = [0; 14];

#[no_mangle] pub extern "C" fn circuit_input_ptr()->*mut u8 { std::ptr::addr_of_mut!(INPUT).cast::<u8>() }
#[no_mangle] pub extern "C" fn circuit_output_ptr()->*mut f64 { std::ptr::addr_of_mut!(OUTPUT).cast::<f64>() }
#[no_mangle] pub extern "C" fn circuit_input_len()->usize { INPUT_BYTES }
#[no_mangle] pub extern "C" fn circuit_output_len()->usize { 15 }
#[cfg(circuit_parity)]
#[no_mangle] pub extern "C" fn circuit_transition_ptr()->*mut u32 { std::ptr::addr_of_mut!(TRANSITION_OUTPUT).cast::<u32>() }
#[cfg(circuit_parity)]
#[no_mangle] pub extern "C" fn circuit_transition_len()->usize { 14 }
#[cfg(circuit_parity)]
#[no_mangle] pub extern "C" fn circuit_transition(action0:i32,action1:i32)->i32 {
    unsafe {
        let i=std::ptr::addr_of!(INPUT).cast::<u8>();
        let byte=|n:usize|*i.add(n);
        let w=byte(0) as usize;let h=byte(1) as usize;let size=w*h;
        if w<2||h<2||size>MAX_CELLS||byte(6)>3||byte(7)>3 {return -1;}
        let mut occ=[0u64;WORDS];
        for cell in 0..size {if (byte(14+cell/8)>>(cell%8))&1!=0{occ[cell/64]|=1u64<<(cell%64);}}
        for cell in size..MAX_CELLS {if (byte(14+cell/8)>>(cell%8))&1!=0{return -2;}}
        let s=State{width:w as u8,height:h as u8,heads:[byte(2) as u16,byte(3) as u16],headings:[byte(4),byte(5)],occupied:occ,alive:[byte(7)&1!=0,byte(7)&2!=0],result:byte(6)};
        if s.heads.iter().any(|x|*x as usize>=size)||s.headings.iter().any(|x|*x>3)||!s.has(s.heads[0] as usize)||!s.has(s.heads[1] as usize){return -2;}
        let next=transition(s,[action0 as i8,action1 as i8]);
        let out=std::ptr::addr_of_mut!(TRANSITION_OUTPUT).cast::<u32>();
        *out=next.result as u32;*out.add(1)=(next.alive[0] as u32)|((next.alive[1] as u32)<<1);
        *out.add(2)=next.heads[0] as u32;*out.add(3)=next.heads[1] as u32;
        *out.add(4)=next.headings[0] as u32;*out.add(5)=next.headings[1] as u32;
        for word in 0..WORDS {let bits=next.occupied[word];*out.add(6+word*2)=bits as u32;*out.add(7+word*2)=(bits>>32) as u32;}
        0
    }
}
#[no_mangle] pub extern "C" fn circuit_plan()->i32 {
    unsafe {
        let i=std::ptr::addr_of!(INPUT).cast::<u8>();
        let byte=|n:usize|*i.add(n);
        let w=byte(0) as usize;let h=byte(1) as usize;let size=w*h;
        if w<2||h<2||size>MAX_CELLS||byte(8)==0||byte(8)>32||byte(6)>3||byte(7)>3 {return -1;}
        let mut occ=[0u64;WORDS];
        // The fixed packet keeps the u32 budget at bytes 10..14 and occupancy begins at 14.
        let budget=u32::from_le_bytes([byte(10),byte(11),byte(12),byte(13)]) as usize;
        for cell in 0..size {if (byte(14+cell/8)>>(cell%8))&1!=0{occ[cell/64]|=1u64<<(cell%64);}}
        for cell in size..MAX_CELLS {if (byte(14+cell/8)>>(cell%8))&1!=0{return -2;}}
        let s=State{width:w as u8,height:h as u8,heads:[byte(2) as u16,byte(3) as u16],headings:[byte(4),byte(5)],occupied:occ,alive:[byte(7)&1!=0,byte(7)&2!=0],result:byte(6)};
        if s.heads.iter().any(|x|*x as usize>=size)||s.headings.iter().any(|x|*x>3)||!s.has(s.heads[0] as usize)||!s.has(s.heads[1] as usize){return -2;}
        let Some(r)=plan(s,byte(8),budget.max(1)) else{return -3;};
        let o=std::ptr::addr_of_mut!(OUTPUT).cast::<f64>();for k in 0..15{*o.add(k)=0.0;}*o.add(0)=r.value;
        for p in 0..2 {for a in 0..4{*o.add(1+p*4+a)=r.policy[p][a];}}
        *o.add(9)=r.masks[0] as f64;*o.add(10)=r.masks[1] as f64;*o.add(11)=r.action as f64;
        *o.add(12)=r.depth as f64;*o.add(13)=r.edges as f64;*o.add(14)=r.leaves as f64;0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(heads: [u16; 2]) -> State {
        let mut s=State{width:4,height:3,heads,headings:[1,3],occupied:[0;WORDS],alive:[true;2],result:0};
        s.set(heads[0] as usize);s.set(heads[1] as usize);s
    }

    #[test]
    fn simultaneous_same_destination_kills_both() {
        let s=state([4,6]);
        let next=transition(s,[1,3]);
        assert_eq!(next.result,3);
        assert_eq!(next.heads,s.heads);
        assert_eq!(next.alive,[false,false]);
    }

    #[test]
    fn invalid_proposal_does_not_cancel_other_riders_move() {
        let s=state([0,6]);
        let next=transition(s,[-1,2]);
        assert_eq!(next.result,2);
        assert_eq!(next.heads,[0,10]);
        assert_eq!(next.alive,[false,true]);
    }

    #[test]
    fn attempted_head_swap_is_a_simultaneous_draw() {
        let s=state([4,5]);
        let next=transition(s,[1,3]);
        assert_eq!(next.result,3);
        assert_eq!(next.heads,s.heads);
    }

    #[test]
    fn insufficient_root_budget_returns_only_legal_actions() {
        let mut s=state([1,7]);
        s.headings=[3,1];
        let result=plan(s,1,1).unwrap();
        assert_eq!(result.depth,0);
        assert!(result.masks[0] & (1 << result.action) != 0);
        assert_eq!(result.action,2);
        assert!((result.policy[0].iter().sum::<f64>()-1.0).abs()<1e-9);
        assert!((result.policy[1].iter().sum::<f64>()-1.0).abs()<1e-9);
    }
}
