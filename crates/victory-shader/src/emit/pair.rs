//! In a group of several console waves on a host of 32-lane waves, the two host waves of a
//! console wave meet by themselves (an arrival counter in LDS) rather than at group barriers,
//! which D3D12 defines only where the whole group comes (`VICTORY_PAIR_SYNC=1`); and the joins
//! of branches a group of several console waves takes together.

use super::*;

/// The mode of a group barrier the translation adds (host waves meeting):
/// a wait for the group with a shared-memory fence; the fence alone with
/// `VICTORY_GROUP_SYNC=0` ([`super::group_sync`]).
/// `VICTORY_PAIR_SYNC=1` (off unless set): in groups of several console waves, the host waves of
/// one console wave meet by themselves ([`Lower::pair_sync`]) instead of at group barriers, so
/// programs whose console waves branch or loop apart translate for 32-lane hosts. Off by default:
/// a meeting costs about as much as a group barrier, but a program the title had refused then
/// runs, and one of them (vfx_triangleClip:CS_DefragHeap_ComputeRelocations, a serial loop of
/// lane reads) takes 15 ms of an RTX 5060 Laptop's time a frame.
pub(super) fn pair_sync_on() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST_PAIR_SYNC.get() {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_PAIR_SYNC").is_ok_and(|v| v.trim() == "1"))
}

#[cfg(test)]
thread_local! {
    /// A test's `VICTORY_PAIR_SYNC`, for its thread only.
    pub(crate) static TEST_PAIR_SYNC: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

impl Lower<'_> {
    /// At the join of group-widened branches ([`Self::wave_wide`] with `group_wide`): a console
    /// wave that did not take a branch keeps the scalars (and SCC) it had before it, as on the
    /// console, where it skipped the region; the group ran the region together, so the join's
    /// phis bring the region's values. Each scalar phi is taken as `took ? joined : before`.
    pub(super) fn keep_skipped_scalars(&mut self, b: usize) -> Result<(), Stop> {
        let p = self.x.p;
        let Some(branches) = self.wide_regions.as_ref().and_then(|w| w.joins.get(&b)).cloned() else {
            return Ok(());
        };
        for br in branches {
            let Some(&took) = self.group_took.get(&br) else { continue };
            let Some(n) = p.blocks[b].preds.iter().position(|&q| q == br) else { continue };
            for phi in &p.blocks[b].phis {
                if matches!(phi.reg, Reg::V(_)) || matches!(phi.reg, Reg::S(c) if c == EXEC || c == EXEC + 1) {
                    continue;
                }
                let before = phi.args[n];
                let (joined, old) = match (self.vals.get(&phi.value).copied(), self.vals.get(&before).copied()) {
                    (Some(Val::I32(j)), Some(Val::I32(o))) | (Some(Val::Bit(j)), Some(Val::Bit(o))) => (j, o),
                    (Some(Val::Missing(_)) | None, _) => continue,
                    _ => {
                        return Err(unsupported(
                            p.blocks[b].offset,
                            "a scalar joined past a group-widened branch with no value before it",
                        ));
                    }
                };
                let v = self.s.m.select(self.cur, took, joined, old);
                let val = match phi.reg {
                    Reg::Scc => Val::Bit(v),
                    _ => Val::I32(v),
                };
                self.vals.insert(phi.value, val);
            }
        }
        Ok(())
    }

    /// The two host waves of this thread's console wave meet: what each wrote to LDS before is
    /// seen by the other after. A group barrier when the group is one console wave; in a group
    /// of several, only the console wave's own pair meets (each host wave adds one to the console
    /// wave's counter and waits until the other has too), so console waves that branch or loop
    /// differently need not meet each other, as on the console, where a wave runs alone. The
    /// meeting is then in new blocks: the result is the block to go on in (`self.cur` too when
    /// `at` was it). `VICTORY_PAIR_SYNC=0` keeps group barriers.
    /// This thread's console wave's arrival counter in LDS ([`Self::pair_sync`]), zeroed in the
    /// entry block, which every thread runs, before the group meets.
    pub(super) fn pair_counter_index(&mut self, base: u32) -> V {
        if let Some(v) = self.pair_counter {
            return v;
        }
        let thread = self.console_thread();
        let e = self.s.entry;
        let six = self.s.m.i32(6);
        let wave = self.s.m.bin(e, Bin::LShr, thread, six);
        let b0 = self.s.m.i32(base);
        let idx = self.s.m.bin(e, Bin::Add, b0, wave);
        let lds = self.s.lds();
        let ptr = self.s.m.element(e, lds, idx);
        let zero = self.s.m.i32(0);
        self.s.m.store(e, ptr, zero);
        self.group_sync(e);
        self.pair_counter = Some(idx);
        idx
    }

    /// With pair meetings, the exchange dwords of the meeting this host wave goes to next: 0 or
    /// 2 past its console wave's first. Before it arrives at its k-th meeting the counter holds
    /// 2k, or 2k + 1 if the other host wave is there first: both take bank k & 1. A bank is
    /// written again two meetings on, which the writer passes only after the other host wave
    /// came to the meeting between, past its reads of the bank. `None` without pair meetings
    /// (or where they are group barriers: the entry block, `VICTORY_GROUP_SYNC=0`).
    pub(super) fn exchange_bank(&mut self, at: usize) -> Option<V> {
        let base = self.pair_slots?;
        if at == self.s.entry || super::super::group_sync() == 0 {
            return None;
        }
        let i32t = self.s.t.i32;
        let idx = self.pair_counter_index(base);
        let lds = self.s.lds();
        let ptr = self.s.m.element(at, lds, idx);
        let count = self.s.m.load(at, ptr);
        let count = self.s.op(at, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, &[count]);
        let two = self.s.m.i32(2);
        Some(self.s.m.bin(at, Bin::And, count, two))
    }

    pub(super) fn pair_sync(&mut self, at: usize) -> usize {
        let Some(base) = self.pair_slots else {
            self.group_sync(at);
            return at;
        };
        if at == self.s.entry || super::super::group_sync() == 0 {
            // (the entry block every thread runs: the group meets there anyway)
            self.group_sync(at);
            return at;
        }
        let (i1t, i32t, void) = (self.s.t.i1, self.s.t.i32, self.s.t.void);
        let idx = self.pair_counter_index(base);
        // `at` goes on after the meeting: what it holds so far moves to a block before it, so
        // that whoever holds `at` (a caller deep in an expression, an IR block's tail) still
        // holds the right block. Branches to `at` that the lowering keeps go there too.
        let head = self.s.m.split_head(at);
        for b in self.block.iter_mut().flatten() {
            if *b == at {
                *b = head;
            }
        }
        for b in self.edges.values_mut() {
            if *b == at {
                *b = head;
            }
        }
        if self.abort == Some(at) {
            self.abort = Some(head);
        }
        let fence = self.s.m.i32(8);
        self.s.op(head, "barrier", Attr::NoDuplicate, void, 80, &[fence]);
        let lds = self.s.lds();
        let (zero, one) = (self.s.m.i32(0), self.s.m.i32(1));
        // Only the host wave's first lane touches the counter (the GPU takes a lane's atomic
        // on one address at a time); the others take its answer.
        let first = self.s.op(head, "waveIsFirstLane", Attr::NoUnwind, i1t, 110, &[]);
        let (arrive, arrived) = (self.s.m.new_block(), self.s.m.new_block());
        self.s.m.terminate(head, Term::CondBr(first, arrive, arrived));
        let ptr = self.s.m.element(arrive, lds, idx);
        let mine = self.s.m.atomic(arrive, Rmw::Add, ptr, one);
        self.s.m.terminate(arrive, Term::Br(arrived));
        let old = self.s.m.phi(arrived, i32t);
        self.s.m.set_phi(old, vec![(mine, arrive), (zero, head)]);
        let old = self.s.op(arrived, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, &[old]);
        // arrivals 2k and 2k + 1 are the k-th meeting's: both wait for the count 2k + 2
        let not1 = self.s.m.i32(!1);
        let even = self.s.m.bin(arrived, Bin::And, old, not1);
        let two = self.s.m.i32(2);
        let target = self.s.m.bin(arrived, Bin::Add, even, two);
        let (spin, poll, polled) = (self.s.m.new_block(), self.s.m.new_block(), self.s.m.new_block());
        self.s.m.terminate(arrived, Term::Br(spin));
        // (bounded: a meeting the other host wave never comes to ends after 2^20 polls, a
        // wrong value rather than a lost device)
        let polls = self.s.m.phi(spin, i32t);
        let first = self.s.op(spin, "waveIsFirstLane", Attr::NoUnwind, i1t, 110, &[]);
        self.s.m.terminate(spin, Term::CondBr(first, poll, polled));
        let ptr = self.s.m.element(poll, lds, idx);
        let seen = self.s.m.atomic(poll, Rmw::Or, ptr, zero);
        self.s.m.terminate(poll, Term::Br(polled));
        let now = self.s.m.phi(polled, i32t);
        self.s.m.set_phi(now, vec![(seen, poll), (zero, spin)]);
        let now = self.s.op(polled, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, &[now]);
        let ahead = self.s.m.bin(polled, Bin::Sub, now, target);
        let done = self.s.m.cmp(polled, pred::SGE, ahead, zero);
        let next = self.s.m.bin(polled, Bin::Add, polls, one);
        let limit = self.s.m.i32(1 << 20);
        let over = self.s.m.cmp(polled, pred::IUGE, next, limit);
        let out = self.s.m.bin(polled, Bin::Or, done, over);
        self.s.m.set_phi(polls, vec![(zero, arrived), (next, polled)]);
        self.s.m.terminate(polled, Term::CondBr(out, at, spin));
        self.s.op(at, "barrier", Attr::NoDuplicate, void, 80, &[fence]);
        at
    }

    /// The two host waves of a console wave meet (a group barrier with a shared-memory
    /// fence): LDS writes before are seen by reads after.
    pub(super) fn lds_meeting(&mut self) {
        self.pair_sync(self.cur);
    }

    /// [`Lower::wave_wide`] in a group of several console waves with group barriers among the
    /// halves (fork pull request #1124): the group runs the branch's body if any of its console
    /// waves does (so that every thread meets every barrier), and a console wave that does not
    /// is kept out of the body by its own `took` (effects guarded, scalars taken back at the
    /// join, [`Lower::keep_skipped_scalars`]). Exact only for a forward branch around a body that
    /// rejoins at its other way (`if (some lane) { ... }`, either way round); a loop or an
    /// `else` the console waves would split over stays refused. `some`: whether some lane of
    /// this host wave speaks for the branch (for `any`) or against it.
    pub(super) fn group_wave_wide(&mut self, b: usize, any: bool, some: V, slot: u32, host_waves: u32) -> Result<V, Why> {
        let c = self.cur;
        let (i1t, i32t) = (self.s.t.i1, self.s.t.i32);
        const SHAPE: Why = "a wave-wide branch in a group of several console waves that is not a forward if";
        if !self.all_group(b) {
            return Err("a wave-wide branch inside a divergent scalar branch");
        }
        let LaneExit::Branch { taken, not_taken, .. } = self.x.st.exits[b] else {
            return Err(SHAPE);
        };
        let regions = self.wide_regions.as_ref().ok_or(SHAPE)?;
        let region = regions.region.get(&b).ok_or(SHAPE)?;
        let body_taken = match (region[taken], region[not_taken]) {
            (true, false) => true,
            (false, true) => false,
            _ => return Err(SHAPE),
        };
        let body = if body_taken { taken } else { not_taken };
        if self.rpo[body] <= self.rpo[b] {
            return Err(SHAPE);
        }
        let outer_took = regions.innermost[b].and_then(|r| self.group_took.get(&r).copied());
        // This console wave's own decision (both of its halves), then whether it enters
        // the body; a console wave already kept out of an enclosing body does not.
        let mine32 = self.s.m.cast(c, Cast::ZExt, some, i32t);
        let (lo, hi) = self.exchange_halves(c, mine32);
        let c = self.cur;
        let both = self.s.m.bin(c, Bin::Or, lo, hi);
        let z0 = self.s.m.i32(0);
        // (the console's branch goes to `taken` when `d`)
        let d = self.s.m.cmp(c, if any { pred::NE } else { pred::EQ }, both, z0);
        let mut took = if body_taken {
            d
        } else {
            let t = self.s.m.bool(true);
            self.s.m.bin(c, Bin::Xor, d, t)
        };
        if let Some(t) = outer_took {
            took = self.s.m.bin(c, Bin::And, took, t);
        }
        self.group_took.insert(b, took);
        // Some console wave of the group enters: one dword per host wave in LDS.
        let thread = self.console_thread();
        let c = self.cur;
        let five = self.s.m.i32(5);
        let hw = self.s.m.bin(c, Bin::LShr, thread, five);
        let took32 = self.s.m.cast(c, Cast::ZExt, took, i32t);
        let base = self.s.m.i32(slot);
        let idx = self.s.m.bin(c, Bin::Add, base, hw);
        let lds = self.s.lds();
        let ptr = self.s.m.element(c, lds, idx);
        self.s.m.store(c, ptr, took32);
        self.group_sync(c);
        let lane = self.s.op(c, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
        let count = self.s.m.i32(host_waves);
        let z = self.s.m.i32(0);
        let in_range = self.s.m.cmp(c, pred::IULT, lane, count);
        let off = self.s.m.select(c, in_range, lane, z);
        let ridx = self.s.m.bin(c, Bin::Add, base, off);
        let rptr = self.s.m.element(c, lds, ridx);
        let val = self.s.m.load(c, rptr);
        let nz = self.s.m.cmp(c, pred::NE, val, z);
        let enter = self.s.op(c, "waveAnyTrue", Attr::NoUnwind, i1t, 113, &[nz]);
        self.group_sync(c);
        Ok(if body_taken {
            enter
        } else {
            let t = self.s.m.bool(true);
            self.s.m.bin(c, Bin::Xor, enter, t)
        })
    }
}
