#![allow(dead_code)]
#![allow(unused_variables)]

use std::{collections::HashMap, default::Default};

use parking_lot::Mutex;

// TODO: this used to filter the gpu profiler stats here, but it doesn't make sense
// for the `puffin` output. Better to filter near where the stats are being displayed instead.
const FILTER_KERNEL_SIZE: usize = 1; //8

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GpuProfilerQueryId(u64);

impl Default for GpuProfilerQueryId {
    fn default() -> Self {
        Self(std::u64::MAX)
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct RenderScopeDesc {
    pub name: String,
    pub id: u64,
}

pub fn create_gpu_query(scope: RenderScopeDesc, user_id: usize) -> GpuProfilerQueryId {
    GPU_PROFILER.lock().create_gpu_query(scope, user_id)
}

pub fn report_durations_ticks(
    ns_per_tick: f32,
    durations: impl Iterator<Item = (GpuProfilerQueryId, u64)>,
) {
    let mut prof = GPU_PROFILER.lock();
    prof.report_durations_ticks(ns_per_tick, durations);
}

pub fn forget_queries(queries: impl Iterator<Item = GpuProfilerQueryId>) {
    let mut prof = GPU_PROFILER.lock();
    prof.forget_queries(queries);
}

pub fn with_stats<F: FnOnce(&GpuProfilerStats)>(f: F) {
    f(&GPU_PROFILER.lock().stats);
}

pub fn get_stats() -> GpuProfilerStats {
    GPU_PROFILER.lock().stats.clone()
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct GpuProfilerScopeId(RenderScopeDesc, usize);

impl GpuProfilerScopeId {
    pub fn new(s: RenderScopeDesc, user_id: usize) -> Self {
        Self(s, user_id)
    }
}

// TODO: currently merges multiple invocations in a frame into a single bucket, and averages it
// should instead report the count per frame along with correct per-hit timing
#[derive(Clone)]
pub struct GpuProfilerScope {
    pub scope: RenderScopeDesc,
    pub hits: Vec<u64>, // nanoseconds
    pub write_head: u32,
}

impl GpuProfilerScope {
    fn new(scope: RenderScopeDesc) -> GpuProfilerScope {
        GpuProfilerScope {
            hits: vec![0u64; FILTER_KERNEL_SIZE],
            write_head: 0,
            scope,
        }
    }
}

impl GpuProfilerScope {
    pub fn hit_count(&self) -> u32 {
        self.write_head.min(self.hits.len() as u32)
    }

    pub fn average_duration_millis(&self) -> f64 {
        let count = (self.hit_count() as f64).max(1.0);
        self.hits.iter().sum::<u64>() as f64 / count / 1_000_000.0
    }
}

#[derive(Default, Clone)]
pub struct GpuProfilerStats {
    pub scopes: HashMap<GpuProfilerScopeId, GpuProfilerScope>,
    pub order: Vec<GpuProfilerScopeId>,
}

struct ActiveQuery {
    id: GpuProfilerQueryId,
    scope: RenderScopeDesc,
    user_id: usize,
}

impl GpuProfilerStats {
    fn report_duration_nanos(
        &mut self,
        query_id: GpuProfilerQueryId,
        duration: u64,
        active_query: ActiveQuery,
    ) {
        let scope_id = GpuProfilerScopeId::new(active_query.scope.clone(), active_query.user_id);
        self.order.push(scope_id.clone());

        let mut entry = self
            .scopes
            .entry(scope_id)
            .or_insert_with(|| GpuProfilerScope::new(active_query.scope));

        let len = entry.hits.len();
        entry.hits[entry.write_head as usize % len] = duration;
        entry.write_head += 1;
    }

    pub fn get_ordered(&self) -> Vec<(RenderScopeDesc, f64)> {
        self.order
            .iter()
            .map(|scope_id| {
                let scope = &self.scopes[scope_id];
                (scope.scope.clone(), scope.average_duration_millis())
            })
            .collect()
    }
}

struct GpuProfiler {
    active_queries: HashMap<GpuProfilerQueryId, ActiveQuery>,
    frame_query_ids: Vec<GpuProfilerQueryId>,
    next_query_id: u64,
    stats: GpuProfilerStats,
}

impl GpuProfiler {
    pub fn new() -> Self {
        Self {
            active_queries: Default::default(),
            frame_query_ids: Default::default(),
            next_query_id: 0,
            stats: Default::default(),
        }
    }

    fn report_durations_ticks(
        &mut self,
        ns_per_tick: f32,
        durations: impl Iterator<Item = (GpuProfilerQueryId, u64)>,
    ) {
        self.stats.order.clear();

        for (query_id, duration_ticks) in durations {
            // Remove the finished queries from the active list
            let q = self.active_queries.remove(&query_id).unwrap();
            let duration = (duration_ticks as f64 * ns_per_tick as f64) as u64;
            self.stats.report_duration_nanos(query_id, duration, q);
        }
    }

    fn forget_queries(&mut self, queries: impl Iterator<Item = GpuProfilerQueryId>) {
        for query_id in queries {
            let q = self.active_queries.remove(&query_id).unwrap();
        }
    }

    fn create_gpu_query(&mut self, scope: RenderScopeDesc, user_id: usize) -> GpuProfilerQueryId {
        let id = GpuProfilerQueryId(self.next_query_id);
        self.next_query_id += 1;
        self.frame_query_ids.push(id);

        // TODO: prune old ones
        self.active_queries
            .insert(id, ActiveQuery { id, scope, user_id });
        assert!(self.active_queries.len() < 8192);
        id
    }
}

lazy_static::lazy_static! {
    static ref GPU_PROFILER: Mutex<GpuProfiler> = Mutex::new(GpuProfiler::new());
}
