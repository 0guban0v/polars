use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Instant;

use polars_async::executor::TaskPriority;
use polars_async::primitives::opt_spawned_future::parallelize_first_to_local;
use polars_core::frame::DataFrame;
use polars_core::prelude::{ArrowField, BooleanChunked, ChunkFilter, Column, DataType, IntoColumn};
use polars_core::series::Series;
use polars_core::utils::arrow::bitmap::{Bitmap, MutableBitmap};
use polars_error::PolarsResult;
use polars_io::RowIndex;
use polars_io::predicates::{
    ColumnPredicateExpr, ColumnPredicates, ScanIOPredicate, SpecializedColumnPredicate,
};
pub use polars_io::prelude::_internal::PrefilterMaskSetting;
use polars_io::prelude::try_set_sorted_flag;
use polars_parquet::read::{Filter, PredicateFilter, PrimitiveLogicalType};
use polars_utils::pl_str::PlSmallStr;
use polars_utils::{IdxSize, UnitVec};

use super::row_group_data_fetch::RowGroupData;
use crate::nodes::io_sources::parquet::projection::ArrowFieldProjection;

static ISSUE_28304_TRACE_ENABLED: LazyLock<bool> =
    LazyLock::new(|| std::env::var("POLARS_ISSUE_28304_TRACE").as_deref() == Ok("1"));
static ISSUE_28304_TRACE_EPOCH: LazyLock<Instant> = LazyLock::new(Instant::now);
static ISSUE_28304_TRACE_EVENT: AtomicU64 = AtomicU64::new(0);
static ISSUE_28304_ACTIVE_DECODES: AtomicUsize = AtomicUsize::new(0);
static ISSUE_28304_PEAK_DECODES: AtomicUsize = AtomicUsize::new(0);

/// Environment-gated instrumentation for #28304 research.
struct Issue28304DecodeTrace {
    event: u64,
    role: &'static str,
    column: PlSmallStr,
    input_rows: usize,
    selected_rows: Option<usize>,
    output_rows: Option<usize>,
    mask_rows: Option<usize>,
    started_ns: u128,
    started_at: Instant,
}

#[derive(Default)]
pub(super) struct Issue28304AdaptiveState {
    issued: AtomicUsize,
    completed: AtomicUsize,
    all_at_once_count: AtomicUsize,
    all_at_once_ns: AtomicU64,
    staged_count: AtomicUsize,
    staged_ns: AtomicU64,
    rolling: Mutex<Issue28304RollingObservations>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Issue28304RollingLevel {
    Marginal,
    Joint,
    Cost,
    TaskSupply,
}

impl Issue28304RollingLevel {
    fn from_env() -> Option<Self> {
        match std::env::var("POLARS_ISSUE_28304_ROLLING_POLICY").as_deref() {
            Ok("marginal") => Some(Self::Marginal),
            Ok("joint") => Some(Self::Joint),
            Ok("cost") => Some(Self::Cost),
            Ok("task_supply") => Some(Self::TaskSupply),
            Ok(value) => panic!(
                "unknown POLARS_ISSUE_28304_ROLLING_POLICY={value:?}; expected marginal, joint, cost, or task_supply"
            ),
            Err(_) => None,
        }
    }

    fn uses_joint(self) -> bool {
        self != Self::Marginal
    }

    fn uses_cost(self) -> bool {
        matches!(self, Self::Cost | Self::TaskSupply)
    }
}

#[derive(Clone, Copy)]
struct Issue28304RollingEstimate {
    value: f64,
    observations: usize,
}

impl Issue28304RollingEstimate {
    fn selectivity_prior() -> Self {
        Self {
            value: 0.5,
            observations: 0,
        }
    }

    fn update_selectivity(&mut self, value: f64) {
        // Discount by row group rather than row count so one large group does
        // not make policy unable to react to distribution drift.
        self.value = 0.5 * self.value + 0.5 * value;
        self.observations += 1;
    }

    fn update_cost(&mut self, value: f64) {
        self.value = if self.observations == 0 {
            value
        } else {
            0.5 * self.value + 0.5 * value
        };
        self.observations += 1;
    }
}

#[derive(Default)]
struct Issue28304RollingObservations {
    all_at_once_groups: usize,
    marginal: BTreeMap<usize, Issue28304RollingEstimate>,
    prefix_joint: BTreeMap<usize, Issue28304RollingEstimate>,
    full_decode_ns_per_row: BTreeMap<usize, Issue28304RollingEstimate>,
}

struct Issue28304PredicateObservation {
    field_index: usize,
    mask: Bitmap,
    elapsed_ns: u64,
}

struct Issue28304RollingDecision {
    issued: usize,
    deferred: Option<usize>,
    predicted_prefix_selectivity: f64,
    predicted_saved_fraction: f64,
    in_flight: usize,
    observed_groups: usize,
    reason: &'static str,
}

impl Issue28304AdaptiveState {
    fn observe_all_at_once(
        &self,
        input_rows: usize,
        observations: &[Issue28304PredicateObservation],
    ) {
        if observations.is_empty() || input_rows == 0 {
            return;
        }

        let mut state = self.rolling.lock().unwrap();
        for observation in observations {
            let marginal = observation.mask.set_bits() as f64 / input_rows as f64;
            state
                .marginal
                .entry(observation.field_index)
                .or_insert_with(Issue28304RollingEstimate::selectivity_prior)
                .update_selectivity(marginal);
            state
                .full_decode_ns_per_row
                .entry(observation.field_index)
                .or_insert(Issue28304RollingEstimate {
                    value: 0.0,
                    observations: 0,
                })
                .update_cost(observation.elapsed_ns as f64 / input_rows as f64);
        }

        for deferred in observations {
            let mut prefix = MutableBitmap::new();
            let mut prefix_masks = observations
                .iter()
                .filter(|observation| observation.field_index != deferred.field_index)
                .map(|observation| &observation.mask);
            let Some(first) = prefix_masks.next() else {
                continue;
            };
            prefix.extend_from_bitmap(first);
            for mask in prefix_masks {
                <&mut MutableBitmap as std::ops::BitAndAssign<&Bitmap>>::bitand_assign(
                    &mut &mut prefix,
                    mask,
                );
            }
            let prefix_selectivity = prefix.set_bits() as f64 / input_rows as f64;
            state
                .prefix_joint
                .entry(deferred.field_index)
                .or_insert_with(Issue28304RollingEstimate::selectivity_prior)
                .update_selectivity(prefix_selectivity);
        }
        state.all_at_once_groups += 1;
    }

    fn observe_staged_prefix(
        &self,
        input_rows: usize,
        deferred: usize,
        prefix_selectivity: f64,
        observations: &[Issue28304PredicateObservation],
    ) {
        if observations.is_empty() || input_rows == 0 {
            return;
        }

        let mut state = self.rolling.lock().unwrap();
        for observation in observations {
            let marginal = observation.mask.set_bits() as f64 / input_rows as f64;
            state
                .marginal
                .entry(observation.field_index)
                .or_insert_with(Issue28304RollingEstimate::selectivity_prior)
                .update_selectivity(marginal);
            state
                .full_decode_ns_per_row
                .entry(observation.field_index)
                .or_insert(Issue28304RollingEstimate {
                    value: 0.0,
                    observations: 0,
                })
                .update_cost(observation.elapsed_ns as f64 / input_rows as f64);
        }
        state
            .prefix_joint
            .entry(deferred)
            .or_insert_with(Issue28304RollingEstimate::selectivity_prior)
            .update_selectivity(prefix_selectivity);
    }

    fn choose_rolling(
        &self,
        predicate_field_indices: &[usize],
        num_pipelines: usize,
        level: Issue28304RollingLevel,
    ) -> Issue28304RollingDecision {
        let issued = self.issued.fetch_add(1, Ordering::Relaxed);
        let completed = self.completed.load(Ordering::Acquire);
        let in_flight = issued.saturating_sub(completed) + 1;
        let state = self.rolling.lock().unwrap();
        let observed_groups = state.all_at_once_groups;

        let fallback = |reason| Issue28304RollingDecision {
            issued,
            deferred: None,
            predicted_prefix_selectivity: 1.0,
            predicted_saved_fraction: 0.0,
            in_flight,
            observed_groups,
            reason,
        };

        if observed_groups < 2 {
            return fallback("cold_start");
        }
        if issued % 8 == 7 {
            return fallback("refresh");
        }

        let mut baseline_cost = 0.0;
        for field_index in predicate_field_indices {
            let Some(marginal) = state.marginal.get(field_index) else {
                return fallback("missing_selectivity");
            };
            let cost = if level.uses_cost() {
                let Some(cost) = state.full_decode_ns_per_row.get(field_index) else {
                    return fallback("missing_cost");
                };
                cost.value
            } else {
                let _ = marginal;
                1.0
            };
            baseline_cost += cost;
        }

        let mut best: Option<(usize, f64, f64)> = None;
        for deferred in predicate_field_indices {
            let prefix_selectivity = if level.uses_joint() {
                let Some(joint) = state.prefix_joint.get(deferred) else {
                    return fallback("missing_joint_selectivity");
                };
                joint.value
            } else {
                predicate_field_indices
                    .iter()
                    .filter(|field_index| *field_index != deferred)
                    .map(|field_index| state.marginal[field_index].value)
                    .product()
            };
            if prefix_selectivity > 0.05 {
                continue;
            }
            let deferred_cost = if level.uses_cost() {
                state.full_decode_ns_per_row[deferred].value
            } else {
                1.0
            };
            let predicted_saved_fraction =
                deferred_cost * (1.0 - prefix_selectivity) / baseline_cost;
            if best.is_none_or(|(_, saved, _)| predicted_saved_fraction > saved) {
                best = Some((*deferred, predicted_saved_fraction, prefix_selectivity));
            }
        }

        let Some((deferred, predicted_saved_fraction, predicted_prefix_selectivity)) = best else {
            return fallback("prefix_not_selective");
        };
        if predicted_saved_fraction <= 0.05 {
            return fallback("gain_below_margin");
        }
        if level == Issue28304RollingLevel::TaskSupply && in_flight < num_pipelines {
            return fallback("insufficient_task_supply");
        }

        Issue28304RollingDecision {
            issued,
            deferred: Some(deferred),
            predicted_prefix_selectivity,
            predicted_saved_fraction,
            in_flight,
            observed_groups,
            reason: "predicted_gain",
        }
    }

    fn complete_rolling_row_group(&self) {
        self.completed.fetch_add(1, Ordering::Release);
    }

    fn choose_staged(&self) -> (usize, bool) {
        let issued = self.issued.fetch_add(1, Ordering::Relaxed);
        let all_at_once_count = self.all_at_once_count.load(Ordering::Acquire);
        let staged_count = self.staged_count.load(Ordering::Acquire);

        if all_at_once_count < 2 || staged_count < 2 {
            return (issued, issued % 2 == 1);
        }

        let all_at_once_mean =
            self.all_at_once_ns.load(Ordering::Relaxed) as f64 / all_at_once_count as f64;
        let staged_mean = self.staged_ns.load(Ordering::Relaxed) as f64 / staged_count as f64;
        let staged_is_best = staged_mean < all_at_once_mean;

        // Keep bounded exploration so changed row-group distribution can
        // overturn current choice.
        let explore = issued % 16 == 15;
        (issued, staged_is_best ^ explore)
    }

    fn observe(&self, staged: bool, elapsed_ns: u64) {
        let (count, total_ns) = if staged {
            (&self.staged_count, &self.staged_ns)
        } else {
            (&self.all_at_once_count, &self.all_at_once_ns)
        };
        total_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
        count.fetch_add(1, Ordering::Release);
        self.completed.fetch_add(1, Ordering::Release);
    }
}

impl Issue28304DecodeTrace {
    fn new(
        role: &'static str,
        column: PlSmallStr,
        input_rows: usize,
        selected_rows: Option<usize>,
    ) -> Option<Self> {
        if !*ISSUE_28304_TRACE_ENABLED {
            return None;
        }

        let event = ISSUE_28304_TRACE_EVENT.fetch_add(1, Ordering::Relaxed);
        let active = ISSUE_28304_ACTIVE_DECODES.fetch_add(1, Ordering::SeqCst) + 1;
        ISSUE_28304_PEAK_DECODES.fetch_max(active, Ordering::SeqCst);
        let started_at = Instant::now();
        let started_ns = ISSUE_28304_TRACE_EPOCH.elapsed().as_nanos();
        eprintln!(
            "POLARS_ISSUE_28304_TRACE phase=start event={event} role={role} column={column} input_rows={input_rows} selected_rows={selected_rows:?} started_ns={started_ns} active={active} peak={}",
            ISSUE_28304_PEAK_DECODES.load(Ordering::SeqCst),
        );

        Some(Self {
            event,
            role,
            column,
            input_rows,
            selected_rows,
            output_rows: None,
            mask_rows: None,
            started_ns,
            started_at,
        })
    }

    fn finish(&mut self, output_rows: usize, mask_rows: Option<usize>) {
        self.output_rows = Some(output_rows);
        self.mask_rows = mask_rows;
    }
}

impl Drop for Issue28304DecodeTrace {
    fn drop(&mut self) {
        let finished_ns = ISSUE_28304_TRACE_EPOCH.elapsed().as_nanos();
        let active = ISSUE_28304_ACTIVE_DECODES.fetch_sub(1, Ordering::SeqCst) - 1;
        eprintln!(
            "POLARS_ISSUE_28304_TRACE phase=end event={} role={} column={} input_rows={} selected_rows={:?} output_rows={:?} mask_rows={:?} started_ns={} finished_ns={} elapsed_ns={} active={} peak={}",
            self.event,
            self.role,
            self.column,
            self.input_rows,
            self.selected_rows,
            self.output_rows,
            self.mask_rows,
            self.started_ns,
            finished_ns,
            self.started_at.elapsed().as_nanos(),
            active,
            ISSUE_28304_PEAK_DECODES.load(Ordering::SeqCst),
        );
    }
}

/// Turns row group data into DataFrames.
pub(super) struct RowGroupDecoder {
    pub(super) num_pipelines: usize,
    pub(super) projected_arrow_fields: Arc<[ArrowFieldProjection]>,
    pub(super) allow_column_predicates: bool,
    pub(super) row_index: Option<RowIndex>,
    pub(super) predicate: Option<ScanIOPredicate>,
    pub(super) use_prefiltered: Option<PrefilterMaskSetting>,
    /// Indices into `projected_arrow_fields. This must be sorted.
    pub(super) predicate_field_indices: Arc<[usize]>,
    /// Indices into `projected_arrow_fields. This must be sorted.
    pub(super) non_predicate_field_indices: Arc<[usize]>,
    pub(super) target_values_per_thread: usize,
    pub(super) issue_28304_adaptive_state: Arc<Issue28304AdaptiveState>,
}

impl RowGroupDecoder {
    pub(super) async fn row_group_data_to_df(
        &self,
        mut row_group_data: RowGroupData,
    ) -> PolarsResult<DataFrame> {
        // If the slice consumes the entire row-group. Don't slice. This allows for prefiltering to
        // happen more often until we properly support prefiltering with pre-slices.
        row_group_data.slice.take_if(|slice| {
            slice.0 == 0 && slice.1 >= row_group_data.row_group_metadata.num_rows()
        });

        if self.use_prefiltered.is_some()
            && row_group_data.slice.is_none()
            && !self.predicate_field_indices.is_empty()
        {
            self.row_group_data_to_df_prefiltered(row_group_data).await
        } else {
            self.row_group_data_to_df_impl(row_group_data).await
        }
    }

    async fn row_group_data_to_df_impl(
        &self,
        row_group_data: RowGroupData,
    ) -> PolarsResult<DataFrame> {
        let row_group_data = Arc::new(row_group_data);

        let out_width = self.row_index.is_some() as usize + self.projected_arrow_fields.len();

        let mut out_columns = Vec::with_capacity(out_width);

        let slice_range = row_group_data
            .slice
            .map(|(offset, len)| offset..offset + len)
            .unwrap_or(0..row_group_data.row_group_metadata.num_rows());

        assert!(slice_range.end <= row_group_data.row_group_metadata.num_rows());

        if let Some(s) = self.materialize_row_index(row_group_data.as_ref(), slice_range.clone())? {
            out_columns.push(s);
        }

        let mut decoded_cols = Vec::with_capacity(row_group_data.row_group_metadata.n_columns());
        self.decode_projected_columns(
            &mut decoded_cols,
            &row_group_data,
            Some(polars_parquet::read::Filter::Range(slice_range.clone())),
        )
        .await?;

        drop(row_group_data);

        let projection_height = slice_range.len();

        out_columns.extend(decoded_cols);

        let df = unsafe { DataFrame::new_unchecked(projection_height, out_columns) };

        let df = if let Some(predicate) = self.predicate.as_ref() {
            let mask = predicate.predicate.evaluate_io(&df)?;
            let mask = mask.bool().unwrap();

            let filtered =
                filter_cols(df.into_columns(), mask, self.target_values_per_thread).await?;

            let height = if let Some(fst) = filtered.first() {
                fst.len()
            } else {
                mask.num_trues()
            };

            unsafe { DataFrame::new_unchecked(height, filtered) }
        } else {
            df
        };

        assert_eq!(df.width(), out_width); // `out_width` should have been calculated correctly

        Ok(df)
    }

    fn materialize_row_index(
        &self,
        row_group_data: &RowGroupData,
        slice_range: core::ops::Range<usize>,
    ) -> PolarsResult<Option<Column>> {
        if let Some(RowIndex { name, offset }) = self.row_index.clone() {
            let projection_height = slice_range.len();

            let offset = offset.saturating_add(
                IdxSize::try_from(row_group_data.row_offset + slice_range.start)
                    .unwrap_or(IdxSize::MAX),
            );

            // The DataFrame can be empty at this point if no columns were projected from the file,
            // so we create the row index column manually instead of using `df.with_row_index` to
            // ensure it has the correct number of rows.
            Ok(Some(Column::new_row_index(
                name,
                offset,
                projection_height,
            )?))
        } else {
            Ok(None)
        }
    }

    /// Potentially parallelizes based on number of rows & columns. Decoded columns are appended to
    /// `out_vec`.
    async fn decode_projected_columns(
        &self,
        out_vec: &mut Vec<Column>,
        row_group_data: &Arc<RowGroupData>,
        filter: Option<polars_parquet::read::Filter>,
    ) -> PolarsResult<()> {
        let projected_arrow_fields = &self.projected_arrow_fields;
        let expected_num_rows = filter
            .as_ref()
            .map_or(row_group_data.row_group_metadata.num_rows(), |x| {
                x.num_rows(row_group_data.row_group_metadata.num_rows())
            });

        // Ensure we provide the same output column order as the pre-filtered decode.
        let get_projected_field_at_output_index = {
            let predicate_field_indices = self.predicate_field_indices.clone();
            let non_predicate_field_indices = self.non_predicate_field_indices.clone();

            move |i: usize| {
                if predicate_field_indices.is_empty() {
                    i
                } else if i < predicate_field_indices.len() {
                    predicate_field_indices[i]
                } else {
                    non_predicate_field_indices[i - predicate_field_indices.len()]
                }
            }
        };

        let cols_per_thread = calc_cols_per_thread(
            row_group_data.row_group_metadata.num_rows(),
            self.target_values_per_thread,
        );

        let projected_arrow_fields = projected_arrow_fields.clone();
        let row_group_data_2 = row_group_data.clone();

        let task_handles = {
            let projected_arrow_fields = projected_arrow_fields.clone();
            let filter = filter.clone();

            parallelize_first_to_local(
                TaskPriority::Low,
                (0..projected_arrow_fields.len())
                    .step_by(cols_per_thread)
                    .map(move |offset| {
                        let row_group_data = row_group_data_2.clone();
                        let projected_arrow_fields = projected_arrow_fields.clone();
                        let filter = filter.clone();
                        let get_projected_field_at_output_index =
                            get_projected_field_at_output_index.clone();

                        async move {
                            // This is exact as we have already taken out the remainder.
                            (offset
                                ..offset
                                    .saturating_add(cols_per_thread)
                                    .min(projected_arrow_fields.len()))
                                .map(|i| {
                                    let projection = &projected_arrow_fields
                                        [get_projected_field_at_output_index(i)];

                                    let (col, pred_true_mask) = decode_column(
                                        projection.arrow_field(),
                                        &row_group_data,
                                        filter.clone(),
                                        expected_num_rows,
                                    )?;

                                    let col = projection.apply_transform(col)?;

                                    Ok((col, pred_true_mask))
                                })
                                .collect::<PolarsResult<UnitVec<_>>>()
                        }
                    }),
            )
        };

        for fut in task_handles {
            out_vec.extend(fut.await?.into_iter().map(|(c, _)| c));
        }

        Ok(())
    }
}

fn decode_column(
    arrow_field: &ArrowField,
    row_group_data: &RowGroupData,
    filter: Option<polars_parquet::read::Filter>,
    expected_num_rows: usize,
) -> PolarsResult<(Column, Bitmap)> {
    let Some(iter) = row_group_data
        .row_group_metadata
        .columns_under_root_iter(&arrow_field.name)
    else {
        return Ok((
            Column::full_null(
                arrow_field.name.clone(),
                expected_num_rows,
                &DataType::from_arrow_field(arrow_field),
            ),
            Bitmap::default(),
        ));
    };

    let columns_to_deserialize = iter
        .map(|col_md| {
            let byte_range = col_md.byte_range();

            (
                col_md,
                row_group_data
                    .fetched_bytes
                    .get_range(byte_range.start as usize..byte_range.end as usize),
            )
        })
        .collect::<Vec<_>>();

    let skip_num_rows_check = matches!(filter, Some(Filter::Predicate(_)));

    let (arrays, pred_true_mask) = polars_io::prelude::_internal::to_deserializer(
        columns_to_deserialize,
        arrow_field.clone(),
        filter,
    )?;

    if !skip_num_rows_check {
        let num_rows = arrays.iter().map(|array| array.len()).sum::<usize>();
        assert_eq!(num_rows, expected_num_rows);
    }

    let mut series = Series::try_from((arrow_field, arrays))?;

    if let Some(col_idxs) = row_group_data
        .row_group_metadata
        .columns_idxs_under_root_iter(&arrow_field.name)
    {
        if col_idxs.len() == 1 {
            try_set_sorted_flag(&mut series, col_idxs[0], &row_group_data.sorting_map);
        }
    }

    // TODO: Also load in the metadata.

    Ok((series.into_column(), pred_true_mask))
}

/// Filters columns, in parallel depending number of rows / columns.
async fn filter_cols(
    cols: Vec<Column>,
    mask: &BooleanChunked,
    target_values_per_thread: usize,
) -> PolarsResult<Vec<Column>> {
    if cols.is_empty() {
        return Ok(cols);
    }

    let cols_per_thread = calc_cols_per_thread(cols[0].len(), target_values_per_thread);
    let mut out_vec = Vec::with_capacity(cols.len());
    let cols = Arc::new(cols);
    let mask = mask.clone();

    let task_handles = {
        let cols = &cols;
        let mask = &mask;

        parallelize_first_to_local(
            TaskPriority::Low,
            (0..cols.len()).step_by(cols_per_thread).map(move |offset| {
                let cols = cols.clone();
                let mask = mask.clone();
                async move {
                    (offset..offset.saturating_add(cols_per_thread).min(cols.len()))
                        .map(|i| cols[i].filter(&mask))
                        .collect::<PolarsResult<UnitVec<_>>>()
                }
            }),
        )
    };

    for fut in task_handles {
        out_vec.extend(fut.await?)
    }

    Ok(out_vec)
}

fn calc_cols_per_thread(n_rows_per_col: usize, target_n_rows_per_thread: usize) -> usize {
    if n_rows_per_col == 0 {
        return usize::MAX;
    }

    let n = target_n_rows_per_thread / n_rows_per_col;
    let floor_distance = target_n_rows_per_thread % n_rows_per_col;
    let ceil_distance = n_rows_per_col - floor_distance;

    if floor_distance <= ceil_distance {
        n.max(1)
    } else {
        n + 1
    }
}

// Pre-filtered

fn decode_column_in_filter(
    arrow_field: &ArrowField,
    use_column_predicates: bool,
    column_predicates: &ColumnPredicates,
    row_group_data: &RowGroupData,
    projection_height: usize,
    input_selection: Option<Bitmap>,
) -> PolarsResult<(Column, Bitmap)> {
    let mut filter = None;
    let mut selected_predicate = None;
    let mut constant = None;
    if use_column_predicates {
        if let Some((column_predicate, specialized)) =
            column_predicates.predicates.get(&arrow_field.name)
        {
            constant = specialized.as_ref().and_then(|s| match s {
                SpecializedColumnPredicate::Equal(sc) if !sc.is_null() => Some(sc),
                _ => None,
            });

            let p = ColumnPredicateExpr::new(
                arrow_field.name.clone(),
                DataType::from_arrow_field(arrow_field),
                column_predicate.clone(),
                specialized.clone(),
            );
            let predicate = PredicateFilter {
                predicate: Arc::new(p) as _,
                include_values: constant.is_none(),
            };
            if let Some(input_selection) = input_selection {
                selected_predicate = Some((predicate, input_selection));
            } else {
                filter = Some(Filter::Predicate(predicate));
            }
        }
    }
    let (mut c, m) = if let Some((predicate, input_selection)) = selected_predicate {
        decode_column_selected(
            arrow_field,
            row_group_data,
            predicate,
            input_selection,
            projection_height,
        )?
    } else {
        decode_column(arrow_field, row_group_data, filter, projection_height)?
    };

    if let Some(constant) = constant {
        c = Column::new_scalar(c.name().clone(), constant.clone(), m.set_bits());
    }

    Ok((c, m))
}

fn decode_column_selected(
    arrow_field: &ArrowField,
    row_group_data: &RowGroupData,
    predicate: PredicateFilter,
    input_selection: Bitmap,
    expected_num_rows: usize,
) -> PolarsResult<(Column, Bitmap)> {
    let Some(iter) = row_group_data
        .row_group_metadata
        .columns_under_root_iter(&arrow_field.name)
    else {
        return Ok((
            Column::full_null(
                arrow_field.name.clone(),
                expected_num_rows,
                &DataType::from_arrow_field(arrow_field),
            ),
            Bitmap::default(),
        ));
    };

    let columns_to_deserialize = iter
        .map(|col_md| {
            let byte_range = col_md.byte_range();
            (
                col_md,
                row_group_data
                    .fetched_bytes
                    .get_range(byte_range.start as usize..byte_range.end as usize),
            )
        })
        .collect::<Vec<_>>();

    let (arrays, pred_true_mask) = polars_io::prelude::_internal::to_deserializer_selected(
        columns_to_deserialize,
        arrow_field.clone(),
        predicate,
        input_selection,
    )?;
    let mut series = Series::try_from((arrow_field, arrays))?;

    if let Some(col_idxs) = row_group_data
        .row_group_metadata
        .columns_idxs_under_root_iter(&arrow_field.name)
        && col_idxs.len() == 1
    {
        try_set_sorted_flag(&mut series, col_idxs[0], &row_group_data.sorting_map);
    }

    Ok((series.into_column(), pred_true_mask))
}

impl RowGroupDecoder {
    fn issue_28304_stages(&self) -> Option<Vec<Vec<usize>>> {
        let spec = std::env::var("POLARS_ISSUE_28304_STAGES").ok()?;
        assert!(
            self.allow_column_predicates,
            "POLARS_ISSUE_28304_STAGES requires independent flat column predicates"
        );

        let mut stages = Vec::new();
        let mut seen = Vec::new();
        for stage_spec in spec.split('|') {
            let mut stage = Vec::new();
            for name in stage_spec
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
            {
                let index = self
                    .predicate_field_indices
                    .iter()
                    .copied()
                    .find(|&index| self.projected_arrow_fields[index].output_name() == name)
                    .unwrap_or_else(|| {
                        panic!("unknown predicate column {name:?} in POLARS_ISSUE_28304_STAGES")
                    });
                assert!(
                    !seen.contains(&index),
                    "duplicate predicate column {name:?} in POLARS_ISSUE_28304_STAGES"
                );
                seen.push(index);
                stage.push(index);
            }
            assert!(
                !stage.is_empty(),
                "empty stage in POLARS_ISSUE_28304_STAGES"
            );
            stages.push(stage);
        }

        let mut expected = self.predicate_field_indices.to_vec();
        expected.sort_unstable();
        seen.sort_unstable();
        assert_eq!(
            seen, expected,
            "POLARS_ISSUE_28304_STAGES must mention every predicate column exactly once"
        );
        Some(stages)
    }

    async fn row_group_data_to_df_prefiltered_staged(
        &self,
        row_group_data: RowGroupData,
        stages: Vec<Vec<usize>>,
    ) -> PolarsResult<DataFrame> {
        debug_assert!(row_group_data.slice.is_none());
        debug_assert!(self.row_index.is_none());

        let row_group_data = Arc::new(row_group_data);
        let projection_height = row_group_data.row_group_metadata.num_rows();
        let scan_predicate = self.predicate.as_ref().unwrap();
        let rolling_deferred = (Issue28304RollingLevel::from_env().is_some()
            && stages.len() == 2
            && stages[1].len() == 1)
            .then(|| stages[1][0]);

        let mut current_mask = Bitmap::new_with_value(true, projection_height);
        let mut live_columns: Vec<(usize, Column)> = Vec::new();

        for (stage_index, stage) in stages.into_iter().enumerate() {
            let stage_len = stage.len();
            let observe_rolling_prefix = stage_index == 0 && rolling_deferred.is_some();
            let input_selection = (stage_index > 0).then(|| current_mask.clone());
            let cols_per_thread = stage_len.div_ceil(self.num_pipelines).max(1);
            let task_handles = {
                let projected_arrow_fields = self.projected_arrow_fields.clone();
                let row_group_data = row_group_data.clone();
                let column_predicates = scan_predicate.column_predicates.clone();

                parallelize_first_to_local(
                    TaskPriority::Low,
                    (0..stage.len())
                        .step_by(cols_per_thread)
                        .map(move |offset| {
                            let stage = stage.clone();
                            let projected_arrow_fields = projected_arrow_fields.clone();
                            let row_group_data = row_group_data.clone();
                            let column_predicates = column_predicates.clone();
                            let input_selection = input_selection.clone();

                            async move {
                                (offset..offset.saturating_add(cols_per_thread).min(stage.len()))
                                    .map(|i| {
                                        let field_index = stage[i];
                                        let projection = &projected_arrow_fields[field_index];
                                        let selected_rows = input_selection
                                            .as_ref()
                                            .map_or(projection_height, Bitmap::set_bits);
                                        let mut trace = Issue28304DecodeTrace::new(
                                            "staged_predicate",
                                            projection.arrow_field().name.clone(),
                                            projection_height,
                                            Some(selected_rows),
                                        );
                                        let rolling_started =
                                            observe_rolling_prefix.then(Instant::now);
                                        let (col, pred_true_mask) = decode_column_in_filter(
                                            projection.arrow_field(),
                                            true,
                                            column_predicates.as_ref(),
                                            row_group_data.as_ref(),
                                            projection_height,
                                            input_selection.clone(),
                                        )?;
                                        let col = projection.apply_transform(col)?;
                                        if let Some(trace) = trace.as_mut() {
                                            trace
                                                .finish(col.len(), Some(pred_true_mask.set_bits()));
                                        }
                                        let rolling_elapsed_ns = rolling_started.map(|started| {
                                            started.elapsed().as_nanos().min(u64::MAX as u128)
                                                as u64
                                        });
                                        Ok((field_index, col, pred_true_mask, rolling_elapsed_ns))
                                    })
                                    .collect::<PolarsResult<UnitVec<_>>>()
                            }
                        }),
                )
            };

            let mut stage_columns = Vec::with_capacity(stage_len);
            let mut stage_masks = Vec::with_capacity(stage_len);
            let mut rolling_observations = Vec::with_capacity(stage_len);
            for fut in task_handles {
                for (field_index, column, mask, elapsed_ns) in fut.await? {
                    stage_columns.push((field_index, column));
                    if let Some(elapsed_ns) = elapsed_ns {
                        rolling_observations.push(Issue28304PredicateObservation {
                            field_index,
                            mask: mask.clone(),
                            elapsed_ns,
                        });
                    }
                    stage_masks.push(mask);
                }
            }

            let mut combined = MutableBitmap::new();
            combined.extend_from_bitmap(stage_masks.first().unwrap());
            for stage_mask in &stage_masks[1..] {
                <&mut MutableBitmap as std::ops::BitAndAssign<&Bitmap>>::bitand_assign(
                    &mut &mut combined,
                    stage_mask,
                );
            }
            let combined = combined.freeze();
            if let Some(deferred) = rolling_deferred.filter(|_| stage_index == 0) {
                self.issue_28304_adaptive_state.observe_staged_prefix(
                    projection_height,
                    deferred,
                    combined.set_bits() as f64 / projection_height as f64,
                    &rolling_observations,
                );
            }
            let combined_ca = BooleanChunked::from_bitmap(PlSmallStr::EMPTY, combined.clone());

            if stage_index > 0 {
                let current_ca = BooleanChunked::from_bitmap(
                    PlSmallStr::EMPTY,
                    std::mem::replace(&mut current_mask, combined.clone()),
                );
                let compact_mask = combined_ca.filter(&current_ca)?;
                for (_, column) in &mut live_columns {
                    *column = column.filter(&compact_mask)?;
                }
            } else {
                current_mask = combined.clone();
            }

            for ((field_index, column), column_mask) in stage_columns.into_iter().zip(stage_masks) {
                let column_mask = BooleanChunked::from_bitmap(PlSmallStr::EMPTY, column_mask);
                let compact_mask = combined_ca.filter(&column_mask)?;
                live_columns.push((field_index, column.filter(&compact_mask)?));
            }
        }

        live_columns.sort_unstable_by_key(|(field_index, _)| *field_index);
        let mut live_columns = live_columns
            .into_iter()
            .map(|(_, column)| column)
            .collect::<Vec<_>>();

        let final_mask = BooleanChunked::from_bitmap(PlSmallStr::EMPTY, current_mask.clone());
        let expected_num_rows = current_mask.set_bits();
        let cols_per_thread = self
            .non_predicate_field_indices
            .len()
            .div_ceil(self.num_pipelines)
            .max(1);
        let task_handles = {
            let non_predicate_field_indices = self.non_predicate_field_indices.clone();
            let projected_arrow_fields = self.projected_arrow_fields.clone();
            let row_group_data = row_group_data.clone();

            parallelize_first_to_local(
                TaskPriority::Low,
                (0..non_predicate_field_indices.len())
                    .step_by(cols_per_thread)
                    .map(move |offset| {
                        let non_predicate_field_indices = non_predicate_field_indices.clone();
                        let projected_arrow_fields = projected_arrow_fields.clone();
                        let row_group_data = row_group_data.clone();
                        let final_mask = final_mask.clone();
                        let current_mask = current_mask.clone();

                        async move {
                            (offset
                                ..offset
                                    .saturating_add(cols_per_thread)
                                    .min(non_predicate_field_indices.len()))
                                .map(|i| {
                                    let projection =
                                        &projected_arrow_fields[non_predicate_field_indices[i]];
                                    let col = decode_column_prefiltered(
                                        projection.arrow_field(),
                                        row_group_data.as_ref(),
                                        &final_mask,
                                        &current_mask,
                                        expected_num_rows,
                                    )?;
                                    projection.apply_transform(col)
                                })
                                .collect::<PolarsResult<UnitVec<_>>>()
                        }
                    }),
            )
        };

        for fut in task_handles {
            live_columns.extend(fut.await?);
        }
        Ok(unsafe { DataFrame::new_unchecked(expected_num_rows, live_columns) })
    }

    async fn row_group_data_to_df_prefiltered(
        &self,
        row_group_data: RowGroupData,
    ) -> PolarsResult<DataFrame> {
        if let Some(level) = Issue28304RollingLevel::from_env() {
            let decision = self.issue_28304_adaptive_state.choose_rolling(
                &self.predicate_field_indices,
                self.num_pipelines,
                level,
            );
            let deferred_name = decision.deferred.map(|field_index| {
                self.projected_arrow_fields[field_index]
                    .output_name()
                    .to_string()
            });
            let result = if let Some(deferred) = decision.deferred {
                let first_stage = self
                    .predicate_field_indices
                    .iter()
                    .copied()
                    .filter(|field_index| *field_index != deferred)
                    .collect();
                self.row_group_data_to_df_prefiltered_staged(
                    row_group_data,
                    vec![first_stage, vec![deferred]],
                )
                .await
            } else {
                self.row_group_data_to_df_prefiltered_all_at_once(row_group_data, true)
                    .await
            };
            self.issue_28304_adaptive_state.complete_rolling_row_group();
            if std::env::var("POLARS_ISSUE_28304_POLICY_TRACE").as_deref() == Ok("1") {
                eprintln!(
                    "POLARS_ISSUE_28304_ROLLING issued={} level={level:?} plan={} deferred={} predicted_prefix_selectivity={:.6} predicted_saved_fraction={:.6} in_flight={} observed_groups={} reason={}",
                    decision.issued,
                    if decision.deferred.is_some() {
                        "staged"
                    } else {
                        "all_at_once"
                    },
                    deferred_name.as_deref().unwrap_or("none"),
                    decision.predicted_prefix_selectivity,
                    decision.predicted_saved_fraction,
                    decision.in_flight,
                    decision.observed_groups,
                    decision.reason,
                );
            }
            return result;
        }

        if let Some(stages) = self.issue_28304_stages() {
            if std::env::var("POLARS_ISSUE_28304_ADAPTIVE").as_deref() == Ok("1") {
                let (issued, staged) = self.issue_28304_adaptive_state.choose_staged();
                let started = Instant::now();
                let result = if staged {
                    self.row_group_data_to_df_prefiltered_staged(row_group_data, stages)
                        .await
                } else {
                    self.row_group_data_to_df_prefiltered_all_at_once(row_group_data, false)
                        .await
                };
                let elapsed_ns = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
                self.issue_28304_adaptive_state.observe(staged, elapsed_ns);
                if *ISSUE_28304_TRACE_ENABLED {
                    eprintln!(
                        "POLARS_ISSUE_28304_POLICY issued={issued} plan={} elapsed_ns={elapsed_ns}",
                        if staged { "staged" } else { "all_at_once" },
                    );
                }
                return result;
            }

            return self
                .row_group_data_to_df_prefiltered_staged(row_group_data, stages)
                .await;
        }

        self.row_group_data_to_df_prefiltered_all_at_once(row_group_data, false)
            .await
    }

    async fn row_group_data_to_df_prefiltered_all_at_once(
        &self,
        row_group_data: RowGroupData,
        observe_rolling: bool,
    ) -> PolarsResult<DataFrame> {
        debug_assert!(row_group_data.slice.is_none()); // Invariant of the optimizer.
        assert!(self.predicate_field_indices.len() <= self.projected_arrow_fields.len());

        let row_group_data = Arc::new(row_group_data);
        let projection_height = row_group_data.row_group_metadata.num_rows();

        let mut live_columns = Vec::with_capacity(
            self.row_index.is_some() as usize
                + self.predicate_field_indices.len()
                + self.non_predicate_field_indices.len(),
        );
        let mut masks = Vec::with_capacity(
            self.row_index.is_some() as usize + self.predicate_field_indices.len(),
        );

        if let Some(s) = self.materialize_row_index(
            row_group_data.as_ref(),
            0..row_group_data.row_group_metadata.num_rows(),
        )? {
            live_columns.push(s);
        }

        let scan_predicate = self.predicate.as_ref().unwrap();

        let use_column_predicates = self.allow_column_predicates
            && !row_group_data
                .row_group_metadata
                .parquet_columns()
                .iter()
                .any(|c| {
                    matches!(
                        c.descriptor().descriptor.primitive_type.logical_type,
                        Some(PrimitiveLogicalType::Float16)
                    )
                });

        let cols_per_thread = (self
            .predicate_field_indices
            .len()
            .div_ceil(self.num_pipelines))
        .max(1);
        let task_handles = {
            let predicate_field_indices = self.predicate_field_indices.clone();
            let projected_arrow_fields = self.projected_arrow_fields.clone();
            let row_group_data = row_group_data.clone();

            parallelize_first_to_local(
                TaskPriority::Low,
                (0..self.predicate_field_indices.len())
                    .step_by(cols_per_thread)
                    .map(move |offset| {
                        let row_group_data = row_group_data.clone();
                        let predicate_field_indices = predicate_field_indices.clone();
                        let projected_arrow_fields = projected_arrow_fields.clone();
                        let column_predicates = scan_predicate.column_predicates.clone();

                        async move {
                            (offset
                                ..offset
                                    .saturating_add(cols_per_thread)
                                    .min(predicate_field_indices.len()))
                                .map(|i| {
                                    let field_index = predicate_field_indices[i];
                                    let projection = &projected_arrow_fields[field_index];

                                    if use_column_predicates {
                                        debug_assert!(matches!(
                                            projection,
                                            ArrowFieldProjection::Plain(_)
                                        ));
                                    }

                                    let mut trace = Issue28304DecodeTrace::new(
                                        "predicate",
                                        projection.arrow_field().name.clone(),
                                        projection_height,
                                        None,
                                    );
                                    let rolling_started = observe_rolling.then(Instant::now);
                                    let (col, pred_true_mask) = decode_column_in_filter(
                                        projection.arrow_field(),
                                        use_column_predicates,
                                        column_predicates.as_ref(),
                                        row_group_data.as_ref(),
                                        projection_height,
                                        None,
                                    )?;

                                    let col = projection.apply_transform(col)?;
                                    if let Some(trace) = trace.as_mut() {
                                        trace.finish(col.len(), Some(pred_true_mask.set_bits()));
                                    }

                                    let rolling_elapsed_ns = rolling_started.map(|started| {
                                        started.elapsed().as_nanos().min(u64::MAX as u128) as u64
                                    });

                                    Ok((field_index, col, pred_true_mask, rolling_elapsed_ns))
                                })
                                .collect::<PolarsResult<UnitVec<_>>>()
                        }
                    }),
            )
        };

        let mut rolling_observations = Vec::with_capacity(masks.capacity());
        for fut in task_handles {
            for (field_index, c, m, elapsed_ns) in fut.await? {
                live_columns.push(c);
                if let Some(elapsed_ns) = elapsed_ns {
                    rolling_observations.push(Issue28304PredicateObservation {
                        field_index,
                        mask: m.clone(),
                        elapsed_ns,
                    });
                }
                masks.push(m);
            }
        }
        if observe_rolling {
            self.issue_28304_adaptive_state
                .observe_all_at_once(projection_height, &rolling_observations);
        }

        let (live_df_filtered, mut mask) = if use_column_predicates {
            assert!(scan_predicate.column_predicates.is_sumwise_complete);
            if let [mask] = masks.as_slice() {
                (
                    unsafe { DataFrame::new_unchecked_infer_height(live_columns) },
                    BooleanChunked::from_bitmap(PlSmallStr::EMPTY, mask.clone()),
                )
            } else {
                let mut mask = MutableBitmap::new();
                mask.extend_from_bitmap(masks.first().unwrap());
                for col_mask in &masks[1..] {
                    <&mut MutableBitmap as std::ops::BitAndAssign<&Bitmap>>::bitand_assign(
                        &mut &mut mask,
                        col_mask,
                    );
                }
                let mask = BooleanChunked::from_bitmap(PlSmallStr::EMPTY, mask.freeze());
                let live_columns = live_columns
                    .into_iter()
                    .zip(masks)
                    .map(|(col, col_mask)| {
                        let col_mask = BooleanChunked::from_bitmap(PlSmallStr::EMPTY, col_mask);
                        let col_mask = mask.filter(&col_mask).unwrap();
                        col.filter(&col_mask).unwrap()
                    })
                    .collect();

                (
                    unsafe { DataFrame::new_unchecked_infer_height(live_columns) },
                    mask,
                )
            }
        } else {
            let mut live_df = unsafe {
                DataFrame::new_unchecked(row_group_data.row_group_metadata.num_rows(), live_columns)
            };

            let mask = scan_predicate.predicate.evaluate_io(&live_df)?;
            let mask = mask.bool().unwrap();

            unsafe {
                live_df.columns_mut().truncate(
                    self.row_index.is_some() as usize + self.predicate_field_indices.len(),
                )
            }

            let filtered =
                filter_cols(live_df.into_columns(), mask, self.target_values_per_thread).await?;

            let filtered_height = if let Some(fst) = filtered.first() {
                fst.len()
            } else {
                mask.num_trues()
            };

            (
                unsafe { DataFrame::new_unchecked(filtered_height, filtered) },
                mask.clone(),
            )
        };

        if self.non_predicate_field_indices.is_empty() {
            // User or test may have explicitly requested prefiltering
            return Ok(live_df_filtered);
        }

        mask.rechunk_mut();
        let mask_bitmap = mask.downcast_as_array();
        let mask_bitmap = match mask_bitmap.validity() {
            None => mask_bitmap.values().clone(),
            Some(v) => mask_bitmap.values() & v,
        };

        assert_eq!(mask_bitmap.len(), projection_height);

        let expected_num_rows = mask_bitmap.set_bits();

        let cols_per_thread = (self
            .predicate_field_indices
            .len()
            .div_ceil(self.num_pipelines))
        .max(1);

        let task_handles = {
            let non_predicate_field_indices = self.non_predicate_field_indices.clone();
            let non_predicate_len = non_predicate_field_indices.len();
            let projected_arrow_fields = self.projected_arrow_fields.clone();
            let row_group_data = row_group_data.clone();

            parallelize_first_to_local(
                TaskPriority::Low,
                (0..non_predicate_len)
                    .step_by(cols_per_thread)
                    .map(move |offset| {
                        let row_group_data = row_group_data.clone();
                        let non_predicate_field_indices = non_predicate_field_indices.clone();
                        let projected_arrow_fields = projected_arrow_fields.clone();
                        let mask = mask.clone();
                        let mask_bitmap = mask_bitmap.clone();

                        async move {
                            (offset
                                ..offset
                                    .saturating_add(cols_per_thread)
                                    .min(non_predicate_len))
                                .map(|i| {
                                    let projection =
                                        &projected_arrow_fields[non_predicate_field_indices[i]];

                                    let mut trace = Issue28304DecodeTrace::new(
                                        "non_predicate",
                                        projection.arrow_field().name.clone(),
                                        projection_height,
                                        Some(expected_num_rows),
                                    );
                                    let col = decode_column_prefiltered(
                                        projection.arrow_field(),
                                        row_group_data.as_ref(),
                                        &mask,
                                        &mask_bitmap,
                                        expected_num_rows,
                                    )?;

                                    let col = projection.apply_transform(col)?;
                                    if let Some(trace) = trace.as_mut() {
                                        trace.finish(col.len(), None);
                                    }
                                    Ok(col)
                                })
                                .collect::<PolarsResult<UnitVec<_>>>()
                        }
                    }),
            )
        };

        drop(row_group_data);

        let live_columns = live_df_filtered.into_columns();

        let mut dead_cols = Vec::with_capacity(self.non_predicate_field_indices.len());
        for fut in task_handles {
            dead_cols.extend(fut.await?);
        }

        let mut merged = live_columns;
        merged.extend(dead_cols);
        let df = unsafe { DataFrame::new_unchecked(expected_num_rows, merged) };
        Ok(df)
    }
}

fn decode_column_prefiltered(
    arrow_field: &ArrowField,
    row_group_data: &RowGroupData,
    mask: &BooleanChunked,
    mask_bitmap: &Bitmap,
    expected_num_rows: usize,
) -> PolarsResult<Column> {
    let Some(iter) = row_group_data
        .row_group_metadata
        .columns_under_root_iter(&arrow_field.name)
    else {
        return Ok(Column::full_null(
            arrow_field.name.clone(),
            expected_num_rows,
            &DataType::from_arrow_field(arrow_field),
        ));
    };

    let columns_to_deserialize = iter
        .map(|col_md| {
            let byte_range = col_md.byte_range();

            (
                col_md,
                row_group_data
                    .fetched_bytes
                    .get_range(byte_range.start as usize..byte_range.end as usize),
            )
        })
        .collect::<Vec<_>>();

    let prefilter = !arrow_field.dtype.is_nested();

    let deserialize_filter =
        prefilter.then(|| polars_parquet::read::Filter::Mask(mask_bitmap.clone()));

    let (array, _) = polars_io::prelude::_internal::to_deserializer(
        columns_to_deserialize,
        arrow_field.clone(),
        deserialize_filter,
    )?;

    let mut series = Series::try_from((arrow_field, array))?;

    if let Some(col_idxs) = row_group_data
        .row_group_metadata
        .columns_idxs_under_root_iter(&arrow_field.name)
    {
        if col_idxs.len() == 1 {
            try_set_sorted_flag(&mut series, col_idxs[0], &row_group_data.sorting_map);
        }
    }

    let series = if !prefilter {
        series.filter(mask)?
    } else {
        series
    };

    assert_eq!(series.len(), expected_num_rows);

    Ok(series.into_column())
}

mod tests {
    #[cfg(test)]
    use super::{
        Bitmap, Issue28304AdaptiveState, Issue28304PredicateObservation, Issue28304RollingLevel,
    };

    #[cfg(test)]
    fn rolling_observations() -> Vec<Issue28304PredicateObservation> {
        vec![
            Issue28304PredicateObservation {
                field_index: 0,
                mask: Bitmap::from_iter([
                    true, false, false, false, false, false, false, false, false, false,
                ]),
                elapsed_ns: 100,
            },
            Issue28304PredicateObservation {
                field_index: 1,
                mask: Bitmap::from_iter([
                    false, true, false, false, false, false, false, false, false, false,
                ]),
                elapsed_ns: 100,
            },
            Issue28304PredicateObservation {
                field_index: 2,
                mask: Bitmap::from_iter([true; 10]),
                elapsed_ns: 1_000,
            },
        ]
    }

    #[test]
    fn test_issue_28304_rolling_policy_progression() {
        let state = Issue28304AdaptiveState::default();
        let predicates = [0, 1, 2];

        let cold = state.choose_rolling(&predicates, 1, Issue28304RollingLevel::Marginal);
        assert_eq!(cold.deferred, None);
        assert_eq!(cold.reason, "cold_start");

        for _ in 0..6 {
            state.observe_all_at_once(10, &rolling_observations());
        }

        let marginal = state.choose_rolling(&predicates, 1, Issue28304RollingLevel::Marginal);
        assert_eq!(marginal.deferred, Some(2));

        let joint = state.choose_rolling(&predicates, 1, Issue28304RollingLevel::Joint);
        assert_eq!(joint.deferred, Some(2));

        let cost = state.choose_rolling(&predicates, 1, Issue28304RollingLevel::Cost);
        assert_eq!(cost.deferred, Some(2));

        let task_supply = state.choose_rolling(&predicates, 16, Issue28304RollingLevel::TaskSupply);
        assert_eq!(task_supply.deferred, None);
        assert_eq!(task_supply.reason, "insufficient_task_supply");

        let observations = rolling_observations();
        state.observe_staged_prefix(10, 2, 0.9, &observations[..2]);
        let drift = state.choose_rolling(&predicates, 1, Issue28304RollingLevel::Joint);
        assert_eq!(drift.deferred, None);
        assert_eq!(drift.reason, "prefix_not_selective");
    }

    #[test]
    fn test_calc_cols_per_thread() {
        use super::calc_cols_per_thread;

        assert_eq!(
            [
                calc_cols_per_thread(0, 5),
                calc_cols_per_thread(1, 5),
                calc_cols_per_thread(2, 5),
                calc_cols_per_thread(3, 5),
                calc_cols_per_thread(4, 5),
                calc_cols_per_thread(5, 5),
            ],
            [usize::MAX, 5, 2, 2, 1, 1]
        );

        assert_eq!(
            [
                calc_cols_per_thread(11_184_810, 16_777_216),
                calc_cols_per_thread(11_184_811, 16_777_216),
            ],
            [2, 1]
        );

        assert_eq!(
            [
                calc_cols_per_thread(0, 0),
                calc_cols_per_thread(0, 99),
                calc_cols_per_thread(99, 0),
                calc_cols_per_thread(99, 99),
            ],
            [usize::MAX, usize::MAX, 1, 1],
        )
    }
}
