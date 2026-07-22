use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock};
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
    all_at_once_count: AtomicUsize,
    all_at_once_ns: AtomicU64,
    staged_count: AtomicUsize,
    staged_ns: AtomicU64,
}

impl Issue28304AdaptiveState {
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

        let mut current_mask = Bitmap::new_with_value(true, projection_height);
        let mut live_columns: Vec<(usize, Column)> = Vec::new();

        for (stage_index, stage) in stages.into_iter().enumerate() {
            let stage_len = stage.len();
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
                                        Ok((field_index, col, pred_true_mask))
                                    })
                                    .collect::<PolarsResult<UnitVec<_>>>()
                            }
                        }),
                )
            };

            let mut stage_columns = Vec::with_capacity(stage_len);
            let mut stage_masks = Vec::with_capacity(stage_len);
            for fut in task_handles {
                for (field_index, column, mask) in fut.await? {
                    stage_columns.push((field_index, column));
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
        if let Some(stages) = self.issue_28304_stages() {
            if std::env::var("POLARS_ISSUE_28304_ADAPTIVE").as_deref() == Ok("1") {
                let (issued, staged) = self.issue_28304_adaptive_state.choose_staged();
                let started = Instant::now();
                let result = if staged {
                    self.row_group_data_to_df_prefiltered_staged(row_group_data, stages)
                        .await
                } else {
                    self.row_group_data_to_df_prefiltered_all_at_once(row_group_data)
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

        self.row_group_data_to_df_prefiltered_all_at_once(row_group_data)
            .await
    }

    async fn row_group_data_to_df_prefiltered_all_at_once(
        &self,
        row_group_data: RowGroupData,
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
                                    let projection =
                                        &projected_arrow_fields[predicate_field_indices[i]];

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

                                    Ok((col, pred_true_mask))
                                })
                                .collect::<PolarsResult<UnitVec<_>>>()
                        }
                    }),
            )
        };

        for fut in task_handles {
            for (c, m) in fut.await? {
                live_columns.push(c);
                masks.push(m);
            }
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
