//! Deterministic minimum-cost one-to-one assignment for radar tracking.

use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AssignmentResult {
    pub rows: Vec<usize>,
    pub columns: Vec<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AssignmentError {
    InvalidCostShape {
        row_count: usize,
        column_count: usize,
        data_length: usize,
    },
    NonFiniteCost,
}

impl fmt::Display for AssignmentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidCostShape {
                row_count,
                column_count,
                data_length,
            } => write!(
                formatter,
                "Assignment cost shape ({row_count}, {column_count}) does not match {data_length} values."
            ),
            Self::NonFiniteCost => write!(formatter, "Assignment costs must be finite."),
        }
    }
}

impl std::error::Error for AssignmentError {}

pub fn linear_sum_assignment(
    costs: &[f64],
    row_count: usize,
    column_count: usize,
) -> Result<AssignmentResult, AssignmentError> {
    validate_costs(costs, row_count, column_count)?;
    if row_count == 0 || column_count == 0 {
        return Ok(AssignmentResult {
            rows: Vec::new(),
            columns: Vec::new(),
        });
    }

    if row_count <= column_count {
        let columns = assign_rows_to_columns(costs, row_count, column_count);
        return Ok(AssignmentResult {
            rows: (0..row_count).collect(),
            columns,
        });
    }

    let transposed = transpose_costs(costs, row_count, column_count);
    let assigned_rows = assign_rows_to_columns(&transposed, column_count, row_count);
    let mut pairs = assigned_rows
        .into_iter()
        .enumerate()
        .map(|(column, row)| (row, column))
        .collect::<Vec<_>>();
    pairs.sort_unstable_by_key(|(row, _)| *row);
    Ok(AssignmentResult {
        rows: pairs.iter().map(|(row, _)| *row).collect(),
        columns: pairs.into_iter().map(|(_, column)| column).collect(),
    })
}

fn validate_costs(
    costs: &[f64],
    row_count: usize,
    column_count: usize,
) -> Result<(), AssignmentError> {
    let expected_length =
        row_count
            .checked_mul(column_count)
            .ok_or(AssignmentError::InvalidCostShape {
                row_count,
                column_count,
                data_length: costs.len(),
            })?;
    if costs.len() != expected_length {
        return Err(AssignmentError::InvalidCostShape {
            row_count,
            column_count,
            data_length: costs.len(),
        });
    }
    if costs.iter().any(|cost| !cost.is_finite()) {
        return Err(AssignmentError::NonFiniteCost);
    }
    Ok(())
}

fn assign_rows_to_columns(costs: &[f64], row_count: usize, column_count: usize) -> Vec<usize> {
    let mut row_potentials = vec![0.0_f64; row_count + 1];
    let mut column_potentials = vec![0.0_f64; column_count + 1];
    let mut column_matches = vec![0_usize; column_count + 1];
    let mut predecessor_columns = vec![0_usize; column_count + 1];

    for row in 1..=row_count {
        column_matches[0] = row;
        let mut current_column = 0;
        let mut minimum_costs = vec![f64::INFINITY; column_count + 1];
        let mut used_columns = vec![false; column_count + 1];
        loop {
            used_columns[current_column] = true;
            let current_row = column_matches[current_column];
            let mut delta = f64::INFINITY;
            let mut next_column = 0;
            for column in 1..=column_count {
                if used_columns[column] {
                    continue;
                }
                let reduced_cost = costs[(current_row - 1) * column_count + (column - 1)]
                    - row_potentials[current_row]
                    - column_potentials[column];
                if reduced_cost < minimum_costs[column] {
                    minimum_costs[column] = reduced_cost;
                    predecessor_columns[column] = current_column;
                }
                if minimum_costs[column] < delta {
                    delta = minimum_costs[column];
                    next_column = column;
                }
            }
            for column in 0..=column_count {
                if used_columns[column] {
                    row_potentials[column_matches[column]] += delta;
                    column_potentials[column] -= delta;
                } else {
                    minimum_costs[column] -= delta;
                }
            }
            current_column = next_column;
            if column_matches[current_column] == 0 {
                break;
            }
        }
        loop {
            let previous_column = predecessor_columns[current_column];
            column_matches[current_column] = column_matches[previous_column];
            current_column = previous_column;
            if current_column == 0 {
                break;
            }
        }
    }

    let mut assignments = vec![0_usize; row_count];
    for (column, &row) in column_matches.iter().enumerate().skip(1) {
        if row != 0 {
            assignments[row - 1] = column - 1;
        }
    }
    assignments
}

fn transpose_costs(costs: &[f64], row_count: usize, column_count: usize) -> Vec<f64> {
    let mut transposed = vec![0.0_f64; costs.len()];
    for row in 0..row_count {
        for column in 0..column_count {
            transposed[column * row_count + row] = costs[row * column_count + column];
        }
    }
    transposed
}

#[cfg(test)]
mod tests {
    use super::{AssignmentError, linear_sum_assignment};

    #[test]
    fn finds_minimum_cost_square_and_rectangular_matchings() {
        let square =
            linear_sum_assignment(&[4.0, 1.0, 3.0, 2.0, 0.0, 5.0, 3.0, 2.0, 2.0], 3, 3).unwrap();
        assert_eq!(square.rows, [0, 1, 2]);
        assert_eq!(square.columns, [1, 0, 2]);

        let tall = linear_sum_assignment(&[10.0, 1.0, 1.0, 10.0, 2.0, 2.0], 3, 2).unwrap();
        assert_eq!(tall.rows, [0, 1]);
        assert_eq!(tall.columns, [1, 0]);
    }

    #[test]
    fn breaks_equal_cost_ties_by_lowest_indices() {
        let result = linear_sum_assignment(&[1.0; 6], 2, 3).unwrap();

        assert_eq!(result.rows, [0, 1]);
        assert_eq!(result.columns, [0, 1]);
    }

    #[test]
    fn accepts_empty_rectangular_costs_and_rejects_invalid_values() {
        let empty = linear_sum_assignment(&[], 0, 3).unwrap();
        assert!(empty.rows.is_empty());
        assert!(empty.columns.is_empty());

        let error = linear_sum_assignment(&[1.0, f64::INFINITY], 1, 2).unwrap_err();
        assert_eq!(error, AssignmentError::NonFiniteCost);
    }
}
