//! Ordered set of indices for atom/residue selection membership.
//!
//! An [`OrderedSet`] represents either a contiguous interval (for dense ranges)
//! or a sorted array of unique indices (for sparse selections).

use std::sync::Arc;

/// Ordered set of indices — either a contiguous interval or sorted array.
///
/// The `Interval` variant provides O(1) membership tests for contiguous ranges.
/// The `Sorted` variant provides O(log n) membership tests for arbitrary selections.
#[derive(Clone, Debug, PartialEq)]
pub enum OrderedSet {
    /// Contiguous range [start, end) — O(1) membership test
    Interval { start: u32, end: u32 },
    /// Sorted unique indices — O(log n) membership test
    Sorted(Arc<[u32]>),
}

impl OrderedSet {
    /// Create a contiguous interval [start, end).
    ///
    /// # Panics
    /// Panics if `start > end`.
    pub fn interval(start: u32, end: u32) -> Self {
        assert!(start <= end, "interval start must be <= end");
        OrderedSet::Interval { start, end }
    }

    /// Create a `Sorted` set from a `Vec<u32>`.
    ///
    /// The input must already be sorted and contain no duplicates.
    /// This function panics if either condition is violated, keeping the
    /// invariant that `Sorted` always holds valid data.
    ///
    /// If you need to accept arbitrary input, sort and dedup before calling:
    /// ```
    /// # use ferritin_core::data::ordered_set::OrderedSet;
    /// let mut v = vec![3u32, 1, 2, 1];
    /// v.sort_unstable();
    /// v.dedup();
    /// let set = OrderedSet::from_sorted(v);
    /// ```
    pub fn from_sorted(indices: Vec<u32>) -> Self {
        // Validate: must be strictly increasing (sorted + no duplicates)
        for window in indices.windows(2) {
            assert!(
                window[0] < window[1],
                "from_sorted requires strictly sorted (no duplicates) input; \
                 found {} >= {}",
                window[0],
                window[1]
            );
        }
        OrderedSet::Sorted(indices.into())
    }

    /// Returns the number of elements in the set.
    pub fn len(&self) -> usize {
        match self {
            OrderedSet::Interval { start, end } => (end - start) as usize,
            OrderedSet::Sorted(v) => v.len(),
        }
    }

    /// Returns `true` if the set contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns `true` if `idx` is a member of this set.
    pub fn contains(&self, idx: u32) -> bool {
        match self {
            OrderedSet::Interval { start, end } => idx >= *start && idx < *end,
            OrderedSet::Sorted(v) => v.binary_search(&idx).is_ok(),
        }
    }

    /// Iterate over the elements of this set in ascending order.
    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        match self {
            OrderedSet::Interval { start, end } => {
                // Use a box to unify the two iterator types
                let iter: Box<dyn Iterator<Item = u32> + '_> =
                    Box::new(*start..*end);
                iter
            }
            OrderedSet::Sorted(v) => {
                let iter: Box<dyn Iterator<Item = u32> + '_> =
                    Box::new(v.iter().copied());
                iter
            }
        }
    }

    /// Compute the union of this set and `other`.
    ///
    /// Uses a merge-style algorithm on two sorted iterators — O(n + m).
    /// Returns an `OrderedSet::Sorted`.
    pub fn union(&self, other: &Self) -> Self {
        let mut result = Vec::with_capacity(self.len() + other.len());
        let mut left = self.iter().peekable();
        let mut right = other.iter().peekable();

        loop {
            match (left.peek(), right.peek()) {
                (None, None) => break,
                (Some(_), None) => {
                    result.extend(left);
                    break;
                }
                (None, Some(_)) => {
                    result.extend(right);
                    break;
                }
                (Some(&l), Some(&r)) => {
                    if l < r {
                        result.push(l);
                        left.next();
                    } else if r < l {
                        result.push(r);
                        right.next();
                    } else {
                        // equal: include once, advance both
                        result.push(l);
                        left.next();
                        right.next();
                    }
                }
            }
        }

        OrderedSet::Sorted(result.into())
    }

    /// Compute the intersection of this set and `other`.
    ///
    /// Uses a merge-style algorithm — O(n + m).
    /// Returns an `OrderedSet::Sorted`.
    pub fn intersection(&self, other: &Self) -> Self {
        let mut result = Vec::new();
        let mut left = self.iter().peekable();
        let mut right = other.iter().peekable();

        loop {
            match (left.peek(), right.peek()) {
                (None, _) | (_, None) => break,
                (Some(&l), Some(&r)) => {
                    if l == r {
                        result.push(l);
                        left.next();
                        right.next();
                    } else if l < r {
                        left.next();
                    } else {
                        right.next();
                    }
                }
            }
        }

        OrderedSet::Sorted(result.into())
    }

    /// Compute the set difference `self \ other` (elements in self but not other).
    ///
    /// Uses a merge-style algorithm — O(n + m).
    /// Returns an `OrderedSet::Sorted`.
    pub fn difference(&self, other: &Self) -> Self {
        let mut result = Vec::new();
        let mut left = self.iter().peekable();
        let mut right = other.iter().peekable();

        loop {
            match left.peek() {
                None => break,
                Some(&l) => match right.peek() {
                    None => {
                        result.extend(left);
                        break;
                    }
                    Some(&r) => {
                        if l < r {
                            result.push(l);
                            left.next();
                        } else if l == r {
                            left.next();
                            right.next();
                        } else {
                            right.next();
                        }
                    }
                },
            }
        }

        OrderedSet::Sorted(result.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_contains() {
        let s = OrderedSet::interval(5, 10);
        assert!(s.contains(5));
        assert!(s.contains(9));
        assert!(!s.contains(4));
        assert!(!s.contains(10));
    }

    #[test]
    fn test_interval_len() {
        let s = OrderedSet::interval(3, 8);
        assert_eq!(s.len(), 5);

        let empty = OrderedSet::interval(4, 4);
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }

    #[test]
    fn test_sorted_contains() {
        let s = OrderedSet::from_sorted(vec![1, 3, 5, 7, 9]);
        assert!(s.contains(1));
        assert!(s.contains(5));
        assert!(s.contains(9));
        assert!(!s.contains(0));
        assert!(!s.contains(2));
        assert!(!s.contains(10));
    }

    #[test]
    fn test_ordered_set_intersection() {
        let a = OrderedSet::from_sorted(vec![1, 3, 5, 7, 9]);
        let b = OrderedSet::from_sorted(vec![2, 3, 5, 8]);
        let result = a.intersection(&b);
        assert_eq!(result, OrderedSet::from_sorted(vec![3, 5]));
    }

    #[test]
    fn test_ordered_set_union() {
        let a = OrderedSet::from_sorted(vec![1, 3, 5]);
        let b = OrderedSet::from_sorted(vec![2, 4, 6]);
        let result = a.union(&b);
        assert_eq!(result, OrderedSet::from_sorted(vec![1, 2, 3, 4, 5, 6]));
    }

    #[test]
    fn test_ordered_set_difference() {
        let a = OrderedSet::from_sorted(vec![1, 3, 5, 7, 9]);
        let b = OrderedSet::from_sorted(vec![3, 5]);
        let result = a.difference(&b);
        assert_eq!(result, OrderedSet::from_sorted(vec![1, 7, 9]));
    }

    #[test]
    fn test_interval_iter() {
        let s = OrderedSet::interval(2, 6);
        let collected: Vec<u32> = s.iter().collect();
        assert_eq!(collected, vec![2, 3, 4, 5]);
    }

    #[test]
    fn test_sorted_iter() {
        let s = OrderedSet::from_sorted(vec![10, 20, 30]);
        let collected: Vec<u32> = s.iter().collect();
        assert_eq!(collected, vec![10, 20, 30]);
    }

    #[test]
    fn test_empty_intersection() {
        let a = OrderedSet::from_sorted(vec![1, 3, 5]);
        let b = OrderedSet::from_sorted(vec![2, 4, 6]);
        let result = a.intersection(&b);
        assert!(result.is_empty());
    }

    #[test]
    #[should_panic(expected = "from_sorted requires strictly sorted")]
    fn test_from_sorted_panics_on_duplicate() {
        // Documented behaviour: from_sorted PANICS on non-sorted or duplicate input.
        // Callers that need to accept arbitrary input should sort+dedup first.
        OrderedSet::from_sorted(vec![1, 2, 2, 3]);
    }

    #[test]
    #[should_panic(expected = "from_sorted requires strictly sorted")]
    fn test_from_sorted_panics_on_unsorted() {
        OrderedSet::from_sorted(vec![3, 1, 2]);
    }

    /// Demonstrate the recommended pattern for arbitrary (unsorted/duplicate) input.
    #[test]
    fn test_from_sorted_sort_dedup_pattern() {
        let mut v = vec![3u32, 1, 2, 1, 3];
        v.sort_unstable();
        v.dedup();
        let s = OrderedSet::from_sorted(v);
        assert_eq!(s.len(), 3);
        assert!(s.contains(1));
        assert!(s.contains(2));
        assert!(s.contains(3));
    }

    #[test]
    fn test_union_with_overlap() {
        let a = OrderedSet::from_sorted(vec![1, 2, 3]);
        let b = OrderedSet::from_sorted(vec![2, 3, 4]);
        let result = a.union(&b);
        assert_eq!(result, OrderedSet::from_sorted(vec![1, 2, 3, 4]));
    }

    #[test]
    fn test_interval_and_sorted_intersection() {
        // Mixed-variant set algebra: Interval ∩ Sorted
        let a = OrderedSet::interval(3, 8); // {3,4,5,6,7}
        let b = OrderedSet::from_sorted(vec![1, 5, 7, 9]);
        let result = a.intersection(&b);
        assert_eq!(result, OrderedSet::from_sorted(vec![5, 7]));
    }
}
