//! CSR-style segmentation primitive for grouping elements into segments.
//!
//! `Segmentation` stores an offset array where `offsets[i]..offsets[i+1]`
//! is the element range for segment `i`.  This is the same layout as a
//! Compressed Sparse Row (CSR) row-pointer array.

use std::ops::Range;

/// CSR-style segmentation: groups elements into segments via offset array.
///
/// `offsets.len() == n_segments + 1`; segment `i` spans
/// `offsets[i]..offsets[i+1]`.
#[derive(Clone, Debug, PartialEq)]
pub struct Segmentation {
    offsets: Vec<u32>, // len = n_segments + 1
}

impl Segmentation {
    /// Build a validated segmentation from pre-computed offsets.
    pub fn try_from_offsets(offsets: Vec<u32>) -> Result<Self, String> {
        if offsets.is_empty() {
            return Err("offsets must have at least one element".to_string());
        }
        if offsets[0] != 0 {
            return Err(format!("offsets must start at 0; found {}", offsets[0]));
        }
        for window in offsets.windows(2) {
            if window[0] > window[1] {
                return Err(format!(
                    "offsets must be monotonically non-decreasing; found {} > {}",
                    window[0], window[1]
                ));
            }
        }
        Ok(Self { offsets })
    }

    /// Build from pre-computed offsets (e.g., `[0, 5, 12, 20]` for 3 segments).
    ///
    /// # Panics
    /// Panics if `offsets` is empty (needs at least the sentinel `[0]`).
    pub fn from_offsets(offsets: Vec<u32>) -> Self {
        Self::try_from_offsets(offsets).expect("invalid segmentation offsets")
    }

    /// Build by detecting change-points in a key sequence.
    ///
    /// Each run of equal consecutive keys becomes one segment.
    ///
    /// # Example
    /// ```
    /// # use ferritin_core::data::Segmentation;
    /// let seg = Segmentation::from_change_points(["A","A","A","B","B","C"].iter().copied());
    /// assert_eq!(seg.count(), 3);
    /// ```
    pub fn from_change_points<K: PartialEq>(keys: impl Iterator<Item = K>) -> Self {
        let mut offsets: Vec<u32> = vec![0];
        let mut prev: Option<K> = None;
        let mut idx: u32 = 0;

        for key in keys {
            match &prev {
                None => {}
                Some(p) if *p == key => {}
                Some(_) => {
                    offsets.push(idx);
                }
            }
            prev = Some(key);
            idx += 1;
        }
        offsets.push(idx); // final sentinel
        Self { offsets }
    }

    /// Number of segments.
    pub fn count(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Total number of elements covered by the segmentation.
    pub fn element_count(&self) -> usize {
        self.offsets.last().copied().unwrap_or(0) as usize
    }

    /// Element range for segment `seg`.
    ///
    /// # Panics
    /// Panics if `seg >= self.count()`.
    pub fn segment(&self, seg: usize) -> Range<usize> {
        let start = self.offsets[seg] as usize;
        let end = self.offsets[seg + 1] as usize;
        start..end
    }

    /// O(log n) binary-search lookup: which segment contains element `elem`?
    ///
    /// Returns the segment index such that `segment(idx)` contains `elem`.
    ///
    /// # Panics
    /// Panics if `elem` is out of range (>= total element count).
    pub fn segment_of(&self, elem: usize) -> usize {
        assert!(
            elem < self.element_count(),
            "element {} is out of range (element_count={})",
            elem,
            self.element_count()
        );
        let elem_u32 = elem as u32;
        // Find the last offset that is <= elem. The partition point gives the
        // first index where offsets[i] > elem, so subtract 1.
        let pos = self.offsets.partition_point(|&o| o <= elem_u32);
        assert!(pos > 0, "element {} is out of range", elem);
        pos - 1
    }

    /// Iterate over all segment ranges in order.
    pub fn iter(&self) -> impl Iterator<Item = Range<usize>> + '_ {
        self.offsets
            .windows(2)
            .map(|w| (w[0] as usize)..(w[1] as usize))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segmentation_from_change_points() {
        let chain_ids = ["A", "A", "A", "B", "B", "C"];
        let seg = Segmentation::from_change_points(chain_ids.iter().copied());
        assert_eq!(seg.count(), 3);
        assert_eq!(seg.segment(0), 0..3);
        assert_eq!(seg.segment(1), 3..5);
        assert_eq!(seg.segment(2), 5..6);
    }

    #[test]
    fn test_segmentation_from_offsets() {
        let seg = Segmentation::from_offsets(vec![0, 3, 5, 6]);
        assert_eq!(seg.count(), 3);
        assert_eq!(seg.segment(0), 0..3);
        assert_eq!(seg.segment(1), 3..5);
        assert_eq!(seg.segment(2), 5..6);
    }

    #[test]
    fn test_segmentation_segment_of() {
        let seg = Segmentation::from_offsets(vec![0, 3, 5, 6]);
        // Segment 0: elements 0,1,2
        assert_eq!(seg.segment_of(0), 0);
        assert_eq!(seg.segment_of(1), 0);
        assert_eq!(seg.segment_of(2), 0);
        // Segment 1: elements 3,4
        assert_eq!(seg.segment_of(3), 1);
        assert_eq!(seg.segment_of(4), 1);
        // Segment 2: element 5
        assert_eq!(seg.segment_of(5), 2);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn test_segmentation_segment_of_rejects_end_sentinel() {
        Segmentation::from_offsets(vec![0, 3, 5]).segment_of(5);
    }

    #[test]
    fn test_segmentation_round_trip() {
        let chain_ids = ["A", "A", "A", "B", "B", "C"];
        let seg = Segmentation::from_change_points(chain_ids.iter().copied());

        // Build the expected mapping by hand
        let expected = [0usize, 0, 0, 1, 1, 2];
        for (elem, &exp_seg) in expected.iter().enumerate() {
            assert_eq!(
                seg.segment_of(elem),
                exp_seg,
                "elem {} should be in segment {}",
                elem,
                exp_seg
            );
        }
    }

    #[test]
    fn test_segmentation_single_segment() {
        let keys = ["X", "X", "X", "X"];
        let seg = Segmentation::from_change_points(keys.iter().copied());
        assert_eq!(seg.count(), 1);
        assert_eq!(seg.segment(0), 0..4);
        for elem in 0..4 {
            assert_eq!(seg.segment_of(elem), 0);
        }
    }

    #[test]
    fn test_segmentation_all_different() {
        let keys = ["A", "B", "C", "D"];
        let seg = Segmentation::from_change_points(keys.iter().copied());
        assert_eq!(seg.count(), 4);
        for i in 0..4 {
            assert_eq!(seg.segment(i), i..(i + 1));
            assert_eq!(seg.segment_of(i), i);
        }
    }

    #[test]
    fn test_segmentation_iter() {
        let seg = Segmentation::from_offsets(vec![0, 3, 5, 6]);
        let ranges: Vec<Range<usize>> = seg.iter().collect();
        assert_eq!(ranges, vec![0..3, 3..5, 5..6]);
    }

    #[test]
    #[should_panic(expected = "monotonically non-decreasing")]
    fn test_segmentation_from_offsets_non_monotonic_panics() {
        Segmentation::from_offsets(vec![0, 10, 5, 15]);
    }

    #[test]
    fn test_segmentation_try_from_offsets_rejects_nonzero_start() {
        assert_eq!(
            Segmentation::try_from_offsets(vec![2, 5]).unwrap_err(),
            "offsets must start at 0; found 2"
        );
    }
}
