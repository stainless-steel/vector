//! Vector database allowing for efficient search of nearest neighbors.
//!
//! The approach is described in “[FANN: Vector Search in 200 Lines of Rust][1]” by Nikhil Garg and
//! Navya Mehta.
//!
//! # Example
//!
//! ```
//! use vector::Index;
//!
//! let vectors = vec![
//!     [4.0, 2.0],
//!     [5.0, 7.0],
//!     [2.0, 9.0],
//!     [7.0, 8.0],
//! ];
//! let index = Index::build(&vectors, 1, 1, 42);
//!
//! let query = [5.0, 5.0];
//! let (indices, distances): (Vec<_>, Vec<_>) = index
//!     .search(&vectors, &query, 2)
//!     .into_iter()
//!     .unzip();
//! assert_eq!(indices, &[1, 0]);
//! ```
//!
//! [1]: https://fennel.ai/blog/vector-search-in-200-lines-of-rust/

use std::collections::BTreeSet;

/// An index.
pub struct Index<const N: usize> {
    roots: Vec<Node<N>>,
}

/// A vector.
pub type Vector<const N: usize> = [f32; N];

enum Node<const N: usize> {
    Branch(Box<Branch<N>>),
    Leaf(Box<Leaf>),
}

struct Branch<const N: usize> {
    plane: Plane<N>,
    above: Node<N>,
    below: Node<N>,
}

type Leaf = Vec<usize>;

struct Plane<const N: usize> {
    normal: Vector<N>,
    offset: f32,
}

impl<const N: usize> Index<N> {
    /// Build an index.
    ///
    /// The forest size is the number of trees built internally, and the leaf size is the maximum
    /// number of vectors a leaf can have without being split. Both arguments should be greater or
    /// equal to one.
    pub fn build(vectors: &[Vector<N>], forest_size: usize, leaf_size: usize, seed: u64) -> Self {
        debug_assert!(forest_size >= 1);
        debug_assert!(leaf_size >= 1);
        let mut source = random::default(seed);
        let indices = unique(vectors);
        let roots = (0..forest_size)
            .map(|_| Node::build(vectors, &indices, leaf_size, &mut source))
            .collect();
        Self { roots }
    }

    /// Find `count` vectors close to `query`.
    ///
    /// The vectors should be the same as the ones passed to `build`.
    pub fn search(
        &self,
        vectors: &[Vector<N>],
        query: &Vector<N>,
        count: usize,
    ) -> Vec<(usize, f32)> {
        let mut indices = BTreeSet::new();
        for root in self.roots.iter() {
            search(root, query, count, &mut indices);
        }
        let mut pairs = indices
            .into_iter()
            .map(|index| (index, distance(&vectors[index], query)))
            .collect::<Vec<_>>();
        pairs.sort_by(|one, other| one.1.partial_cmp(&other.1).unwrap());
        pairs.truncate(count);
        pairs
    }
}

impl<const N: usize> Node<N> {
    fn build<T: random::Source>(
        vectors: &[Vector<N>],
        indices: &[usize],
        leaf_size: usize,
        source: &mut T,
    ) -> Self {
        if indices.len() <= leaf_size {
            return Self::Leaf(Box::new(indices.to_vec()));
        }
        let (plane, above, below) = Plane::build(vectors, indices, source);
        let above = Self::build(vectors, &above, leaf_size, source);
        let below = Self::build(vectors, &below, leaf_size, source);
        Self::Branch(Box::new(Branch::<N> {
            plane,
            above,
            below,
        }))
    }
}

impl<const N: usize> Plane<N> {
    fn build<T: random::Source>(
        vectors: &[Vector<N>],
        indices: &[usize],
        source: &mut T,
    ) -> (Self, Vec<usize>, Vec<usize>) {
        debug_assert!(vectors.len() > 1);
        let i = source.read::<usize>() % indices.len();
        let mut j = i;
        while i == j {
            j = source.read::<usize>() % indices.len();
        }
        let one = &vectors[indices[i]];
        let other = &vectors[indices[j]];
        let normal = subtract(other, one);
        let offset = -product(&normal, &average(one, other));
        let plane = Plane::<N> { normal, offset };
        let (above, below) = indices
            .iter()
            .partition(|index| plane.is_above(&vectors[**index]));
        (plane, above, below)
    }

    fn is_above(&self, vector: &Vector<N>) -> bool {
        product(&self.normal, vector) + self.offset > 0.0
    }
}

fn average<const N: usize>(one: &Vector<N>, other: &Vector<N>) -> Vector<N> {
    one.iter()
        .zip(other)
        .map(|(one, other)| (one + other) / 2.0)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

fn distance<const N: usize>(one: &Vector<N>, other: &Vector<N>) -> f32 {
    one.iter()
        .zip(other)
        .map(|(one, other)| (one - other).powi(2))
        .sum()
}

fn product<const N: usize>(one: &Vector<N>, other: &Vector<N>) -> f32 {
    one.iter().zip(other).map(|(one, other)| one * other).sum()
}

fn search<const N: usize>(
    root: &Node<N>,
    vector: &Vector<N>,
    count: usize,
    indices: &mut BTreeSet<usize>,
) {
    match root {
        Node::Branch(node) => {
            let (primary, secondary) = if node.plane.is_above(vector) {
                (&node.above, &node.below)
            } else {
                (&node.below, &node.above)
            };
            search(primary, vector, count, indices);
            if indices.len() < count {
                search(secondary, vector, count, indices);
            }
        }
        Node::Leaf(node) => {
            for index in node.iter() {
                if indices.len() < count {
                    indices.insert(*index);
                } else {
                    break;
                }
            }
        }
    }
}

fn subtract<const N: usize>(one: &Vector<N>, other: &Vector<N>) -> Vector<N> {
    one.iter()
        .zip(other)
        .map(|(one, other)| one - other)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

fn unique<const N: usize>(vectors: &[Vector<N>]) -> Vec<usize> {
    let mut indices = Vec::with_capacity(vectors.len());
    let mut seen = BTreeSet::default();
    for (index, vector) in vectors.iter().enumerate() {
        let key: [u32; N] = vector
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        if !seen.contains(&key) {
            seen.insert(key);
            indices.push(index);
        }
    }
    indices
}

#[cfg(test)]
mod tests {
    use super::{Index, Plane};

    #[test]
    fn index() {
        let vectors = vec![
            [4.0, 2.0], // A1
            [5.0, 7.0], // B1
            [2.0, 9.0], // A2
            [7.0, 8.0], // B2
            [1.0, 3.0],
            [4.0, 10.0],
        ];
        let cases = vec![
            ([5.0, 0.0], 1, vec![0]),
            ([0.0, 0.0], 10, vec![4, 0, 1, 2, 3, 5]),
            ([5.0, 10.0], 1, vec![5]),
            ([7.0, 8.0], 2, vec![3, 1]),
        ];

        let index = Index::build(&vectors, 1, 1, 42);
        for (query, count, indices) in cases {
            assert_eq!(
                index
                    .search(&vectors, &query, count)
                    .into_iter()
                    .map(|(index, _)| index)
                    .collect::<Vec<_>>(),
                indices,
            );
        }
    }

    #[test]
    fn plane() {
        let mut source = random::default(25);
        let vectors = vec![
            [4.0, 2.0], // A1
            [5.0, 7.0], // B1
            [2.0, 9.0], // A2
            [7.0, 8.0], // B2
        ];
        let indices = (0..vectors.len()).collect::<Vec<_>>();
        let (plane, above, below) = Plane::build(&vectors, &indices, &mut source);
        assert::close(&plane.normal, &[1.0, 5.0], 1e-6);
        assert::close(plane.offset, -27.0, 1e-6);
        assert_eq!(above, &[1, 2, 3]);
        assert_eq!(below, &[0]);
    }
}
