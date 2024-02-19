//! Vector database.

// The implementation is based on:
// https://fennel.ai/blog/vector-search-in-200-lines-of-rust/

use std::collections::{HashMap, HashSet};

/// An index.
pub struct Index<const N: usize> {
    vectors: Vec<Vector<N>>,
    roots: Vec<Node<N>>,
}

/// A vector.
#[derive(Clone, Copy)]
pub struct Vector<const N: usize>(pub [f32; N]);

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct Key<const N: usize>([u32; N]);

struct Plane<const N: usize> {
    normal: Vector<N>,
    offset: f32,
}

enum Node<const N: usize> {
    Branch(Box<Branch<N>>),
    Leaf(Box<Leaf<N>>),
}

struct Branch<const N: usize> {
    plane: Plane<N>,
    above: Node<N>,
    below: Node<N>,
}

struct Leaf<const N: usize>(Vec<usize>);

impl<const N: usize> Index<N> {
    /// Build an index.
    pub fn build(vectors: Vec<Vector<N>>, forest_size: usize, leaf_size: usize, seed: u64) -> Self {
        let mut source = random::default(seed);
        let vectors = deduplicate(vectors);
        let indices = (0..vectors.len()).collect::<Vec<_>>();
        let roots = (0..forest_size)
            .map(|_| Node::build(&vectors, &indices, leaf_size, &mut source))
            .collect();
        Self { vectors, roots }
    }

    /// Search neighbor vectors.
    pub fn search(
        &self,
        vector: &Vector<N>,
        count: usize,
    ) -> impl Iterator<Item = (&Vector<N>, f32)> {
        let mut indices = HashSet::new();
        for root in self.roots.iter() {
            search(root, vector, count, &mut indices);
        }
        let mut pairs = indices
            .into_iter()
            .map(|index| (index, self.vectors[index].distance(vector)))
            .collect::<Vec<_>>();
        pairs.sort_by(|one, other| one.1.partial_cmp(&other.1).unwrap());
        pairs
            .into_iter()
            .take(count)
            .map(|(index, distance)| (&self.vectors[index], distance))
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
            return Self::Leaf(Box::new(Leaf::<N>(indices.to_vec())));
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

impl<const N: usize> Vector<N> {
    fn average(&self, other: &Self) -> Self {
        Self(
            self.0
                .iter()
                .zip(other.0)
                .map(|(one, other)| (one + other) / 2.0)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        )
    }

    fn distance(&self, other: &Self) -> f32 {
        self.0
            .iter()
            .zip(other.0)
            .map(|(one, other)| (one - other).powi(2))
            .sum()
    }

    fn product(&self, other: &Self) -> f32 {
        self.0
            .iter()
            .zip(other.0)
            .map(|(one, other)| one * other)
            .sum::<f32>()
    }

    fn subtract(&self, other: &Self) -> Self {
        Self(
            self.0
                .iter()
                .zip(other.0)
                .map(|(one, other)| one - other)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        )
    }

    fn as_key(&self) -> Key<N> {
        Key::<N>(
            self.0
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        )
    }
}

impl<const N: usize> Plane<N> {
    fn build<T: random::Source>(
        vectors: &[Vector<N>],
        indices: &[usize],
        source: &mut T,
    ) -> (Self, Vec<usize>, Vec<usize>) {
        debug_assert!(vectors.len() > 1);
        let i = source.read::<usize>();
        let mut j = i;
        while i == j {
            j = source.read::<usize>();
        }
        let normal = vectors[j].subtract(&vectors[i]);
        let offset = -normal.product(&vectors[i].average(&vectors[j]));
        let plane = Plane::<N> { normal, offset };
        let (above, below) = indices
            .iter()
            .cloned()
            .partition(|index| plane.is_above(&vectors[*index]));
        (plane, above, below)
    }

    fn is_above(&self, vector: &Vector<N>) -> bool {
        self.normal.product(vector) + self.offset > 0.0
    }
}

fn deduplicate<const N: usize>(vectors: Vec<Vector<N>>) -> Vec<Vector<N>> {
    vectors
        .into_iter()
        .map(|value| (value.as_key(), value))
        .collect::<HashMap<_, _>>()
        .into_values()
        .collect()
}

fn search<const N: usize>(
    root: &Node<N>,
    vector: &Vector<N>,
    count: usize,
    indices: &mut HashSet<usize>,
) -> usize {
    match root {
        Node::Branch(node) => {
            let (primary, secondary) = if node.plane.is_above(vector) {
                (&node.above, &node.below)
            } else {
                (&node.below, &node.above)
            };
            let mut found = search(primary, vector, count, indices);
            if found < count {
                found += search(secondary, vector, count - found, indices);
            }
            found
        }
        Node::Leaf(node) => {
            let found = std::cmp::min(count, node.0.len());
            for i in 0..found {
                indices.insert(node.0[i]);
            }
            found
        }
    }
}
