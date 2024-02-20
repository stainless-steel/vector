# Vector [![Package][package-img]][package-url] [![Documentation][documentation-img]][documentation-url] [![Build][build-img]][build-url]

The package provides a vector database allowing for efficient search of nearest
neighbors.

## Example

```rust
use vector::Index;

let vectors = vec![
    [4.0, 2.0],
    [5.0, 7.0],
    [2.0, 9.0],
    [7.0, 8.0],
];
let index = Index::build(&vectors, 1, 1, 42);

let query = [5.0, 5.0];
let (indices, distances): (Vec<_>, Vec<_>) = index
    .search(&vectors, &query, 2)
    .into_iter()
    .unzip();
assert_eq!(indices, &[1, 0]);
```

## Contribution

Your contribution is highly appreciated. Do not hesitate to open an issue or a
pull request. Note that any contribution submitted for inclusion in the project
will be licensed according to the terms given in [LICENSE.md](LICENSE.md).

[build-img]: https://github.com/stainless-steel/vector/workflows/build/badge.svg
[build-url]: https://github.com/stainless-steel/vector/actions/workflows/build.yml
[documentation-img]: https://docs.rs/vector/badge.svg
[documentation-url]: https://docs.rs/vector
[package-img]: https://img.shields.io/crates/v/vector.svg
[package-url]: https://crates.io/crates/vector
