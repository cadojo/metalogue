# ⚔️ `metalogue`

*Notes and executable examples for programming parallel applications with Apple Silicon.*

> [!NOTE]
> This project is not affiliated with Apple.

## Motivation

Using the Metal API in Rust necessarily involves dereferencing raw pointers, which requires `unsafe` code.
Each instance of `unsafe` code adds developer friction via lines of code, mental effort, and possible memory bugs.
If safe abstractions around the necessary `unsafe` code existed, developers could more easily explore GPU programming on Apple Silicon.

## History

This project --- a learning tool for the author --- originally aimed to provide simple, safe abstractions around `unsafe` code using Apple's Metal API.
The Metal API is exposed through Objective-C and Swift.
The [`objc2`](https://docs.rs/objc2/latest/objc2/) [project](https://github.com/madsmtm/objc2) exposes Objective-C interfaces with Rust bindings, including the Metal API via [`objc2_metal`](https://docs.rs/objc2-metal/latest/objc2_metal/).
With `metalogue-v0.0.1`, developers can compile kernels, allocate memory with various storage specifications, specify threadgroup and grid sizes, and submit command buffers to the GPU for execution without any `unsafe` code.

As it turns out, Apple has already more than met this need (and more) with its array framework [MLX](https://opensource.apple.com/projects/mlx).
Via MLX, developers in Python, Swift, Objective-C, C, and any other language which interacts with the C API, can stream instructions to Apple Silicon CPU and GPU devices and utilize algorithm-specific optimization packages.
**There's no need to develop safe software abstractions for Apple Silicon hardware: Apple's research team has already done this for us.**
Still, to **benchmark** kernels on Apple Silicon, it helps to hand-write kernel code.
For this purpose, `metalogue` provides a Rust API with Python bindings for compiling and executing Metal kernels.

To continue growing as a GPU programmer, and to use that effort to build a resource that's (hopefully) helpful to the larger device programming community, this project's content and purpose has changed.
Instead of developing safe abstractions, this project will aim to *use* safe abstractions provided by `mlx` and `metalogue` to apply lessons from an essential textbook --- [Programming Massively Parallel Processors](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0) --- directly to Apple Silicon.
Examples will be available as source-code, and within a note-set that accompanies the reference textbook for Apple Silicon programmers.

## Usage

Today, very little example content exists.
Please follow along as the note-set is written here, with rendered content available at [metalogue.loopy.codes](https://metalogue.loopy.codes).
Integration tests and benchmarks (under [`tests`](/tests/) and [`benches`](/benches/) respectively) currently provide the best summary of available features.

You may ask: why `loopy.codes`?
Shouldn't GPU kernels explicitly avoid large loops in favor of dispatching across many threads?
Isn't that the whole point of parallel programming?
The name `loopy.codes` is a playful reference to outputs of astronomy software: *loopy* orbits.
For more information about the author, and the astronomy codes the author maintains, see the author's website: [loopy.codes](https://loopy.codes).
