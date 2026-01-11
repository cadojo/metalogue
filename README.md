# ⚔️ `metalogue`

*Today: vaporware; tomorrow: safe and simple abstractions around Apple's Metal API.*

> [!CAUTION]
> This project is currently experimental and unstable!

## Motivation

Using the Metal API in Rust necessarily involves dereferencing raw pointers, which requires `unsafe` code.
Each instance of `unsafe` code adds developer friction via lines of code, mental effort, and possible memory bugs.
If safe abstractions around the necessary `unsafe` code existed, developers could more easily explore GPU programming on Apple Silicon.

This project provides simple, safe abstractions around `unsafe` code using Apple's Metal API.
The Metal API is exposed through Objective-C and Swift.
The [`objc2`](https://docs.rs/objc2/latest/objc2/) [project](https://github.com/madsmtm/objc2) exposes Objective-C interfaces with Rust bindings, including the Metal API via [`objc2_metal`](https://docs.rs/objc2-metal/latest/objc2_metal/).
With `metalogue`, developers can compile kernels, allocate memory with various storage specifications, specify threadgroup and grid sizes, and submit command buffers to the GPU for execution without any `unsafe` code.

## Usage

More documentation is en route!
