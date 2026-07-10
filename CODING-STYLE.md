## Coding Guidelines

### Rust Code Style

- **Edition**: Rust 2024 (set in the workspace root)
- **Formatting**: Use the project's specific formatting rules:
  ```bash
  # Format all code
  cargo fmt --all
  ```
- **Naming**:
  - Modules: `snake_case`
  - Types: `PascalCase`
  - Traits: Verb-noun combinations
  - **Important**: Names must describe what code does, not implementation details or history
  - **Avoid**: Implementation details ("JsonParserImpl", "RedisWrapper"), temporal context ("NewAPI",
    "LegacyHandler"), pattern names unless they add clarity
- **Imports**: All `use` statements at top of file/module (avoid per-function imports)
- **Avoid deep nesting**: Rust code -- with its extensive use of matching over ADT -- can trend
  towards deeply nested code that becomes increasingly difficult to read. To avoid this there are a
  number of techniques:
  - **Early Returns**: Short-circuit out of your function on _negative_ conditions, leaving the
    _positive_ case for last. This helps the reader understand the codeflow and emphasizes the
    function's overall purpose.
    - Handle error cases and invalid conditions first
    - Return early with `?` operator for `Result`/`Option` types
    - Makes the "happy path" clear and uncluttered
  - **Let-else statements**: Use `let else` for conditional binding with early returns on failure
  - **Match guards and let-chains**: Put dependent pattern checks in `match` guards to avoid nested
    `if let` and separate the successful path from the failure cases.
  - **Avoid `else` branches**: Generally prefer early returns over `else` branches on `if`
    statements
  - **Factor out into separate functions**: Break complicated deeply nested blocks into smaller,
    focused functions

#### Example: Match let-chains to avoid nested guards

```rust
match item {
    Some(value) if let Ok(qty) = compute(value) => {
        println!("{value} has quantity {qty}");
    }
    _ => {}
}
```

In this pattern:
- only the successful branch stays in the main arm body,
- invalid data falls through to the `_` arm,
- and the pattern matching logic remains flat and readable.

#### Example: Transforming Nested Code to Early Returns

**Before (Deeply Nested):**

```rust
fn process_user_input(input: &str) -> Result<User, Error> {
    if !input.is_empty() {
        let trimmed = input.trim();
        if trimmed.len() >= 3 {
            if let Ok(user) = User::parse(trimmed) {
                if user.is_valid() {
                    Ok(user)
                } else {
                    Err(Error::InvalidUser)
                }
            } else {
                Err(Error::ParseFailed)
            }
        } else {
            Err(Error::TooShort)
        }
    } else {
        Err(Error::EmptyInput)
    }
}
```

**After (Early Returns):**

```rust
fn process_user_input(input: &str) -> Result<User, Error> {
    if input.is_empty() {
        return Err(Error::EmptyInput);
    }

    let trimmed = input.trim();
    if trimmed.len() < 3 {
        return Err(Error::TooShort);
    }

    let user = User::parse(trimmed)?;

    if !user.is_valid() {
        return Err(Error::InvalidUser);
    }

    Ok(user)
}
```

**Using let-else:**

```rust
fn process_user_input(input: &str) -> Result<User, Error> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(Error::EmptyInput);
    }

    let Ok(user) = User::parse(trimmed) else {
        return Err(Error::ParseFailed);
    };

    user.is_valid().then_some(user).ok_or(Error::InvalidUser)
}
```

- **Comments**: Describe what code does NOW, not historical context or implementation details

### Performance Considerations

Performance is paramount, especially in critical paths in decode loops, CUDA launch paths, device-memory movement, and kernel-facing buffer management

- Prefer low or zero-copy solutions where possible
- Follow cache-friendly patterns
- Consider vectorization for amenable problems
- Avoid unnecessary allocations
- Measure your changes before and after
- If you're writing code which is an optimization... prove it!

### Dependency Policy

We avoid gratuitous Cargo dependencies. Every dependency is part of the system's
long-term maintenance surface, and every transitive dependency is code we may
need to audit, reason about, build, and debug.

- Prefer the standard library and existing workspace crates when they are a
  reasonable fit.
- Add a new crate only when it provides clear value that would be expensive,
  risky, or distracting to reproduce locally.
- Keep dependency versions centralised in the root workspace `Cargo.toml`.
  Member crates should use `workspace = true` instead of specifying versions.
- Before adding a dependency, inspect its transitive dependency graph, feature
  defaults, licence, maintenance state, and whether it pulls in async runtimes,
  logging stacks, serialization frameworks, or platform code we do not need.
- Disable default features when practical and enable only the features Eider
  actually uses.
- Avoid dependencies for small helpers, thin wrappers, trivial derive
  conveniences, or one-off algorithms that are clearer to write directly.
- Be especially conservative in the low-level `nvfp4` crate, where dependency choices affect CUDA builds and the entire inference stack.

The goal is not zero dependencies. The goal is a transitive dependency set that
is small enough, stable enough, and intentional enough that we can understand
it.

## Testing

### Testing Philosophy

We follow strict testing practices:

- **Comprehensive Coverage**: Unit, integration, and end-to-end tests
- **Don't Overuse Mocked Behavior**: Tests should exercise real logic where possible
- **Clean Output**: Test output must be pristine to pass
- **Test-Driven Development**: TDD can be a helpful approach - write tests first when practical

### Eider-specific checks

Before handing work back, run the checks that match the change:

```sh
cargo fmt --all
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps
```

For kernel or performance changes, add or update a `micromeasure` benchmark and
compare both correctness and end-to-end decode behaviour. Keep CUDA stream
semantics explicit: non-blocking streams do not synchronize with the default
stream, and host readbacks should be visible in the API boundary.

Document public Rust APIs with rustdoc. Keep FFI declarations private to the
crate unless a stable safe wrapper is intentionally part of the public API.
