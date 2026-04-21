// RUN: aie-opt --split-input-file --verify-diagnostics %s
//
// M3: conduit.put_memref / conduit.get_memref verifier regression tests.
//
// Section 1: valid put_memref + get_memref — no errors expected.
// Section 2: put_memref num_elems=0 — must be > 0.
// Section 3: get_memref num_elems=0 — must be > 0.
// Section 4: put_memref offsets/sizes length mismatch.
// Section 5: get_memref strides/sizes length mismatch.
// Section 6: put_memref sizes[i]=0 — must be > 0.
// Section 7: get_memref num_elems != product(sizes).

// -----

// Section 1: PASS — valid put_memref and get_memref with consistent attributes.

func.func @valid_put_get() {
  conduit.put_memref {name = @ch, num_elems = 64 : i64,
                      offsets = array<i64: 0, 0>,
                      sizes   = array<i64: 8, 8>,
                      strides = array<i64: 16, 1>}
  conduit.get_memref {name = @ch, num_elems = 256 : i64,
                      offsets = array<i64: 0>,
                      sizes   = array<i64: 256>,
                      strides = array<i64: 1>}
  return
}

// -----

// Section 2: put_memref num_elems=0 — must be > 0.

func.func @put_memref_zero_elems() {
  // expected-error @+1 {{'conduit.put_memref' op num_elems must be > 0, got 0}}
  conduit.put_memref {name = @ch, num_elems = 0 : i64,
                      offsets = array<i64: 0>,
                      sizes   = array<i64: 0>,
                      strides = array<i64: 1>}
  return
}

// -----

// Section 3: get_memref num_elems=0 — must be > 0.

func.func @get_memref_zero_elems() {
  // expected-error @+1 {{'conduit.get_memref' op num_elems must be > 0, got 0}}
  conduit.get_memref {name = @ch, num_elems = 0 : i64,
                      offsets = array<i64: 0>,
                      sizes   = array<i64: 0>,
                      strides = array<i64: 1>}
  return
}

// -----

// Section 4: put_memref offsets/sizes length mismatch.

func.func @put_memref_length_mismatch() {
  // expected-error @+1 {{'conduit.put_memref' op offsets length (1) does not match sizes length (2)}}
  conduit.put_memref {name = @ch, num_elems = 64 : i64,
                      offsets = array<i64: 0>,
                      sizes   = array<i64: 8, 8>,
                      strides = array<i64: 16, 1>}
  return
}

// -----

// Section 5: get_memref strides/sizes length mismatch.

func.func @get_memref_strides_mismatch() {
  // expected-error @+1 {{'conduit.get_memref' op strides length (1) does not match sizes length (2)}}
  conduit.get_memref {name = @ch, num_elems = 64 : i64,
                      offsets = array<i64: 0, 0>,
                      sizes   = array<i64: 8, 8>,
                      strides = array<i64: 1>}
  return
}

// -----

// Section 6: put_memref sizes[i]=0 — must be > 0.

func.func @put_memref_zero_size() {
  // expected-error @+1 {{'conduit.put_memref' op sizes[1] must be > 0, got 0}}
  conduit.put_memref {name = @ch, num_elems = 8 : i64,
                      offsets = array<i64: 0, 0>,
                      sizes   = array<i64: 8, 0>,
                      strides = array<i64: 1, 1>}
  return
}

// -----

// Section 7: get_memref num_elems != product(sizes).

func.func @get_memref_product_mismatch() {
  // expected-error @+1 {{'conduit.get_memref' op num_elems (100) does not match product of sizes (64)}}
  conduit.get_memref {name = @ch, num_elems = 100 : i64,
                      offsets = array<i64: 0, 0>,
                      sizes   = array<i64: 8, 8>,
                      strides = array<i64: 16, 1>}
  return
}
