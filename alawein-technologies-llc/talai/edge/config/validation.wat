;; WebAssembly Text Format for TalAI Edge Validation
;; Compiles to validation.wasm for edge request validation

(module
  ;; Import memory from JavaScript
  (import "env" "memory" (memory 1))

  ;; Export validation function
  (export "validate" (func $validate))
  (export "validateJSON" (func $validateJSON))
  (export "validateAPIKey" (func $validateAPIKey))
  (export "validateResearchRequest" (func $validateResearchRequest))

  ;; Main validation function
  ;; Returns 0 for valid, error code for invalid
  (func $validate (param $ptr i32) (param $len i32) (result i32)
    (local $result i32)

    ;; Check if length is reasonable (1 byte to 10MB)
    (if (i32.or
          (i32.lt_u (local.get $len) (i32.const 1))
          (i32.gt_u (local.get $len) (i32.const 10485760)))
      (return (i32.const 1)))  ;; Error: Invalid length

    ;; Validate JSON structure
    (local.set $result (call $validateJSON (local.get $ptr) (local.get $len)))
    (if (i32.ne (local.get $result) (i32.const 0))
      (return (local.get $result)))

    ;; All validations passed
    (i32.const 0))

  ;; JSON validation
  (func $validateJSON (param $ptr i32) (param $len i32) (result i32)
    (local $i i32)
    (local $char i32)
    (local $depth i32)
    (local $inString i32)
    (local $escaped i32)

    (local.set $i (i32.const 0))
    (local.set $depth (i32.const 0))
    (local.set $inString (i32.const 0))
    (local.set $escaped (i32.const 0))

    ;; Iterate through the buffer
    (loop $loop
      ;; Get current character
      (local.set $char (i32.load8_u (i32.add (local.get $ptr) (local.get $i))))

      ;; Handle escape sequences
      (if (local.get $escaped)
        (then
          (local.set $escaped (i32.const 0)))
        (else
          ;; Check for escape character
          (if (i32.eq (local.get $char) (i32.const 92))  ;; backslash
            (local.set $escaped (i32.const 1)))

          ;; Handle strings
          (if (i32.eq (local.get $char) (i32.const 34))  ;; quote
            (local.set $inString (i32.xor (local.get $inString) (i32.const 1))))

          ;; Track braces/brackets depth when not in string
          (if (i32.eqz (local.get $inString))
            (then
              ;; Opening brace or bracket
              (if (i32.or
                    (i32.eq (local.get $char) (i32.const 123))  ;; {
                    (i32.eq (local.get $char) (i32.const 91)))   ;; [
                (local.set $depth (i32.add (local.get $depth) (i32.const 1))))

              ;; Closing brace or bracket
              (if (i32.or
                    (i32.eq (local.get $char) (i32.const 125))  ;; }
                    (i32.eq (local.get $char) (i32.const 93)))   ;; ]
                (then
                  (local.set $depth (i32.sub (local.get $depth) (i32.const 1)))
                  ;; Check for negative depth (unmatched closing)
                  (if (i32.lt_s (local.get $depth) (i32.const 0))
                    (return (i32.const 2)))))))))  ;; Error: Unmatched brackets

      ;; Increment counter
      (local.set $i (i32.add (local.get $i) (i32.const 1)))

      ;; Continue loop if not at end
      (br_if $loop (i32.lt_u (local.get $i) (local.get $len))))

    ;; Check final state
    (if (i32.ne (local.get $depth) (i32.const 0))
      (return (i32.const 3)))  ;; Error: Unclosed brackets

    (if (local.get $inString)
      (return (i32.const 4)))  ;; Error: Unclosed string

    (i32.const 0))  ;; Valid JSON structure

  ;; API Key validation
  (func $validateAPIKey (param $ptr i32) (param $len i32) (result i32)
    ;; Check length (should be 32-64 characters)
    (if (i32.or
          (i32.lt_u (local.get $len) (i32.const 32))
          (i32.gt_u (local.get $len) (i32.const 64)))
      (return (i32.const 5)))  ;; Error: Invalid API key length

    ;; Check prefix (should start with 'sk_' or 'pk_')
    (if (i32.and
          (i32.ne (i32.load8_u (local.get $ptr)) (i32.const 115))       ;; 's'
          (i32.ne (i32.load8_u (local.get $ptr)) (i32.const 112)))      ;; 'p'
      (return (i32.const 6)))  ;; Error: Invalid API key prefix

    (if (i32.ne (i32.load8_u (i32.add (local.get $ptr) (i32.const 1))) (i32.const 107))  ;; 'k'
      (return (i32.const 6)))

    (if (i32.ne (i32.load8_u (i32.add (local.get $ptr) (i32.const 2))) (i32.const 95))   ;; '_'
      (return (i32.const 6)))

    ;; Validate characters (alphanumeric and underscore only)
    (local $i i32)
    (local $char i32)
    (local.set $i (i32.const 3))

    (loop $loop
      (local.set $char (i32.load8_u (i32.add (local.get $ptr) (local.get $i))))

      ;; Check if alphanumeric or underscore
      (if (i32.and
            (i32.and
              (i32.lt_u (local.get $char) (i32.const 48))   ;; < '0'
              (i32.gt_u (local.get $char) (i32.const 57)))  ;; > '9'
            (i32.and
              (i32.and
                (i32.lt_u (local.get $char) (i32.const 65))   ;; < 'A'
                (i32.gt_u (local.get $char) (i32.const 90)))  ;; > 'Z'
              (i32.and
                (i32.and
                  (i32.lt_u (local.get $char) (i32.const 97))   ;; < 'a'
                  (i32.gt_u (local.get $char) (i32.const 122))) ;; > 'z'
                (i32.ne (local.get $char) (i32.const 95)))))    ;; != '_'
        (return (i32.const 7)))  ;; Error: Invalid character in API key

      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br_if $loop (i32.lt_u (local.get $i) (local.get $len))))

    (i32.const 0))  ;; Valid API key

  ;; Research request validation
  (func $validateResearchRequest (param $ptr i32) (param $len i32) (result i32)
    ;; First validate as JSON
    (local $result i32)
    (local.set $result (call $validateJSON (local.get $ptr) (local.get $len)))
    (if (i32.ne (local.get $result) (i32.const 0))
      (return (local.get $result)))

    ;; Check for required fields
    ;; This is a simplified check - in production, use a proper JSON parser

    ;; Check for "query" field
    (if (i32.eqz (call $containsString
                   (local.get $ptr)
                   (local.get $len)
                   (i32.const 0x79726575716)  ;; "query" in hex
                   (i32.const 5)))
      (return (i32.const 8)))  ;; Error: Missing query field

    ;; Check for "userId" or "apiKey" field
    (if (i32.and
          (i32.eqz (call $containsString
                     (local.get $ptr)
                     (local.get $len)
                     (i32.const 0x646972657375)  ;; "userId"
                     (i32.const 6)))
          (i32.eqz (call $containsString
                     (local.get $ptr)
                     (local.get $len)
                     (i32.const 0x79654b697061)  ;; "apiKey"
                     (i32.const 6))))
      (return (i32.const 9)))  ;; Error: Missing authentication

    (i32.const 0))  ;; Valid research request

  ;; Helper function to check if buffer contains a string
  (func $containsString (param $ptr i32) (param $len i32) (param $needle i64) (param $needleLen i32) (result i32)
    (local $i i32)
    (local $j i32)
    (local $found i32)

    (local.set $i (i32.const 0))

    (loop $outer
      (local.set $found (i32.const 1))
      (local.set $j (i32.const 0))

      ;; Check if we have enough space left
      (if (i32.gt_u (i32.add (local.get $i) (local.get $needleLen)) (local.get $len))
        (return (i32.const 0)))

      ;; Compare bytes
      (loop $inner
        (if (i32.ne
              (i32.load8_u (i32.add (local.get $ptr) (i32.add (local.get $i) (local.get $j))))
              (i32.wrap_i64 (i64.shr_u (local.get $needle) (i64.mul (i64.extend_i32_u (local.get $j)) (i64.const 8)))))
          (then
            (local.set $found (i32.const 0))
            (br $inner)))

        (local.set $j (i32.add (local.get $j) (i32.const 1)))
        (br_if $inner (i32.lt_u (local.get $j) (local.get $needleLen))))

      ;; If found, return 1
      (if (local.get $found)
        (return (i32.const 1)))

      ;; Continue searching
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br_if $outer (i32.lt_u (local.get $i) (local.get $len))))

    (i32.const 0))  ;; Not found

  ;; Memory management
  (global $heapBase (mut i32) (i32.const 0x10000))

  (func $malloc (param $size i32) (result i32)
    (local $ptr i32)
    (local.set $ptr (global.get $heapBase))
    (global.set $heapBase (i32.add (local.get $ptr) (local.get $size)))
    (local.get $ptr))

  (func $free (param $ptr i32)
    ;; Simple allocator - no free in this implementation
    (nop))
)