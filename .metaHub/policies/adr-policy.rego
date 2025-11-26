# ADR Policy
# Encourages architecture decision records presence for significant changes

package adr_policy

default warn := []

# Warn if repository lacks ADR directory
warn[msg] {
  not adr_dir_exists
  msg := "ADR directory not found (expected 'docs/adr/' or 'adr/'). Add ADRs for significant changes"
}

adr_dir_exists {
  files := input.files
  some f
  f := files[_]
  startswith(f, "docs/adr/")
}

adr_dir_exists {
  files := input.files
  some f
  f := files[_]
  startswith(f, "adr/")
}

