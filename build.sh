mkdir -p bin/bench
mkdir -p obj/bench
mkdir -p bin/ssb
mkdir -p obj/ssb
make ssb/binpack
make ssb/deltabinpack
make ssb/rlebinpack
make bin/ssb/q11r
make bin/ssb/q21r
make bin/ssb/q31r
make bin/ssb/q41r
