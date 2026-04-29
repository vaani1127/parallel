# ─────────────────────────────────────────────────────────────────
# Makefile – Hybrid MPI + OpenMP Parallel Document Similarity Search
# ─────────────────────────────────────────────────────────────────

CC      = mpicc
# -fopenmp  : enables OpenMP pragmas and links libgomp
# -O3       : full optimisation (loop vectorisation, inlining, etc.)
# -Wall     : all warnings
CFLAGS  = -Wall -O3 -std=c99 -fopenmp
LDFLAGS = -lm

TARGET  = docsim
SOURCES = main.c utils.c
HEADERS = utils.h
OBJECTS = $(SOURCES:.c=.o)

# Default dataset parameters
DOCS    = documents.txt
QUERY   = query.txt
DSIZE   = 10          # dict_size  (features per document)
K       = 10          # top-k

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

# ── Basic run ─────────────────────────────────────────────────────
run: $(TARGET)
	mpirun -np 4 ./$(TARGET) $(DSIZE) $(K) $(DOCS) $(QUERY) cosine 2

# ── Similarity metric comparison ──────────────────────────────────
test-metrics: $(TARGET)
	@echo "=== Power ===" && mpirun -np 4 ./$(TARGET) $(DSIZE) $(K) $(DOCS) $(QUERY) power    1
	@echo "=== Cosine ===" && mpirun -np 4 ./$(TARGET) $(DSIZE) $(K) $(DOCS) $(QUERY) cosine   1
	@echo "=== Euclidean ===" && mpirun -np 4 ./$(TARGET) $(DSIZE) $(K) $(DOCS) $(QUERY) euclidean 1
	@echo "=== Manhattan ===" && mpirun -np 4 ./$(TARGET) $(DSIZE) $(K) $(DOCS) $(QUERY) manhattan 1

# ── MPI strong scaling (fixed problem, more ranks, 1 OMP thread) ──
test-mpi-scaling: $(TARGET)
	@echo "--- MPI strong scaling (OMP threads=1) ---"
	@for NP in 1 2 4; do \
	    echo "--- np=$$NP ---"; \
	    mpirun -np $$NP ./$(TARGET) $(DSIZE) $(K) $(DOCS) $(QUERY) cosine 1; \
	done

# ── OpenMP scaling (fixed 1 MPI rank, more OMP threads) ──────────
test-omp-scaling: $(TARGET)
	@echo "--- OpenMP scaling (MPI ranks=1) ---"
	@for T in 1 2 4; do \
	    echo "--- threads=$$T ---"; \
	    mpirun -np 1 ./$(TARGET) $(DSIZE) $(K) $(DOCS) $(QUERY) cosine $$T; \
	done

# ── Hybrid scaling (vary both MPI ranks and OMP threads) ─────────
test-hybrid: $(TARGET)
	@echo "--- Hybrid MPI+OMP scaling ---"
	mpirun -np 1 ./$(TARGET) $(DSIZE) $(K) $(DOCS) $(QUERY) cosine 1
	mpirun -np 2 ./$(TARGET) $(DSIZE) $(K) $(DOCS) $(QUERY) cosine 1
	mpirun -np 2 ./$(TARGET) $(DSIZE) $(K) $(DOCS) $(QUERY) cosine 2
	mpirun -np 4 ./$(TARGET) $(DSIZE) $(K) $(DOCS) $(QUERY) cosine 1
	mpirun -np 4 ./$(TARGET) $(DSIZE) $(K) $(DOCS) $(QUERY) cosine 2

.PHONY: all clean run test-metrics test-mpi-scaling test-omp-scaling test-hybrid
